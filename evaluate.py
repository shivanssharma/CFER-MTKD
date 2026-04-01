import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, multilabel_confusion_matrix, confusion_matrix
from torch.cuda.amp import autocast
from models.repvgg import repvgg_model_convert

# ---------------------------
# Compound Expression Setup
# ---------------------------
compound_classes = [
    "happilysurprised", "happilydisgusted", "sadlyfearful", "sadlyangry",
    "fearfullydisgusted", "angrilysurprised", "angrilydisgusted",
    "disgustedlysurprised", "happilyfearful", "happilysad",
    "sadlysurprised", "sadlydisgusted", "fearfullyangry", "fearfullysurprised"
]


compound_map = {
    (1, 0, 0, 1, 0, 0): 0,   # happilysurprised → Surprise + Happiness
    (0, 0, 1, 1, 0, 0): 1,   # happilydisgusted → Disgust + Happiness
    (0, 1, 0, 0, 1, 0): 2,   # sadlyfearful → Fear + Sadness
    (0, 0, 0, 0, 1, 1): 3,   # sadlyangry → Sadness + Anger
    (0, 1, 1, 0, 0, 0): 4,   # fearfullydisgusted → Fear + Disgust
    (1, 0, 0, 0, 0, 1): 5,   # angrilysurprised → Surprise + Anger
    (0, 0, 1, 0, 0, 1): 6,   # angrilydisgusted → Disgust + Anger
    (1, 0, 1, 0, 0, 0): 7,   # disgustedlysurprised → Surprise + Disgust
    (0, 1, 0, 1, 0, 0): 8,   # happilyfearful → Fear + Happiness
    (0, 0, 0, 1, 1, 0): 9,   # happilysad → Happiness + Sadness
    (1, 0, 0, 0, 1, 0): 10,  # sadlysurprised → Surprise + Sadness
    (0, 0, 1, 0, 1, 0): 11,  # sadlydisgusted → Disgust + Sadness
    (0, 1, 0, 0, 0, 1): 12,  # fearfullyangry → Fear + Anger
    (1, 1, 0, 0, 0, 0): 13   # fearfullysurprised → Surprise + Fear
}

# Track best epoch for student
best_epoch_student = None
best_acc_student = -1
best_compound_acc = None


# ---------------------------
# Compute mAP for multilabel
# ---------------------------
def compute_mAP(labels, outputs):
    APs = []
    for j in range(labels.shape[1]):
        new_AP = average_precision_score(labels[:, j], outputs[:, j])
        APs.append(new_AP)
    mAP = np.mean(APs)
    return APs, mAP


# ---------------------------
# Evaluation metrics
# ---------------------------
def test(outputs, targets, is_au=False):
    idxs = np.sum(targets == 1, axis=1).astype(int)
    sorted_outputs = np.sort(-outputs, axis=1)
    thr = -sorted_outputs[range(len(targets)), idxs].reshape(len(sorted_outputs), 1)

    preds = np.zeros(outputs.shape, dtype=np.int64)
    preds[outputs > thr] = 1

    APs, mAP = compute_mAP(targets, outputs)
    of1 = f1_score(targets, preds, average="micro")
    cf1 = f1_score(targets, preds, average="macro")
    acc = accuracy_score(targets, preds)

    # --------------------------
    # Compound (Expr) branch
    # --------------------------
    if not is_au:
        y_true = []
        y_pred = []
        for t, p in zip(targets, preds):
            t_tuple = tuple(t.tolist())
            if t_tuple in compound_map:
                y_true.append(compound_map[t_tuple])
                p_tuple = tuple(p.tolist())
                if p_tuple in compound_map:
                    y_pred.append(compound_map[p_tuple])
                else:
                    y_pred.append(-1)  # invalid compound prediction
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        valid_mask = (y_pred != -1)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) > 0:
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(compound_classes)))
            row_sums = cm.sum(axis=1).astype(float)
            diag = np.diag(cm).astype(float)
            per_class_acc = np.divide(diag, row_sums, out=np.zeros_like(diag, dtype=float), where=(row_sums != 0))
            mean_diag = np.mean(per_class_acc) * 100
        else:
            mean_diag = 0.0

    # --------------------------
    # AU branch (multi-label)
    # --------------------------
    else:
        # Compute mean diagonal as average AU accuracy (per AU)
        cm_list = multilabel_confusion_matrix(targets, preds)
        per_au_acc = []
        for cm in cm_list:
            tn, fp, fn, tp = cm.ravel()
            total = tp + fn  # all actual positives
            acc_per_class = 100.0 * (tp / total) if total > 0 else 0.0
            per_au_acc.append(acc_per_class)
        mean_diag = np.mean(per_au_acc)

    print("  mAP: {:.2f}  OF1: {:.2f}  CF1: {:.2f}  Acc: {:.2f}  Mean Diag: {:.2f}%".format(
        mAP * 100, of1 * 100, cf1 * 100, acc * 100, mean_diag))

    return APs, mAP, of1, cf1, acc, None, mean_diag, preds

# ---------------------------
# Full Evaluation Function
# ---------------------------
def evaluate(eval_loader, model, epoch=None, is_student=False):
    global best_epoch_student, best_acc_student, best_compound_acc

    print(f"\n--- Evaluation (Epoch {epoch if epoch is not None else '?'}) ---")

    if hasattr(model, "repvgg_flag") and model.repvgg_flag:
        deploy_model = repvgg_model_convert(model)
    else:
        deploy_model = model

    deploy_model.eval()
    outputs_expr, targets_expr = [], []
    outputs_au, targets_au = [], []

    for imgs, targets in eval_loader:
        imgs = imgs.cuda()
        if isinstance(targets, (list, tuple)):
            expr_labels, au_labels = targets
            expr_labels, au_labels = expr_labels.cuda(), au_labels.cuda()
        else:
            expr_labels = targets.cuda()
            au_labels = None

        with torch.cuda.amp.autocast():
            outputs = deploy_model(imgs)
            if isinstance(outputs, (list, tuple)):
                out_expr = torch.sigmoid(outputs[0].detach())
                out_au = torch.sigmoid(outputs[1].detach())
            else:
                out_expr = torch.sigmoid(outputs.detach())
                out_au = None

        outputs_expr.append(out_expr.cpu().numpy())
        targets_expr.append(expr_labels.cpu().numpy())
        if out_au is not None and au_labels is not None:
            outputs_au.append(out_au.cpu().numpy())
            targets_au.append(au_labels.cpu().numpy())

    # Expression metrics
    outputs_expr = np.concatenate(outputs_expr)
    targets_expr = np.concatenate(targets_expr)
    print("→ Expr:", end="")
    expr_metrics = test(outputs_expr, targets_expr)
    _, _, _, _,acc_expr, _, mean_diag_expr, preds_expr = expr_metrics

    # 🔹 Only for STUDENT: Track best epoch + compute compound accuracy
    if is_student:
        if acc_expr > best_acc_student:
            best_acc_student = acc_expr
            best_epoch_student = epoch

            # Compute compound expression accuracy
            compound_correct = np.zeros(len(compound_classes))
            compound_total = np.zeros(len(compound_classes))

            for t, p in zip(targets_expr, preds_expr):
                key = tuple(t.tolist())
                if key in compound_map:
                    idx = compound_map[key]
                    compound_total[idx] += 1
                    if np.array_equal(t, p):
                        compound_correct[idx] += 1

            compound_acc = np.divide(compound_correct, compound_total,
                                     out=np.zeros_like(compound_correct),
                                     where=compound_total > 0)
            best_compound_acc = compound_acc

    # AU metrics
    if outputs_au and targets_au:
        outputs_au = np.concatenate(outputs_au)
        targets_au = np.concatenate(targets_au)
        print("→  AU :", end="")
        _ = test(outputs_au, targets_au, is_au=True)

    # 🔹 At end of training, print best compound accuracies
    if is_student and epoch is not None and epoch > 0 and (epoch + 1) % 80 == 0:
        print(f"\n>>> Best Epoch for Student: {best_epoch_student}")
        print(f">>> Best Accuracy: {best_acc_student * 100:.2f}%")
        print(">>> Compound Expression Accuracy at Best Epoch:")
        for cls, acc in zip(compound_classes, best_compound_acc):
            print(f"{cls}: {acc * 100:.2f}%")
        print(f"Mean Compound Accuracy: {np.nanmean(best_compound_acc) * 100:.2f}%")

    selected_classes = [
        "happilysurprised", "happilydisgusted", "sadlyfearful", "sadlyangry",
        "fearfullydisgusted", "angrilysurprised", "angrilydisgusted",
        "disgustedlysurprised", "happilyfearful", "happilysad",
        "sadlysurprised", "sadlydisgusted", "fearfullyangry", "fearfullysurprised"
    ]

    # Map selected class names to compound indices
    selected_indices = [compound_classes.index(cls) for cls in selected_classes]

    # Convert all predictions and targets to class indices
    true_class_indices = []
    pred_class_indices = []

    for t, p in zip(targets_expr, preds_expr):
        t_tuple = tuple(t.tolist())
        p_tuple = tuple(p.tolist())
        if t_tuple in compound_map and p_tuple in compound_map:
            true_class_indices.append(compound_map[t_tuple])
            pred_class_indices.append(compound_map[p_tuple])

    true_class_indices = np.array(true_class_indices)
    pred_class_indices = np.array(pred_class_indices)

    # Only keep samples belonging to selected subset
    subset_mask = np.isin(true_class_indices, selected_indices)
    true_subset = true_class_indices[subset_mask]
    pred_subset = pred_class_indices[subset_mask]

    # Compute per-class accuracy for the subset
    subset_correct = []
    subset_total = []

    for idx in selected_indices:
        mask = true_subset == idx
        total = np.sum(mask)
        correct = np.sum(pred_subset[mask] == idx)
        subset_total.append(total)
        subset_correct.append(correct)

    subset_correct = np.array(subset_correct)
    subset_total = np.array(subset_total)

    per_class_subset_acc = np.divide(subset_correct, subset_total,
                                    out=np.zeros_like(subset_correct, dtype=float),
                                    where=subset_total > 0)
    mean_per_class_acc = np.mean(per_class_subset_acc) * 100
    overall_subset_acc = np.sum(subset_correct) / np.sum(subset_total) * 100

    # Print subset results
    print("\n>>> Subset (7-class) Compound Accuracy Summary:")
    for cls, acc in zip(selected_classes, per_class_subset_acc):
        print(f"{cls}: {acc * 100:.2f}%")
    print(f"Mean Subset Accuracy (per class): {mean_per_class_acc:.2f}%")
    print(f"Overall Subset Expression Accuracy: {overall_subset_acc:.2f}%")
    print("-------------------------------------------------------------------")

    return expr_metrics

