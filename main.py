'''
Implementation of MTKD: Multi Label Knowledge Distillation of Compound Expression using Multi-Task Learning        
Authors: Shivansh Sharma, K. Vengata Krishna, Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 01-04-2026
Email: shivanshsharma5102003@gmail.com
'''


import argparse
import os
import warnings
from visualize import run_visualizations
import torch
import torch.nn as nn
from mmcv import Config
from torch.utils.tensorboard import SummaryWriter

import criterion
import models
import dataloader
from evaluate import evaluate
from learner import Learner, Learner_KD
from tools.add_weight_decay import add_weight_decay
from tools.set_up_seed import set_seed
from train import train
from criterion.loss.asl_loss import AsymmetricLoss
from torch import optim

def build_optimizer_and_scheduler(model_parameters, cfg, train_loader_len):
    opt_type = getattr(cfg, "opt_type", "adam").lower()
    weight_decay = getattr(cfg, "weight_decay", 1e-4)

    if opt_type == "sgd":
        optimizer = optim.SGD(
            model_parameters,
            lr=cfg.lr_s if hasattr(cfg, "lr_s") else 0.01,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=False
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epoch_s if hasattr(cfg, "max_epoch_s") else 100
        )
    else:  # default to Adam + OneCycle
        max_lr = getattr(cfg, "onecycle_max_lr", 3e-4)   
        base_lr = getattr(cfg, "lr_s", 1e-4)
        optimizer = optim.Adam(
            model_parameters,
            lr=base_lr,
            weight_decay=weight_decay
        )

        steps_per_epoch = max(1, train_loader_len)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=cfg.max_epoch_s,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,            
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4,
        )
    return optimizer, scheduler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default=None, type=str, help="path of cfg file")
    parser.add_argument("--data_root", default=None, type=str, help="path of data files")
    args = parser.parse_args()
    return args


def main(args):
    cfg = Config.fromfile(args.cfg_file)
    print(
        "\nDataset:%s T:%s (lr:%e %d in %d) ===%s===> S:%s (lr:%e %d in %d) | img size:%d  batch size:%d"
        % (
            cfg.dataset,
            cfg.model_t,
            cfg.lr_t,
            cfg.stop_epoch_t,
            cfg.max_epoch_t,
            cfg.criterion_t2s_para["name"],
            cfg.model_s,
            cfg.lr_s,
            cfg.stop_epoch_s,
            cfg.max_epoch_s,
            cfg.img_size,
            cfg.batch_size,
        )
    )
    torch.cuda.empty_cache()
    set_seed(0)

    writer = SummaryWriter(
        comment=" %s T model:%s (lr:%e %din%d) =%s=> S model:%s (lr:%e %din%d)|%d %d"
        % (
            cfg.dataset,
            cfg.model_t,
            cfg.lr_t,
            cfg.stop_epoch_t,
            cfg.max_epoch_t,
            cfg.criterion_t2s_para["name"],
            cfg.model_s,
            cfg.lr_s,
            cfg.stop_epoch_s,
            cfg.max_epoch_s,
            cfg.img_size,
            cfg.batch_size,
        )
    )

    train_loader, test_loader = dataloader.__dict__[cfg.dataset](cfg, args.data_root)

    weight_decay = 1e-4

    # teacher model
    num_expr_classes = train_loader.num_classes
    num_au_classes = getattr(cfg, "num_au_classes", 37)
    if "swin" not in cfg.model_t:
        model_teacher = models.__dict__[cfg.model_t](train_loader.num_classes, pretrained=True,num_au_classes=num_au_classes)
    else:
        model_teacher = models.__dict__[cfg.model_t](
            train_loader.num_classes, pretrained=True, img_size=cfg.img_size,num_au_classes=num_au_classes
        )
    model_teacher = nn.DataParallel(model_teacher).cuda()
    parameters_t = add_weight_decay(model_teacher, weight_decay)

    # student model
    if "swin" not in cfg.model_s:
        model_student = models.__dict__[cfg.model_s](train_loader.num_classes, pretrained=True,num_au_classes=num_au_classes)
    else:
        model_student = models.__dict__[cfg.model_s](
            train_loader.num_classes, pretrained=True, img_size=cfg.img_size,num_au_classes=num_au_classes
        )
    model_student = nn.DataParallel(model_student).cuda()
    parameters_s = add_weight_decay(model_student, weight_decay)

    criterion_t = criterion.BCE()
    criterion_t_expr =criterion.BCE()
    criterion_t_au =criterion.BCE()
    lambda_au = getattr(cfg, "lambda_au", 0.5)
    # teacher training
    best_acc_teacher = 0.0
    best_epoch_teacher = 0

    if not cfg.teacher_pretrained:
        optimizer_t, scheduler_t = build_optimizer_and_scheduler(
            parameters_t, cfg, len(train_loader)
        )
        model_teacher.train()
        for epoch in range(cfg.max_epoch_t):
            if epoch >= cfg.stop_epoch_t:
                break
            for imgs, (expr_labels, au_labels) in train_loader:
                imgs = imgs.cuda()
                expr_labels = expr_labels.cuda()
                au_labels = au_labels.cuda()

                logits_expr, logits_au = model_teacher(imgs)

                loss_expr = criterion_t_expr(logits_expr, expr_labels)
                loss_au = criterion_t_au(logits_au, au_labels)
                if cfg.training_mode == "STL":
                    loss = loss_expr
                elif cfg.training_mode == "MTL":
                    loss = loss_expr + cfg.lambda_au * loss_au
                else:
                    raise ValueError(f"Unknown training mode: {cfg.training_mode}")

                optimizer_t.zero_grad()
                loss.backward()
                optimizer_t.step()
                try:
                    scheduler_t.step()
                except Exception:
                    # safe fallback if scheduler expects per-epoch stepping
                    pass
            model_teacher.eval()
            AP, mAP, of1, cf1, acc, cm, mean_diag,preds = evaluate(test_loader, model_teacher,epoch=epoch,is_student=False)
            writer.add_scalar("Teacher mAP", mAP, epoch)
            writer.add_scalar("Teacher OF1", of1, epoch)
            writer.add_scalar("Teacher CF1", cf1, epoch)
            writer.add_scalar("Teacher Acc", acc, epoch)
            writer.add_scalar("Teacher Diag", mean_diag, epoch)

            if acc > best_acc_teacher:
                best_acc_teacher = acc
                best_epoch_teacher = epoch

                save_path = f"pretrained_models/best_teacher_{cfg.model_t}_{cfg.dataset}_{cfg.img_size}.pth"
                torch.save(model_teacher.state_dict(), save_path)

                print(f">>> New BEST Teacher Model Saved at Epoch {epoch} | Acc: {acc*100:.2f}%")        
    else:
        model_teacher.load_state_dict(
            torch.load(
                "pretrained_models/best_teacher_%s_%s_%d.pth"
                % (cfg.model_t, cfg.dataset, cfg.img_size)
            )
        )

    best_teacher_path = f"pretrained_models/best_teacher_{cfg.model_t}_{cfg.dataset}_{cfg.img_size}.pth"

    if os.path.exists(best_teacher_path):
        print(f"\nLoading BEST Teacher from: {best_teacher_path}")
        model_teacher.load_state_dict(torch.load(best_teacher_path))
        print(f"Best Teacher Epoch: {best_epoch_teacher}  Acc: {best_acc_teacher*100:.2f}%")
    else:
        print("Best teacher checkpoint not found, using last epoch weights")    
    model_teacher.eval()
    print("Before distillation, evaluate teacher model and student model firstly:")
    epoch=cfg.max_epoch_t - 1
    _, mAP_t, of1_t, cf1_t, acc_t, cm_t, mean_diag_t,preds = evaluate(test_loader, model_teacher,epoch=epoch,is_student=False)
    evaluate(test_loader, model_student,epoch=epoch)
    print("Finished!\n")

    criterion_s = criterion.BCE()
    criterion_t2s = criterion.distiller_zoo.BaseDistiller(**cfg.criterion_t2s_para["para"])
    
    optimizer_s, scheduler_s = build_optimizer_and_scheduler(
        parameters_s, cfg, len(train_loader)
    )
    learner_s = Learner_KD(
        model_teacher, model_student, criterion_s, criterion_t2s, optimizer_s, scheduler_s,cfg=cfg
    )
    best_acc_student = 0.0
    best_epoch_student = 0
    for epoch in range(cfg.max_epoch_s):
        if epoch >= cfg.stop_epoch_s:
            break
        train(epoch, train_loader, learner_s)
        AP, mAP, of1, cf1, acc, cm, mean_diag,preds = evaluate(test_loader, model_student,epoch=epoch,is_student=True)
        writer.add_scalar("Student mAP", mAP, epoch)
        writer.add_scalar("Student OF1", of1, epoch)
        writer.add_scalar("Student CF1", cf1, epoch)
        writer.add_scalar("Student Acc", acc, epoch)
        writer.add_scalar("Student Diag", mean_diag, epoch)

        if acc > best_acc_student:
            best_acc_student = acc
            best_epoch_student = epoch

            save_path = f"pretrained_models/best_student_{cfg.model_s}_{cfg.dataset}_{cfg.img_size}.pth"
            torch.save(model_student.state_dict(), save_path)

            print(f">>> New BEST Student Model Saved at Epoch {epoch} | Acc: {acc*100:.2f}%")



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    if not os.path.exists("runs"):
        os.mkdir("runs")
    args = get_args()
    main(args)

 