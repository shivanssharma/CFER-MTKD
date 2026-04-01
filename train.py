def train(epoch, train_loader, learner):
    print("Epoch:%2d" % epoch, end="  ")
    learner.train()

    for i, (inputData, (expr_labels, au_labels)) in enumerate(train_loader):
        inputData = inputData.cuda()
        expr_labels = expr_labels.cuda()
        au_labels = au_labels.cuda()

        # Pass both labels to learner
        loss, output = learner.learn(inputData, (expr_labels, au_labels))

        if i == len(train_loader) - 1:
            print(
                "[{}/{}], LR {:.4e}, Loss: {:.4e}".format(
                    i + 1,
                    len(train_loader),
                    learner.optimizer.param_groups[0]["lr"],
                    loss.item(),
                ),
                end="  ",
            )


