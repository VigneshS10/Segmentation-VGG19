import torch


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target).float().sum(0)
        acc = correct * 100 / batch_size
    return [acc], pred, target


