import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# class mse_loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.loss_fn = nn.MSELoss()
#     def forward(self, output, target):
#         return self.loss_fn(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


def l1_loss(output, target):
    return F.l1_loss(output, target)


def loss_predict_loss(out, target, pred_loss):
    target_loss = F.mse_loss(out, target, reduction='none')
    return torch.sum(target_loss) / (
            target.shape[0] * target.shape[1] * target.shape[2] * target.shape[3]), LossPredLoss(pred_loss,
                                                                                                 target_loss)


def pin_loss(q_upper, q_lower, target):
    q_lo_loss = PinballLoss(0.05)
    q_hi_loss = PinballLoss(0.95)
    loss = q_lo_loss(q_lower, target) + q_hi_loss(q_upper, target)
    return loss


def SampleLossPredLoss(input, target, margin=1.0, reduction='mean'):
    # input: (b, w * h)

    b = input.shape[0]
    target = target.detach()
    target = target.view(b, -1)
    target = torch.mean(target, dim=1)
    input = input.view(b, -1)
    input = torch.mean(input, dim=1)
    assert input.shape[0] % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    input = (input - input.flip(0))[
            :input.shape[0] // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:target.shape[0] // 2]
    target = target.detach()
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / (input.size(0))  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    return loss


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    # input: (b, w * h)

    b = input.shape[0]
    target = target.view(b, -1)
    input = input.view(b, -1)
    assert input.shape[1] % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(1).shape
    index_shuffle = torch.randperm(input.shape[1])

    input = input[:, index_shuffle]
    target = target[:, index_shuffle]
    input = (input - input.flip(1))[:,
            :input.shape[1] // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(1))[:, :target.shape[1] // 2]
    target = target.detach()
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / (input.size(0) * input.size(1))  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    return loss


def pin_loss2(q_lower, q_uper, out, target):
    q_lo_loss = PinballLoss(0.05)
    q_hi_loss = PinballLoss(0.95)
    loss = q_lo_loss(q_lower, target) + q_hi_loss(q_uper, target) + mse_loss(out, target)
    return loss


def mse_var_loss(output, target, variance, weight=1):
    variance = weight * variance
    loss1 = torch.mul(torch.exp(-variance), (output - target) ** 2)
    loss2 = variance
    loss = .5 * (loss1 + loss2)
    return loss.mean()

def mse_var_loss2(output, target, variance, var_weight):
    # print((1-var_weight).max(), (1-var_weight).min())
    variance = variance * torch.clamp(var_weight, min=1e-2, max=1)
    loss1 = torch.mul(torch.exp(-variance), (output - target) ** 2)
    loss2 = variance
    loss = .5 * (loss1 + loss2)
    return loss.mean()


def mse_var_loss_sample(output, target, variance, weight=1):
    # variance = 4 * variance
    target_loss = (output - target) ** 2
    loss1 = torch.mul(torch.exp(-variance), target_loss)
    loss2 = variance
    loss3 = SampleLossPredLoss(variance, target_loss,reduction='mean')
    var_loss = .5 * (loss1 +  loss2)

    return var_loss.mean() + loss3



class MSE_VAR(nn.Module):
    def __init__(self, var_weight):
        super(MSE_VAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, results, label):
        mean, var = results['mean'], results['var']
        var = self.var_weight * var

        loss1 = torch.mul(torch.exp(-var), (mean - label) ** 2)
        loss2 = var
        loss = .5 * (loss1 + loss2)
        return loss.mean()


class PinballLoss():

    def __init__(self, quantile=0.10, reduction='mean'):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction

    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1 - self.quantile) * (abs(error)[bigger_index])

        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
