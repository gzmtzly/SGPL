from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch

class SGPL(nn.Module):
    def __init__(self, alpha_neg=-2, beta_neg=6, clip=0.05, ls=0.1, disable_torch_grad_focal_loss=True):
        super(SGPL, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.alpha_neg = alpha_neg
        self.beta_neg = beta_neg
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.ls = ls

    def forward(self, input, target, K_ci):
        """
        Parameters
        ----------
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size, 1)
        """
        self.targets_classes = torch.zeros_like(input).scatter_(1, target.long().unsqueeze(1), 1)  # one-hot label
        num_classes = input.size()[-1]

        target = target.view(-1, 1)
        log_preds = self.logsoftmax(input)
        prob = torch.exp(log_preds)
        pt = prob.gather(1, target)   # pt

        K_ci_ = np.array([K_ci[target[i]] for i in range(len(target))])
        K_ci_ = np.reshape(K_ci_[np.newaxis, :], (len(target), -1))
        K_ci_ = Variable(torch.from_numpy(K_ci_).long()).cuda()

        phi_pt = (pt + K_ci_ * self.clip).clamp(max=1)  

        lop_phi_pt = torch.log(phi_pt)   # log_phi(pt)
        pusai = (self.alpha_neg*(1-pt) + self.beta_neg) * K_ci_   # pusai_pt
        one_sided_w = torch.pow(1 - pt, pusai)  # (1-pt)^pusai_pt

        loss = lop_phi_pt * one_sided_w
        targets_loss = torch.zeros_like(input).scatter_(1, target, loss)
        loss = targets_loss.add(log_preds.mul(1 - self.targets_classes))

        if self.ls > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.ls).add(self.ls / num_classes)
        loss = loss.mul(self.targets_classes)
        loss = loss.sum(dim=-1)

        return -torch.mean(loss)


