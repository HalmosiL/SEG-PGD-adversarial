
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def weight_correct(result, target, criterion, miscls_loss_lamb, ignore_label=255):
    bs = len(result)
    result_max = result.max(1)[1]
    correct_ind = ((result_max==target) & (target!=ignore_label)).reshape(bs, -1)
    false_ind   = ((result_max!=target) & (target!=ignore_label)).reshape(bs, -1)
    result_ = result.permute((0, 2, 3, 1))
    result_ = result_.reshape((bs, -1, result_.shape[-1]))
    target  = target.reshape(bs, -1)

    num_posi = (correct_ind).sum(-1)
    num_nega = (false_ind).sum(-1)
    num_pixels = num_posi + num_nega

    loss_bs = []
    for i in range(bs):
        loss = criterion(result_[i], target[i])
        correct_loss = loss[correct_ind[i]].mean() * num_posi[i] / num_pixels[i]
        false_loss = loss[false_ind[i]].mean() * num_nega[i] / num_pixels[i]
        loss_bs.append( miscls_loss_lamb * false_loss + (1. - miscls_loss_lamb) * correct_loss)

    loss = torch.mean(torch.stack(loss_bs))
    return loss

