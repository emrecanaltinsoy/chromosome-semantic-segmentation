import torch
import torch.nn as nn

class evals(nn.Module):
    def __init__(self):
        super(evals, self).__init__()
        pass

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        dsc_arr = []
        for im_num in range(y_pred.shape[0]):
            im_arr = []
            for channel_num in range(y_pred.shape[1]):
                y_pred_flat = y_pred[im_num, channel_num].contiguous().view(-1)
                y_true_flat = y_true[im_num, channel_num].contiguous().view(-1)
                TP = (y_pred_flat * y_true_flat).sum().detach().cpu().numpy()
                TN = ((1.-y_pred_flat) * (1.-y_true_flat)).sum().detach().cpu().numpy()
                FP = (y_pred_flat * (1.-y_true_flat)).sum().detach().cpu().numpy()
                FN = ((1.-y_pred_flat) * y_true_flat).sum().detach().cpu().numpy()

                im_arr.append([TP,TN,FP,FN])
            dsc_arr.append(im_arr)
        return dsc_arr


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/17cbfe0b68148d129a3ddaa227696496
    @author: wassname
    """
    intersection= (y_true * y_pred).abs().sum()
    sum_ = torch.sum(y_true.abs() + y_pred.abs())
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return 1 - jac

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred_f = y_pred.contiguous().view(-1)
        y_true_f = y_true.contiguous().view(-1)
        intersection = (y_pred_f * y_true_f).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred_f.sum() + y_true_f.sum() + self.smooth
        )
        return 1. - dsc
