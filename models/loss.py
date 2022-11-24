import torch.nn.functional as F
import torch

def model_loss_train(disp_ests, disp_gts, img_masks, disp_gt_low, mask_low):
    weights = [1.0, 0.3, 0.5, 0.3]
    all_losses = []
    disp_gt_low_index = disp_gt_low * mask_low.float() / 4.
    disp_gt_low_index = torch.floor(disp_gt_low_index).type(torch.long)
    gt_low_probability = torch.zeros_like(disp_ests['probability']).scatter_(1, disp_gt_low_index.unsqueeze(1), 1)
    loss_probability = - torch.sum(gt_low_probability * torch.log(disp_ests['probability']), dim=1)
    all_losses.append(torch.mean(loss_probability[mask_low]))

    for disp_est, disp_gt, weight, mask_img in zip(disp_ests['disparity'], disp_gts, weights, img_masks):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask_img], disp_gt[mask_img], size_average=True))
    return sum(all_losses)

def model_loss_test(disp_ests, disp_gts,img_masks):
    weights = [1.0]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts, weights, img_masks):
        all_losses.append(weight * F.l1_loss(disp_est[mask_img], disp_gt[mask_img], size_average=True))
    return sum(all_losses)