from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
import gc
import time
import timm

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained =  True
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]

class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL
        y4, y8, y16, y32 = featR

        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)
        
        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)

        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)

        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]

class channelAtt(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1))

        self.weight_init()

    def forward(self, cv, im):

        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att)*cv
        return cv

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),)


        self.feature_att_8 = channelAtt(in_channels*2, 96)
        self.feature_att_16 = channelAtt(in_channels*4, 192)
        self.feature_att_up_8 = channelAtt(in_channels*2, 96)

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, imgs[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, imgs[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg(conv1)
        conv1 = self.feature_att_up_8(conv1, imgs[1])

        conv = self.conv1_up(conv1)

        return conv


class hourglass_att(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_att, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up_ = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv_final = nn.Conv3d(in_channels, 1, 3, 1, 1, bias=False)

        self.agg = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.feature_att_16 = channelAtt(in_channels*2, 192)
        self.feature_att_32 = channelAtt(in_channels*4, 160)
        self.feature_att_up_16 = channelAtt(in_channels*2, 192)

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_16(conv1, imgs[2])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_32(conv2, imgs[3])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg(conv1)
        conv1 = self.feature_att_up_16(conv1, imgs[2])

        conv = self.conv1_up_(conv1)
        conv = self.conv_final(conv)

        return conv

class Fast_ACVNet(nn.Module):
    def __init__(self, maxdisp, att_weights_only):
        super(Fast_ACVNet, self).__init__()
        self.att_weights_only = att_weights_only
        self.maxdisp = maxdisp 
        self.feature = Feature()
        self.feature_up = FeatUp()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(2*torch.ones(1))
        chans = [16, 24, 32, 96, 160]

        self.patch = nn.Conv3d(12, 12, kernel_size=(1,3,3), stride=1, dilation=1, groups=12, padding=(0,1,1), bias=False)


        self.stem_2 = nn.Sequential(
                      BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                      nn.BatchNorm2d(32), nn.ReLU())
        self.stem_4 = nn.Sequential(
                      BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(48, 48, 3, 1, 1, bias=False),
                      nn.BatchNorm2d(48), nn.ReLU())

        self.stem_8 = nn.Sequential(
                      BasicConv(48, 48, kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(48, 32, 3, 1, 1, bias=False),
                      nn.BatchNorm2d(32), nn.ReLU())

        self.concat_feature = nn.Sequential(
                              BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
                              nn.Conv2d(32, 16, 3, 1, 1, bias=False))

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(24, 32, True)
        self.spx_4 = nn.Sequential(
                     BasicConv(96, 24, kernel_size=3, stride=1, padding=1),
                     nn.Conv2d(24, 24, 3, 1, 1, bias=False),
                     nn.BatchNorm2d(24), nn.ReLU())

        self.corr_feature_att_8 = channelAtt(12, 96)
        self.concat_feature_att_4 = channelAtt(16, 96)
        self.hourglass_att = hourglass_att(12)
        self.hourglass = hourglass(16)
        self.concat_stem = BasicConv(32, 16, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.propagation = Propagation()
        self.propagation_prob = Propagation_prob()

    def concat_volume_generator(self, left_input, right_input, disparity_samples):
        right_feature_map, left_feature_map = SpatialTransformer_grid(left_input,
                                                                       right_input, disparity_samples)
        concat_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        return concat_volume

    def forward(self, left, right):

        features_left = self.feature(left)
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right)

        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)

        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)
        stem_8y = self.stem_8(stem_4y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        features_left[1] = torch.cat((features_left[1], stem_8x), 1)
        features_right[1] = torch.cat((features_right[1], stem_8y), 1)

        corr_volume = build_gwc_volume_norm(features_left[1], features_right[1], self.maxdisp//8, 12)
        corr_volume = self.patch(corr_volume)
        cost_att = self.corr_feature_att_8(corr_volume, features_left[1])
        cost_att = self.hourglass_att(cost_att, features_left)
        att_weights = F.interpolate(cost_att, [self.maxdisp//4, left.size()[2]//4, left.size()[3]//4], mode='trilinear')
        
        pred_att = torch.squeeze(att_weights, 1)
        pred_att_prob = F.softmax(pred_att, dim=1)
        pred_att = disparity_regression(pred_att_prob, self.maxdisp // 4)
        pred_variance = disparity_variance(pred_att_prob, self.maxdisp // 4, pred_att.unsqueeze(1))
        pred_variance = self.beta + self.gamma * pred_variance
        pred_variance = torch.sigmoid(pred_variance)
        pred_variance_samples = self.propagation(pred_variance)

        disparity_samples = self.propagation(pred_att.unsqueeze(1))
        right_feature_x4, left_feature_x4 = SpatialTransformer_grid(stem_4x, stem_4y, disparity_samples)
        disparity_sample_strength = (left_feature_x4 * right_feature_x4).mean(dim=1)
        disparity_sample_strength = torch.softmax(disparity_sample_strength*pred_variance_samples, dim=1)
        att_weights = self.propagation_prob(att_weights)
        att_weights = att_weights * disparity_sample_strength.unsqueeze(2)
        att_weights = torch.sum(att_weights, dim=1, keepdim=True)
        att_weights_prob = F.softmax(att_weights, dim=2)

        _, ind = att_weights_prob.sort(2, True)
        k = 24
        ind_k = ind[:, :, :k]
        ind_k = ind_k.sort(2, False)[0]
        att_topk = torch.gather(att_weights_prob, 2, ind_k)

        disparity_sample_topk = ind_k.squeeze(1).float()

        if not self.att_weights_only:
            concat_features_left = self.concat_feature(features_left[0])
            concat_features_right = self.concat_feature(features_right[0])
            concat_volume = self.concat_volume_generator(concat_features_left, concat_features_right, disparity_sample_topk)
            volume = att_topk * concat_volume    
            volume = self.concat_stem(volume)
            volume = self.concat_feature_att_4(volume, features_left[0])
            cost = self.hourglass(volume, features_left)
        
        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        if self.training:
            att_prob = torch.gather(att_weights, 2, ind_k).squeeze(1)
            att_prob = F.softmax(att_prob, dim=1)
            pred_att = torch.sum(att_prob*disparity_sample_topk, dim=1)
            pred_att_up = context_upsample(pred_att.unsqueeze(1), spx_pred)
            if self.att_weights_only:
                return [pred_att_up*4, pred_att*4]
            else:
                pred = regression_topk(cost.squeeze(1), disparity_sample_topk, 2)
                pred_up = context_upsample(pred, spx_pred)
                return [pred_up*4, pred.squeeze(1)*4, pred_att_up*4, pred_att*4]

        else:
            if self.att_weights_only:
                att_prob = torch.gather(att_weights, 2, ind_k).squeeze(1)
                att_prob = F.softmax(att_prob, dim=1)
                pred_att = torch.sum(att_prob*disparity_sample_topk, dim=1)
                pred_att_up = context_upsample(pred_att.unsqueeze(1), spx_pred)
                return [pred_att_up*4]

            pred = regression_topk(cost.squeeze(1), disparity_sample_topk, 2)
            pred_up = context_upsample(pred, spx_pred)
            return [pred_up*4]
