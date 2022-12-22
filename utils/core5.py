"""
from photo_wct.py of https://github.com/NVIDIA/FastPhotoStyle
Copyright (C) 2018 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0
"""
import torch
import numpy as np
from PIL import Image
from torch.nn.functional import interpolate

def svd(feat, iden=False, device='cpu'):
    size = feat.size()
    mean = torch.mean(feat, 1)
    mean = mean.unsqueeze(1).expand_as(feat)
    _feat = feat.clone()
    _feat -= mean
    if size[1] > 1:
        conv = torch.mm(_feat, _feat.t()).div(size[1] - 1)
    else:
        conv = torch.mm(_feat, _feat.t())
    if iden:
        conv += torch.eye(size[0]).to(device)
    u, e, v = torch.svd(conv, some=False)
    return u, e, v


def get_squeeze_feat(feat):
    _feat = feat.squeeze(0)
    size = _feat.size(0)
    return _feat.view(size, -1).clone()


def get_rank(singular_values, dim, eps=0.00001):
    r = dim
    for i in range(dim - 1, -1, -1):
        if singular_values[i] >= eps:
            r = i + 1
            break
    return r

def wct_core(cont_feat, styl_feat, weight=1, registers=None, device='cpu'):
    cont_feat = get_squeeze_feat(cont_feat)
    cont_min = cont_feat.min()
    cont_max = cont_feat.max()
    cont_mean = torch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
    cont_feat -= cont_mean

    if not registers:
        _, c_e, c_v = svd(cont_feat, iden=True, device=device)

        styl_feat = get_squeeze_feat(styl_feat)
        s_mean = torch.mean(styl_feat, 1)
        _, s_e, s_v = svd(styl_feat, iden=True, device=device)
        k_s = get_rank(s_e, styl_feat.size()[0])
        s_d = (s_e[0:k_s]).pow(0.5)
        EDE = torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d) * weight), (s_v[:, 0:k_s].t()))

        if registers is not None:
            registers['EDE'] = EDE
            registers['s_mean'] = s_mean
            registers['c_v'] = c_v
            registers['c_e'] = c_e
    else:
        EDE = registers['EDE']
        s_mean = registers['s_mean']
        _, c_e, c_v = svd(cont_feat, iden=True, device=device)

    k_c = get_rank(c_e, cont_feat.size()[0])
    c_d = (c_e[0:k_c]).pow(-0.5)
    # TODO could be more fast
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cont_feat)

    targetFeature = torch.mm(EDE, whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    targetFeature.clamp_(cont_min, cont_max)

    return targetFeature

# def wct_core(cont_feat, styl_feat, cont_feat_view, content_index, style_index, weight=1, registers=None, device='cpu'):
#     cont_feat = get_squeeze_feat(cont_feat)
#     cont_min = cont_feat.min()
#     cont_max = cont_feat.max()
#     cont_mean = torch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
#     cont_feat -= cont_mean
#
#     cont_feat_result = cont_feat_view.clone()
#     #cont_min = cont_feat_view.min()
#     #cont_max = cont_feat_view.max()
#     cont_view_mean = torch.mean(cont_feat_view, 1).unsqueeze(1).expand_as(cont_feat_view)
#     cont_feat_view -= cont_view_mean
#
#     if not registers:
#         _, c_e, c_v = svd(cont_feat, iden=True, device=device)
#
#         styl_feat = get_squeeze_feat(styl_feat)
#         s_mean = torch.mean(styl_feat, 1)
#         _, s_e, s_v = svd(styl_feat, iden=True, device=device)
#         k_s = get_rank(s_e, styl_feat.size()[0])
#         s_d = (s_e[0:k_s]).pow(0.5)
#         EDE = torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d) * weight), (s_v[:, 0:k_s].t()))
#
#         if registers is not None:
#             registers['EDE'] = EDE
#             registers['s_mean'] = s_mean
#             registers['c_v'] = c_v
#             registers['c_e'] = c_e
#     else:
#         EDE = registers['EDE']
#         s_mean = registers['s_mean']
#         _, c_e, c_v = svd(cont_feat, iden=True, device=device)
#
#     k_c = get_rank(c_e, cont_feat.size()[0])
#     c_d = (c_e[0:k_c]).pow(-0.5)
#     # TODO could be more fast
#     step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
#     step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
#     whiten_cF = torch.mm(step2, cont_feat)
#     #whiten_cF = torch.mm(step2, cont_feat_view)
#
#     targetFeature = torch.mm(EDE, whiten_cF)
#     targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
#     targetFeature.clamp_(cont_min, cont_max)
#
#     # target_shift = targetFeature.mean(dim=1)-cont_feat.mean(dim=1)
#
#     return targetFeature
#
#     # cont_all_index = torch.zeros(cont_feat_view.size(1))
#     # cont_all_index[content_index] = 1
#     # cont_vers_index = torch.where(cont_all_index==0)[0]
#     # unTargetFeature = torch.index_select(cont_feat_view, 1, cont_vers_index.cuda(0))
#     #
#     # target_shift = targetFeature.mean(dim=1)-cont_feat.mean(dim=1)
#     # unTargetFeature = unTargetFeature+target_shift.unsqueeze(1).expand_as(unTargetFeature)
#     # cont_feat_result[:,content_index] = targetFeature
#     # cont_feat_result[:,cont_vers_index] = unTargetFeature
#     #
#     # return cont_feat_result


def wct_core_segment(content_feat, style_feat, content_segment, style_segment,
                     label_set, label_indicator, weight=1, registers=None,
                     device='cpu'):
    def resize(feat, target):
        size = (target.size(1), target.size(2))
        if len(feat.shape) == 2:
            return interpolate(feat.unsqueeze(0).unsqueeze(0), size, mode='nearest')[0,0]
            # return np.asarray(Image.fromarray(feat).resize(size, Image.NEAREST))
        else:
            return interpolate(feat, size, mode='nearest')
            # return np.asarray(Image.fromarray(feat, mode='RGB').resize(size, Image.NEAREST))

    def get_index(feat, label):
        # mask = np.where(feat.reshape(feat.shape[0] * feat.shape[1]) == label)
        mask = torch.where(feat.view(-1)==label)
        if mask[0].size(0) <= 0:
            return None
        return mask[0].long()

    invalid_label = 5
    mininum_rate = 0.1

    squeeze_content_feat = content_feat.squeeze(0)
    squeeze_style_feat = style_feat.squeeze(0)

    content_feat_view = squeeze_content_feat.view(squeeze_content_feat.size(0), -1).clone()
    style_feat_view = squeeze_style_feat.view(squeeze_style_feat.size(0), -1).clone()

    resized_content_segment = resize(content_segment, squeeze_content_feat)
    resized_style_segment = resize(style_segment, squeeze_style_feat)

    target_feature = content_feat_view.clone()
    shift_according = []
    # resized_content_segment = np.copy(resized_content_segment)
    for label in resized_content_segment.unique():
        if not ((resized_content_segment==label).sum()>resized_content_segment.numel()*mininum_rate and \
            (resized_style_segment==label).sum()>resized_style_segment.numel()*mininum_rate):
            if (resized_content_segment==label).sum()>0:
                # np.where(resized_content_segment==label)
                resized_content_segment[resized_content_segment==label] = invalid_label
            continue

        content_index = get_index(resized_content_segment, label)
        style_index = get_index(resized_style_segment, label)
        masked_content_feat = torch.index_select(content_feat_view, 1, content_index)
        masked_style_feat = torch.index_select(style_feat_view, 1, style_index)
        _target_feature = wct_core(masked_content_feat, masked_style_feat, weight, registers, device=device)
        shift_according.append(content_index)
        target_feature.index_copy_(1, content_index, _target_feature)

    cont_vers_index = get_index(resized_content_segment, invalid_label)
    if cont_vers_index==None:
        return target_feature

    target_shift = torch.zeros([target_feature.size(0)], dtype=torch.float32).cuda(0)
    num_total = 0.0
    for item in shift_according:
        num_total = num_total+item.size(0)
        # print(item.size())

    for content_index in shift_according:
        _cont_feat = content_feat_view[:, content_index]
        _target_feature = target_feature[:, content_index]
        _target_shift = _target_feature.mean(dim=1) - _cont_feat.mean(dim=1)
        target_shift = target_shift+(content_index.size(0)/num_total)*_target_shift

    unTargetFeature = torch.index_select(content_feat_view, 1, cont_vers_index.cuda(0))
    unTargetFeature = unTargetFeature+target_shift.unsqueeze(1).expand_as(unTargetFeature)
    target_feature[:,cont_vers_index] = unTargetFeature

    return target_feature


def feature_wct(content_feat, style_feat, content_segment=None, style_segment=None,
                label_set=None, label_indicator=None, weight=1, registers=None, alpha=1, device='cpu'):
    if label_set is not None:
        target_feature = wct_core_segment(content_feat, style_feat, content_segment, style_segment,
                                          label_set, label_indicator, weight, registers, device=device)
    else:
        target_feature = wct_core(content_feat, style_feat, device=device)
    target_feature = target_feature.view_as(content_feat)
    target_feature = alpha * target_feature + (1 - alpha) * content_feat
    return target_feature
