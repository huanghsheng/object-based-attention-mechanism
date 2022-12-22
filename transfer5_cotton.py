"""
transfer3.py: Only transfers with the maximum cluster
"""

import os
import tqdm
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image

from model import WaveEncoder, WaveDecoder

from utils.core5 import feature_wct
from utils.io5 import Timer, open_image, load_segment, compute_label_info
import sys
from PIL import Image
import numpy as np
import os
import time

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class WCT2:
    def __init__(self, model_path='./model_checkpoints', transfer_at=['encoder', 'skip', 'decoder'], option_unpool='cat5', device='cuda:0', verbose=False):

        self.transfer_at = set(transfer_at)
        assert not(self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(transfer_at)
        assert self.transfer_at, 'empty transfer_at'

        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][2] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def extract_PatchesAndNorm(self, feats, p_size=3, stride=1):
        '''
        feats = (B,C,H,W), with B being 1
        core function: torch.nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
        return: patches = (num_patch, C, patch_size, patch_size) , kernel_norm = (num_patch, 1, 1, 1)
        '''
        B, C, H, W = feats.size()
        padding = p_size // 2

        # => (B, patch_size * patch_size * C, num_patch)
        unfold = nn.Unfold(kernel_size=(p_size, p_size), padding=0, stride=stride)

        content_pad = nn.ReflectionPad2d(padding)
        feats = content_pad(feats)
        raw_patches = unfold(feats)

        # => (B, num_patch, patch_size * patch_size * C)
        raw_patches = raw_patches.permute(0, 2, 1)

        return raw_patches

    def self_attention_map(self, content, style, content_feat, style_feats):
        import numpy as np
        from PIL import Image
        import torch
        import torch.nn as nn
        import matplotlib.pyplot as plt
        eps = 10e-5

        cont_img = Image.fromarray((content * 255).squeeze(0).permute(1, 2, 0).byte().cpu().numpy())
        cont_img = cont_img.resize([100, 75])
        cont_np = np.array(cont_img)
        cont_ts = torch.tensor(cont_np)
        # plt.figure(1)
        # plt.imshow(cont_np, cmap=plt.get_cmap('gray'))
        # plt.show()

        styl_img = Image.fromarray((style * 255).squeeze(0).permute(1, 2, 0).byte().cpu().numpy())
        styl_img = styl_img.resize([100, 75])
        styl_np = np.array(styl_img)
        styl_ts = torch.tensor(styl_np)
        # plt.figure(2)
        # plt.imshow(styl_np, cmap=plt.get_cmap('gray'))
        # plt.show()

        x = content_feat.clone()
        s = style_feats['encoder'][4].clone()
        feat1 = self.extract_PatchesAndNorm(x.clone(), p_size=3)
        feat2 = self.extract_PatchesAndNorm(s.clone(), p_size=3)

        norm11 = feat1.norm(p=2, dim=-1)
        norm12 = feat1 / (norm11.unsqueeze(-1).expand_as(feat1) + eps)

        norm21 = feat2.norm(p=2, dim=-1)
        norm22 = feat2 / (norm21.unsqueeze(-1).expand_as(feat2) + eps)
        norm23 = norm22.permute(0, 2, 1)

        f1 = torch.matmul(norm12, norm23)
        f2 = f1.max(dim=-1)[0].view(75, 100) * 255
        # f3 = Image.fromarray(f2.byte().cpu().numpy())
        # plt.figure(4)
        # plt.imshow(f3, cmap=plt.get_cmap('gray'))

        f2_loc = f1.max(dim=-1)[1][0]
        x_locs = f2_loc // 100
        y_locs = f2_loc % 100
        # result1 = styl_ts[x_locs, y_locs, :].view(75, 100, 3).to(self.device)
        # cont_ts = cont_ts.to(self.device)
        # styl_ts = styl_ts.to(self.device)
        # thresh = 180
        # result1 = (f2 > thresh).unsqueeze(-1).expand_as(result1) * result1 + (f2 <= thresh).unsqueeze(-1).expand_as(
        #     result1) * cont_ts
        # result1 = result1.cpu().numpy()
        # plt.figure(4)
        # plt.imshow(result1)

        return f2, x_locs, y_locs

    def rice_attention_map(self, content, style, content_feats, style_feats):
        import numpy as np
        from PIL import Image
        import torch
        import torch.nn as nn
        import matplotlib.pyplot as plt
        eps = 10e-5

        # cont_img = Image.fromarray((content * 255).squeeze(0).permute(1, 2, 0).byte().cpu().numpy())
        # cont_img = cont_img.resize([100, 75])
        # cont_np = np.array(cont_img)
        # cont_ts = torch.tensor(cont_np)
        # # plt.figure(1)
        # # plt.imshow(cont_np, cmap=plt.get_cmap('gray'))
        # # plt.show()
        #
        # styl_img = Image.fromarray((style * 255).squeeze(0).permute(1, 2, 0).byte().cpu().numpy())
        # styl_img = styl_img.resize([100, 75])
        # styl_np = np.array(styl_img)
        # styl_ts = torch.tensor(styl_np)
        # # plt.figure(2)
        # # plt.imshow(styl_np, cmap=plt.get_cmap('gray'))
        # # plt.show()

        # content_feats, content_skips = self.get_all_feature(content)
        # style_feats, style_skips = self.get_all_feature(style)

        x = content_feats['encoder'][2].clone()
        x = torch.nn.functional.interpolate(x, scale_factor=0.25)
        s = style_feats['encoder'][2].clone()
        s = torch.nn.functional.interpolate(s, scale_factor=0.25)
        feat1 = self.extract_PatchesAndNorm(x.clone(), p_size=3)
        feat2 = self.extract_PatchesAndNorm(s.clone(), p_size=3)

        norm11 = feat1.norm(p=2, dim=-1)
        norm12 = feat1 / (norm11.unsqueeze(-1).expand_as(feat1) + eps)

        norm21 = feat2.norm(p=2, dim=-1)
        norm22 = feat2 / (norm21.unsqueeze(-1).expand_as(feat2) + eps)
        norm23 = norm22.permute(0, 2, 1)

        f1 = torch.matmul(norm12, norm23)
        f2 = f1.max(dim=-1)[0].view(75, 100) * 255
        # f3 = Image.fromarray(f2.byte().cpu().numpy())
        # plt.figure(3)
        # plt.imshow(f3, cmap=plt.get_cmap('gray'))

        # cont_img = Image.open('./data/2017-9-30/images/DJI_0452_3_3.png').convert("RGB")
        # cont_ts = torch.tensor(cont_np)
        # plt.figure(1)
        # plt.imshow(cont_np, cmap=plt.get_cmap('gray'))
        #
        # styl_ts = torch.tensor(styl_np)
        # plt.figure(2)
        # plt.imshow(styl_np, cmap=plt.get_cmap('gray'))

        f2_loc = f1.max(dim=-1)[1][0]
        x_locs = f2_loc // 100
        y_locs = f2_loc % 100
        # result1 = styl_ts[x_locs, y_locs, :].view(75, 100, 3).to(self.device)
        # cont_ts = cont_ts.to(self.device)
        # styl_ts = styl_ts.to(self.device)
        # thresh = 160
        # result1 = (f2 > thresh).unsqueeze(-1).expand_as(result1) * result1 + (f2 <= thresh).unsqueeze(-1).expand_as(
        #     result1) * cont_ts
        #
        # result1 = result1.cpu().numpy()
        # plt.figure(4)
        # plt.imshow(result1)

        return f2, x_locs, y_locs

    def get_semantic_maps(self, style, content, style_feats, f2, x_locs, y_locs):
        import matplotlib.pyplot as plt
        from kmeans_pytorch import kmeans
        import torchvision
        import torch

        # styl_img = (style*255)[0].permute(1,2,0).clone().detach().byte().cpu().numpy()
        # plt.figure(1)
        # plt.imshow(styl_img, cmap=plt.get_cmap('gray'))
        #
        # cont_img = (content*255)[0].permute(1,2,0).clone().detach().byte().cpu().numpy()
        # plt.figure(2)
        # plt.imshow(cont_img, cmap=plt.get_cmap('gray'))

        s4 = style_feats['encoder'][4].clone()
        s4 = s4.view(512, -1).permute(1, 0)
        cluster_ids_x, cluster_centers = kmeans(
            X=s4, num_clusters=3, distance='euclidean', device=torch.device('cuda:0')
        )
        styl_map = cluster_ids_x.view(75, 100)
        # styl_map_img = styl_map.clone().detach().cpu().numpy()
        # plt.figure(5)
        # plt.imshow(styl_map_img)

        cont_map = styl_map[x_locs, y_locs].view(75,100).to(self.device)
        # cont_map_img = cont_map.clone().detach().cpu().numpy()
        # plt.figure(6)
        # plt.imshow(cont_map_img)

        self.thresh = 160
        cont_map[f2<=self.thresh] = 5
        # cont_map_img = cont_map.clone().detach().cpu().numpy()
        # plt.figure(4)
        # plt.imshow(cont_map_img)

        styl_map = styl_map.unsqueeze(0).unsqueeze(0).float().to(self.device)
        styl_map = torch.nn.functional.interpolate(styl_map, scale_factor=8)
        # styl_map_img = styl_map[0][0].clone().detach().cpu().numpy()
        # plt.figure(5)
        # plt.imshow(styl_map_img)

        cont_map = cont_map.unsqueeze(0).unsqueeze(0).float().to(self.device)
        cont_map = torch.nn.functional.interpolate(cont_map, scale_factor=8)
        # cont_map_img = cont_map[0][0].clone().detach().cpu().numpy()
        # plt.figure(6)
        # plt.imshow(cont_map_img)

        return cont_map, styl_map

    def transfer(self, content, style, alpha=1):
        #label_set, label_indicator = compute_label_info(content_segment, style_segment)
        # content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)
        content_feats, content_skips = self.get_all_feature(content)
        content_feat = content_feats['encoder'][4]

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        # for level in [1, 2, 3, 4]:
        #     content_feat = self.encode(content_feat, content_skips, level)
            # if 'encoder' in self.transfer_at and level in wct2_enc_level:
            #     content_feat = feature_wct(content_feat, style_feats['encoder'][level],
            #                                content_segment, style_segment,
            #                                label_set, label_indicator,
            #                                alpha=alpha, device=self.device)
            #     self.print_('transfer at encoder {}'.format(level))
        # if 'skip' in self.transfer_at:
        #     for skip_level in wct2_skip_level:
        #         for component in [0, 1, 2]:  # component: [LH, HL, HH]
        #             content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component], style_skips[skip_level][component],
        #                                                                content_segment, style_segment,
        #                                                                label_set, label_indicator,
        #                                                                alpha=alpha, device=self.device)
        #         self.print_('transfer at skip {}'.format(skip_level))


        medium_pause = 1
        f2, x_locs, y_locs = self.self_attention_map(content, style, content_feat, style_feats)
        # f2, x_locs, y_locs = self.rice_attention_map(content, style, content_feats, style_feats)
        content_segment, style_segment = self.get_semantic_maps(style, content, style_feats, f2, x_locs, y_locs)
        label_set, label_indicator = 1, 1

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, style_feats['decoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                # self.print_('transfer at decoder {}'.format(level))
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat


def get_all_transfer():
    ret = []
    for e in ['encoder']:
        for d in ['decoder']:
            for s in ['skip']:
                _ret = set([e, d, s]) & set(['encoder', 'decoder', 'skip'])
                if _ret:
                    ret.append(_ret)
    return ret


def run_training(config):
    device = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
    device = torch.device(device)
    
    _transfer_at = get_all_transfer()[0]
    wct2 = WCT2(transfer_at=_transfer_at, option_unpool=config.option_unpool, device=device, verbose=config.verbose)       
    os.chdir("/home/cvi/Desktop/task/test/semantic_transfer")
    mask_dir = "./result_seg/test_outputs"
    saved_dir = "./compares/local_transfer/WCT2-new/result/test_outputs"
    _style = "./data/2018-10-8/images/DJI_0481_3_3.png"
    # _style_segment = "./result_seg/test_outputs/2018-10-8/DJI_0481_3_3.png"
    # _style = "./data/2017-10-8/images/DJI_0481_3_3.png"
    #_style_segment = "./result_seg/test_outputs/2018-10-8/DJI_0481_3_3.png"
    
    for param in wct2.encoder.parameters():
        param.requires_grad = False
    dec_optim = torch.optim.Adam(
               filter(lambda p: p.requires_grad, wct2.decoder.parameters()),
               lr = config.lr
            )
    
    files = "./data/val_beans.txt"
    fid = open(files, "r")
    count = 0
    start_time = time.time()
    for item in fid:
        date = item.strip().split("/")[2]
        img_name = item.strip().split("/")[-1]
        _content = item.strip()
        _content_segment = mask_dir+"/"+date+"/"+img_name
        _result = saved_dir+"/"+date+"/"+img_name
        
        content = open_image(_content, config.image_size).to(device)
        style = open_image(_style, config.image_size).to(device)
        content_segment = np.array(Image.open(_content_segment))
        style_segment = np.array(Image.open(_style_segment))
        
        with torch.no_grad():
            img = wct2.transfer(content, style, content_segment, style_segment, alpha=config.alpha)
        #save_image(img.clamp_(0, 1), _result, padding=0)
        
        count = count+1
        print(count, item.strip())
    fid.close()
    end_time = time.time()
    duration = end_time-start_time
    
    print("good")


def run_bulk(config):
    device = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
    device = torch.device(device)
    
    _transfer_at = get_all_transfer()[0]
    wct2 = WCT2(transfer_at=_transfer_at, option_unpool=config.option_unpool, device=device, verbose=config.verbose)       
    os.chdir("/home/cvi/Desktop/task/test/semantic_transfer")
    # mask_dir = "./result_seg/test_outputs"
    saved_dir = "./compares/local_transfer/WCT2-new/result/test_outputs"
    _style = "./data/2017-7-15-field1/images/DJI_0068_4_1.png"
    # _style_segment = "./result_seg/test_outputs/2017-9-30/DJI_0426_4_4.png"
    # _style = "./data/2017-10-8/images/DJI_0481_3_3.png"
    #_style_segment = "./result_seg/test_outputs/2018-10-8/DJI_0481_3_3.png"
    _style1 = "./data/2017-7-15-field1/images/DJI_0068_4_1.png"
    _style2 = "./data/2017-7-15-field2/images/DJI_0258_1_0.png"
    
    files = "./data/val_cotton.txt"
    fid = open(files, "r")
    count = 0
    start_time = time.time()

    for item in fid:
        # item = "./data/2017-7-15-field2/images/DJI_0241_0_1.png"
        date = item.strip().split("/")[2]
        img_name = item.strip().split("/")[-1]
        if date in _style1:
            _style = _style1
        if date in _style2:
            _style = _style2
        style = open_image(_style, config.image_size).to(device)
        _content = item.strip()
        # _content_segment = mask_dir+"/"+date+"/"+img_name
        _result = saved_dir+"/"+date+"/"+img_name
        
        content = open_image(_content, config.image_size).to(device)
        # style = open_image(_style, config.image_size).to(device)
        # content_segment = np.array(Image.open(_content_segment))
        # style_segment = np.array(Image.open(_style_segment))
        
        with torch.no_grad():
            img = wct2.transfer(content, style, alpha=config.alpha)
        save_image(img.clamp_(0, 1), _result, padding=0)

        # import matplotlib.pyplot as plt
        #
        # content_img = (content * 255)[0].permute(1, 2, 0).byte().clone().detach().cpu().numpy()
        # plt.figure(1)
        # plt.imshow(content_img)
        #
        # style_img = (style * 255)[0].permute(1, 2, 0).byte().clone().detach().cpu().numpy()
        # plt.figure(2)
        # plt.imshow(style_img)
        #
        # img2 = img.clamp_(0, 1)
        # result_img = (img2 * 255)[0].permute(1, 2, 0).byte().clone().detach().cpu().numpy()
        # plt.figure(3)
        # plt.imshow(result_img)

        count = count+1
        # print(count, item.strip())
    fid.close()
    end_time = time.time()
    duration = end_time-start_time
    
    print("good")

if __name__ == '__main__':
    sys.argv.extend(['-a'])
    sys.argv.extend(['--verbose'])
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='./examples/content')
    parser.add_argument('--content_segment', type=str, default=None)
    parser.add_argument('--style', type=str, default='./examples/style')
    parser.add_argument('--style_segment', type=str, default=None)
    parser.add_argument('--output', type=str, default='./outputs')
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--option_unpool', type=str, default='cat5', choices=['sum', 'cat5'])
    parser.add_argument('-e', '--transfer_at_encoder', action='store_true')
    parser.add_argument('-d', '--transfer_at_decoder', action='store_true')
    parser.add_argument('-s', '--transfer_at_skip', action='store_true')
    parser.add_argument('-a', '--transfer_all', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    config = parser.parse_args()
    
    config.content = "./examples/content"
    config.content_segment = "./examples/content_segment"
    config.style = "./examples/style"
    config.style_segment = "./examples/style_segment"
    config.option_unpool = "cat5"
    #config.image_size = 512
    config.output = "./output"
    config.lr = 5e-5
    print(config)

    if not os.path.exists(os.path.join(config.output)):
        os.makedirs(os.path.join(config.output))

    '''
    CUDA_VISIBLE_DEVICES=6 python transfer.py --content ./examples/content --style ./examples/style --content_segment ./examples/content_segment --style_segment ./examples/style_segment/ --output ./outputs/ --verbose --image_size 512 -a
    '''
    #run_training(config)
    run_bulk(config)
    
    print("good")
