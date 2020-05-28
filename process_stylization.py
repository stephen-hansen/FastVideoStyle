"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import time
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
import torch
from smooth_filter import smooth_filter
import cv2

class ReMapping:
    def __init__(self):
        self.remapping = []

    def process(self, seg):
        new_seg = seg.copy()
        for k, v in self.remapping.items():
            new_seg[seg == k] = v
        return new_seg


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def memory_limit_image_resize(cont_img):
    # prevent too small or too big images
    MINSIZE=256
    MAXSIZE=960
    orig_width = cont_img.width
    orig_height = cont_img.height
    if max(cont_img.width,cont_img.height) < MINSIZE:
        if cont_img.width > cont_img.height:
            cont_img.thumbnail((int(cont_img.width*1.0/cont_img.height*MINSIZE), MINSIZE), Image.BICUBIC)
        else:
            cont_img.thumbnail((MINSIZE, int(cont_img.height*1.0/cont_img.width*MINSIZE)), Image.BICUBIC)
    if min(cont_img.width,cont_img.height) > MAXSIZE:
        if cont_img.width > cont_img.height:
            cont_img.thumbnail((MAXSIZE, int(cont_img.height*1.0/cont_img.width*MAXSIZE)), Image.BICUBIC)
        else:
            cont_img.thumbnail(((int(cont_img.width*1.0/cont_img.height*MAXSIZE), MAXSIZE)), Image.BICUBIC)
    print("Resize image: (%d,%d)->(%d,%d)" % (orig_width, orig_height, cont_img.width, cont_img.height))
    return cont_img.width, cont_img.height


def stylize_image(stylization_module, smoothing_module, cont_img, styl_img, cont_seg, styl_seg, cuda,
        no_post, cont_seg_remapping=None, styl_seg_remapping=None):
    with torch.no_grad():
        new_cw, new_ch = memory_limit_image_resize(cont_img)
        new_sw, new_sh = memory_limit_image_resize(styl_img)
        cont_pilimg = cont_img.copy()
        cw = cont_pilimg.width
        ch = cont_pilimg.height
        try:
            cont_seg.resize((new_cw,new_ch),Image.NEAREST)
            styl_seg.resize((new_sw,new_sh),Image.NEAREST)
        except:
            cont_seg = []
            styl_seg = []

        cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
        styl_img = transforms.ToTensor()(styl_img).unsqueeze(0)

        if cuda:
            cont_img = cont_img.cuda(0)
            styl_img = styl_img.cuda(0)
            stylization_module.cuda(0)

        # cont_img = Variable(cont_img, volatile=True)
        # styl_img = Variable(styl_img, volatile=True)

        cont_seg = np.asarray(cont_seg)
        styl_seg = np.asarray(styl_seg)
        if cont_seg_remapping is not None:
            cont_seg = cont_seg_remapping.process(cont_seg)
        if styl_seg_remapping is not None:
            styl_seg = styl_seg_remapping.process(styl_seg)

        with Timer("Elapsed time in stylization: %f"):
            stylized_img = stylization_module.transform(cont_img, styl_img, cont_seg, styl_seg)
        if ch != new_ch or cw != new_cw:
            print("De-resize image: (%d,%d)->(%d,%d)" %(new_cw,new_ch,cw,ch))
            stylized_img = nn.functional.upsample(stylized_img, size=(ch,cw), mode='bilinear')
        grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray(ndarr)

        with Timer("Elapsed time in propagation: %f"):
            out_img = smoothing_module.process(out_img, cont_pilimg)

        if no_post is False:
            with Timer("Elapsed time in post processing: %f"):
                out_img = smooth_filter(out_img, cont_pilimg, f_radius=15, f_edge=1e-1)
        return out_img


def stylization(stylization_module, smoothing_module, content_image_path, style_image_path, content_seg_path, style_seg_path, output_image_path,
                cuda, no_post, cont_seg_remapping=None, styl_seg_remapping=None):
    # Load image
    with torch.no_grad():
        cont_img = Image.open(content_image_path).convert('RGB')
        styl_img = Image.open(style_image_path).convert('RGB')
        try:
            cont_seg = Image.open(content_seg_path)
            styl_seg = Image.open(style_seg_path)
        except:
            cont_seg = []
            styl_seg = []
        out_img = stylize_image(stylization_module, smoothing_module, cont_img, styl_img, cont_seg,
                styl_seg, cuda, no_post, cont_seg_remapping, styl_seg_remapping)
        out_img.save(output_image_path)

def video_stylization_basic(stylization_module, smoothing_module, content_video_path, style_image_path,
        content_seg_video_path, style_seg_path, output_video_path, cuda, no_post, cont_seg_remapping=None, styl_seg_remapping=None):
    # Load image
    with torch.no_grad():
        cap = cv2.VideoCapture(content_video_path)
        success, cont_img_array = cap.read()
        styl_img = Image.open(style_image_path).convert('RGB')
        try:
            seg_cap = cv2.VideoCapture(content_seg_path)
            seg_success, cont_seg_array = seg_cap.read()
            styl_seg = Image.open(style_seg_path)
        except:
            seg_cap = None
            cont_seg = []
            styl_seg = []

        frames = []
        while (success):
            cont_img = Image.fromarray(cv2.cvtColor(cont_img_array,cv2.COLOR_BGR2RGB))
            if seg_cap != None:
                cont_seg = Image.fromarray(cv2.cvtColor(cont_seg_array,cv2.COLOR_BGR2RGB))

            out_img = stylize_image(stylization_module, smoothing_module, cont_img, styl_img, cont_seg,
                styl_seg, cuda, no_post, cont_seg_remapping, styl_seg_remapping)
            frames.append(out_img)
            
            success, cont_img_array = cap.read()
            if seg_cap != None:
                seg_success, cont_seg_array = seg_cap.read()
                if not seg_success:
                    break
        
        height, width, layers = frames[0].shape
        size = (width,height)
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, size)
        for f in frames:
            out.write(f)
        out.release()

