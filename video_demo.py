"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import argparse
import torch
import process_stylization
from photo_wct import PhotoWCT
parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth')
parser.add_argument('--content_video_path', default='./videos/video1.mp4')
parser.add_argument('--content_seg_video_path', default=[])
parser.add_argument('--style_image_path', default='./images/style1.png')
parser.add_argument('--style_seg_path', default=[])
parser.add_argument('--output_video_path', default='./results/example1.avi')
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--no_post', action='store_true', default=False)
parser.add_argument('--cuda', type=int, default=1, help='Enable CUDA.')
parser.add_argument('--nframes', type=int, default=-1)
parser.add_argument('--general_flow', action='store_true', default=False)
parser.add_argument('--color_mapping', action='store_true', default=False)
parser.add_argument('--optical_flow', action='store_true', default=False)
args = parser.parse_args()

# Load model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model))

if args.fast:
    from photo_gif import GIFSmoothing
    p_pro = GIFSmoothing(r=35, eps=0.001)
else:
    from photo_smooth import Propagator
    p_pro = Propagator()
if args.cuda:
    p_wct.cuda(0)

if args.optical_flow:
    process_stylization.video_stylization_optical_flow(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        content_video_path=args.content_video_path,
        style_image_path=args.style_image_path,
        content_seg_video_path=args.content_seg_video_path,
        style_seg_path=args.style_seg_path,
        output_video_path=args.output_video_path,
        cuda=args.cuda,
        no_post=args.no_post,
        nframes=args.nframes
    )
elif args.color_mapping:
    process_stylization.video_stylization_color_mapping(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        content_video_path=args.content_video_path,
        style_image_path=args.style_image_path,
        content_seg_video_path=args.content_seg_video_path,
        style_seg_path=args.style_seg_path,
        output_video_path=args.output_video_path,
        cuda=args.cuda,
        no_post=args.no_post,
        nframes=args.nframes
    )
elif args.general_flow:
    process_stylization.video_stylization_general_flow(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        content_video_path=args.content_video_path,
        style_image_path=args.style_image_path,
        content_seg_video_path=args.content_seg_video_path,
        style_seg_path=args.style_seg_path,
        output_video_path=args.output_video_path,
        cuda=args.cuda,
        no_post=args.no_post,
        nframes=args.nframes
    )
else:
    process_stylization.video_stylization_basic(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        content_video_path=args.content_video_path,
        style_image_path=args.style_image_path,
        content_seg_video_path=args.content_seg_video_path,
        style_seg_path=args.style_seg_path,
        output_video_path=args.output_video_path,
        cuda=args.cuda,
        no_post=args.no_post,
        nframes=args.nframes
    )
