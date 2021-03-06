"""
process_stylization.py

Contains new video stylization methods and has old methods for image stylization.
"""

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
from model import *

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
    MAXSIZE=480 # limit to 480p
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
    """
    Stylize a single image, a single frame.
    stylization_module : the style model
    smoothing_module : post-processing smoothing module
    cont_img : content, Pillow Image
    styl_img : style, Pillow Image
    cont_seg : segmentation map for content, Pillow Image
    styl_seg : segmentation map for style, Pillow Image
    cuda : use GPU or not
    no_post : enable post_processing
    cont_seg_remapping : map segmentation to image sections
    styl_seg_remapping : map segmentation to style sections
    """
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
    """
    Stylize and save a single image.
    stylization_module : the style model
    smoothing_module : post-processing smoothing module
    content_image_path : path of content image
    style_image_path : path of style image
    content_seg_path : path of content segmentation
    style_seg_path : path of style segmentation
    output_image_path : place to store result image
    cuda : use GPU or not
    no_post : enable post_processing
    cont_seg_remapping : map segmentation to image sections
    styl_seg_remapping : map segmentation to style sections
    """
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
        content_seg_video_path, style_seg_path, output_video_path, cuda, no_post, cont_seg_remapping=None,
        styl_seg_remapping=None, nframes=-1):
    """
    Stylize and save a video. Style each frame separately.
    stylization_module : the style model
    smoothing_module : post-processing smoothing module
    content_video_path : path of content video
    style_image_path : path of style image
    content_seg_video_path : path of content segmentation video
    style_seg_path : path of style segmentation
    output_video_path : place to store result video
    cuda : use GPU or not
    no_post : enable post_processing
    cont_seg_remapping : map segmentation to image sections
    styl_seg_remapping : map segmentation to style sections
    nframes : number of frames to stylize. if -1, stylize all frames.
    """
    with torch.no_grad():
        cap = cv2.VideoCapture(content_video_path) # create video capture for content
        success, cont_img_array = cap.read()
        styl_img = Image.open(style_image_path).convert('RGB')
        try:
            seg_cap = cv2.VideoCapture(content_seg_path) # create video capture for segmentation, if any
            seg_success, cont_seg_array = seg_cap.read()
            styl_seg = Image.open(style_seg_path)
        except:
            seg_cap = None
            cont_seg = []
            styl_seg = []

        frames = []
        count = 0
        while success and (nframes == -1 or count < nframes): # Loop over frames, while reading or if nframes limit is hit
            count += 1
            cont_img = Image.fromarray(cv2.cvtColor(cont_img_array,cv2.COLOR_BGR2RGB)) # Load the current frame
            if seg_cap != None:
                cont_seg = Image.fromarray(cv2.cvtColor(cont_seg_array,cv2.COLOR_BGR2RGB)) # Load current seg

            # Stylize the frame
            out_img = stylize_image(stylization_module, smoothing_module, cont_img, styl_img, cont_seg,
                styl_seg, cuda, no_post, cont_seg_remapping, styl_seg_remapping)
            frames.append(np.array(out_img)[:,:,::-1].copy()) # Add to list of frames
            
            # Load next frame
            success, cont_img_array = cap.read()
            if seg_cap != None:
                seg_success, cont_seg_array = seg_cap.read()
                if not seg_success:
                    break
        if len(frames) < 1:
            return
        height, width, layers = frames[0].shape
        size = (width,height)
        # Save the video, 30 FPS, XVID
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, size)
        for f in frames:
            out.write(f)
        out.release()

def video_stylization_general_flow(stylization_module, smoothing_module, content_video_path, style_image_path,
        content_seg_video_path, style_seg_path, output_video_path, cuda, no_post, cont_seg_remapping=None,
        styl_seg_remapping=None, nframes=-1):
    """
    Stylize and save a video.
    Check for pixels between scenes that are identical colors. If so, copy the model pixel color
    from one frame to the next. Otherwise, use a new generated image for the color.
    Should be equally as slow as the normal basic method, but add some improvement to
    removing artifacts.
    stylization_module : the style model
    smoothing_module : post-processing smoothing module
    content_video_path : path of content video
    style_image_path : path of style image
    content_seg_video_path : path of content segmentation video
    style_seg_path : path of style segmentation
    output_video_path : place to store result video
    cuda : use GPU or not
    no_post : enable post_processing
    cont_seg_remapping : map segmentation to image sections
    styl_seg_remapping : map segmentation to style sections
    nframes : number of frames to stylize. if -1, stylize all frames.
    """

    with torch.no_grad():
        cap = cv2.VideoCapture(content_video_path)
        # Load video, load first frame
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

        prev_cont_img = None
        prev_out_img = None
        frames = []
        count = 0
        # Loop over frames
        while success and (nframes == -1 or count < nframes):
            count += 1
            cont_img = Image.fromarray(cv2.cvtColor(cont_img_array,cv2.COLOR_BGR2RGB))
            memory_limit_image_resize(cont_img)
            if seg_cap != None:
                cont_seg = Image.fromarray(cv2.cvtColor(cont_seg_array,cv2.COLOR_BGR2RGB))

            out_img = stylize_image(stylization_module, smoothing_module, cont_img, styl_img, cont_seg,
                styl_seg, cuda, no_post, cont_seg_remapping, styl_seg_remapping)
            
            # If prev content image exists,
            # loop over pixels, if current color equals previous set the out color
            # to the previous out color
            if prev_cont_img != None:
                width, height = prev_cont_img.size
                for x in range(width):
                    for y in range(height):
                        color = cont_img.getpixel((x,y))
                        prev_color = prev_cont_img.getpixel((x,y))
                        if color == prev_color:
                            out_img.putpixel((x,y), prev_out_img.getpixel((x,y)))

            frames.append(np.array(out_img)[:,:,::-1].copy())
            
            # Save previous input and output frames
            prev_cont_img = cont_img
            prev_out_img = out_img

            # Read next frame
            success, cont_img_array = cap.read()
            if seg_cap != None:
                seg_success, cont_seg_array = seg_cap.read()
                if not seg_success:
                    break
        if len(frames) < 1:
            return
        height, width, layers = frames[0].shape
        size = (width,height)
        # Save video
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, size)
        for f in frames:
            out.write(f)
        out.release()

def video_stylization_color_mapping(stylization_module, smoothing_module, content_video_path, style_image_path,
        content_seg_video_path, style_seg_path, output_video_path, cuda, no_post, cont_seg_remapping=None,
        styl_seg_remapping=None, nframes=-1):
    """
    Stylize and save a video.
    Create a mapping of input colors to output model colors. If a mapping does not exist, use the model result
    and add it to the mapping. Otherwise, use the existing mapping result.
    stylization_module : the style model
    smoothing_module : post-processing smoothing module
    content_video_path : path of content video
    style_image_path : path of style image
    content_seg_video_path : path of content segmentation video
    style_seg_path : path of style segmentation
    output_video_path : place to store result video
    cuda : use GPU or not
    no_post : enable post_processing
    cont_seg_remapping : map segmentation to image sections
    styl_seg_remapping : map segmentation to style sections
    nframes : number of frames to stylize. if -1, stylize all frames.
    """

    with torch.no_grad():
        # Load video, read a frame
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

        color_mapping = {}
        frames = []
        count = 0
        # Loop over frames
        while success and (nframes == -1 or count < nframes):
            count += 1
            cont_img = Image.fromarray(cv2.cvtColor(cont_img_array,cv2.COLOR_BGR2RGB))
            memory_limit_image_resize(cont_img)
            if seg_cap != None:
                cont_seg = Image.fromarray(cv2.cvtColor(cont_seg_array,cv2.COLOR_BGR2RGB))

            # Stylize the frame
            out_img = stylize_image(stylization_module, smoothing_module, cont_img, styl_img, cont_seg,
                styl_seg, cuda, no_post, cont_seg_remapping, styl_seg_remapping)
            
            # loop over pixels, if current color is in mapping set the out color
            # to the previous out color
            width, height = cont_img.size
            for x in range(width):
                for y in range(height):
                    color = cont_img.getpixel((x,y))
                    if color in color_mapping:
                        out_img.putpixel((x,y), color_mapping[color])
                    else:
                        color_mapping[color] = out_img.getpixel((x,y))

            frames.append(np.array(out_img)[:,:,::-1].copy())
            
            # Read the next frame
            success, cont_img_array = cap.read()
            if seg_cap != None:
                seg_success, cont_seg_array = seg_cap.read()
                if not seg_success:
                    break
        if len(frames) < 1:
            return
        height, width, layers = frames[0].shape
        size = (width,height)
        # Save video
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, size)
        for f in frames:
            out.write(f)
        out.release()

def video_stylization_optical_flow(stylization_module, smoothing_module, content_video_path, style_image_path,
        content_seg_video_path, style_seg_path, output_video_path, cuda, no_post, cont_seg_remapping=None,
        styl_seg_remapping=None, nframes=-1):
    """
    Stylize and save a video.
    Take the previous frame and current frame, find the forward/backward flows between each. Warp each image
    in the flow direction and do an alpha blend of both warped results.
    stylization_module : the style model
    smoothing_module : post-processing smoothing module
    content_video_path : path of content video
    style_image_path : path of style image
    content_seg_video_path : path of content segmentation video
    style_seg_path : path of style segmentation
    output_video_path : place to store result video
    cuda : use GPU or not
    no_post : enable post_processing
    cont_seg_remapping : map segmentation to image sections
    styl_seg_remapping : map segmentation to style sections
    nframes : number of frames to stylize. if -1, stylize all frames.
    """
    
    with torch.no_grad():
        # Load video, first frame
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

        set_prev = False
        prev_cont_img_gray = None
        prev_out_img = None
        frames = []
        count = 0
        # Iterate over frames
        while success and (nframes == -1 or count < nframes):
            count += 1
            cont_img = Image.fromarray(cv2.cvtColor(cont_img_array,cv2.COLOR_BGR2RGB))
            memory_limit_image_resize(cont_img)
            cont_img_array = np.array(cont_img)
            # Get the current frame in grayscale
            cont_img_gray = cv2.cvtColor(cont_img_array,cv2.COLOR_BGR2GRAY)
            if seg_cap != None:
                cont_seg = Image.fromarray(cv2.cvtColor(cont_seg_array,cv2.COLOR_BGR2RGB))

            # Stylize the image
            out_img = stylize_image(stylization_module, smoothing_module, cont_img, styl_img, cont_seg,
                styl_seg, cuda, no_post, cont_seg_remapping, styl_seg_remapping)
           
            out_img_arr = np.array(out_img)[:,:,::-1].copy()
            new_out_img_arr = np.array(out_img)[:,:,::-1].copy()
            new_out_img_arr2 = np.array(out_img)[:,:,::-1].copy()

            if set_prev: # if a previous frame exists
                # Get forward flow
                flow = cv2.calcOpticalFlowFarneback(prev_cont_img_gray, cont_img_gray, None, 0.5, 5, 15, 3, 5, 1.1, 0)
                # Get backwards flow
                bflow = cv2.calcOpticalFlowFarneback(cont_img_gray, prev_cont_img_gray, None, 0.5, 5, 15, 3, 5, 1.1, 0)
                height = prev_out_img_arr.shape[0]
                width = prev_out_img_arr.shape[1]
                xs = np.arange(width)
                ys = np.arange(height)
                grid = np.meshgrid(xs, ys)
                x_grid = grid[0]
                y_grid = grid[1]
                new_grid = np.copy(grid).astype(np.float64)
                # Warp the meshgrid to the forward flow
                new_grid[0] += flow[:,:,0]
                new_grid[1] += flow[:,:,1]
                new_grid = np.around(new_grid).astype(np.int64)
                mask = (new_grid[0] < 0) | (new_grid[0] >= width) # Prevent out of bounds error
                new_x_grid = np.copy(new_grid[0])
                new_x_grid[mask] = x_grid[mask]
                mask = (new_grid[1] < 0) | (new_grid[1] >= height) # Prevent out of bounds error
                new_y_grid =  np.copy(new_grid[1])
                new_y_grid[mask] = y_grid[mask]
                # Perform the forward warp
                new_out_img_arr[new_y_grid,new_x_grid] = prev_out_img_arr[y_grid,x_grid]
                new_grid2 = np.copy(grid).astype(np.float64)
                # Warp the meshgrid to the backward flow
                new_grid2[0] += bflow[:,:,0]
                new_grid2[1] += bflow[:,:,1]
                new_grid2 = np.around(new_grid2).astype(np.int64)
                mask = (new_grid2[0] < 0) | (new_grid2[0] >= width) # Prevent out of bounds error
                new_x_grid2 = np.copy(new_grid2[0])
                new_x_grid2[mask] = x_grid[mask]
                mask = (new_grid2[1] < 0) | (new_grid2[1] >= height) # Prevent out of bounds error
                new_y_grid2 =  np.copy(new_grid2[1])
                new_y_grid2[mask] = y_grid[mask]
                # Perform the backward warp
                new_out_img_arr2[new_y_grid2,new_x_grid2] = out_img_arr[y_grid,x_grid]
                # Convert warped images to Pillow Image
                new_out_img = Image.fromarray(cv2.cvtColor(new_out_img_arr, cv2.COLOR_BGR2RGB))
                new_out_img2 = Image.fromarray(cv2.cvtColor(new_out_img_arr2, cv2.COLOR_BGR2RGB))

                # Blend both images at 50%
                final_out = Image.blend(new_out_img, new_out_img2, 0.5)
            else:
                final_out = out_img
                set_prev = True

            frames.append(np.array(final_out)[:,:,::-1].copy())
            
            # Save the previous output and previous grayscale frame
            prev_cont_img_gray = cont_img_gray
            prev_out_img_arr = np.array(final_out)[:,:,::-1].copy()

            # Read a frame
            success, cont_img_array = cap.read()
            if seg_cap != None:
                seg_success, cont_seg_array = seg_cap.read()
                if not seg_success:
                    break
        if len(frames) < 1:
            return
        height, width, layers = frames[0].shape
        size = (width,height)
        # Save video
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, size)
        for f in frames:
            out.write(f)
        out.release()

def video_stylization_smart_optical_flow(stylization_module, smoothing_module, content_video_path, style_image_path,
        content_seg_video_path, style_seg_path, output_video_path, cuda, no_post, cont_seg_remapping=None,
        styl_seg_remapping=None, nframes=-1):
    """
    Stylize and save a video.
    Use gradient descent to train the output image based on content loss, style loss, and
    temporal loss. Slower than other methods but produces far more interesting results.
    stylization_module : the style model
    smoothing_module : post-processing smoothing module
    content_video_path : path of content video
    style_image_path : path of style image
    content_seg_video_path : path of content segmentation video
    style_seg_path : path of style segmentation
    output_video_path : place to store result video
    cuda : use GPU or not
    no_post : enable post_processing
    cont_seg_remapping : map segmentation to image sections
    styl_seg_remapping : map segmentation to style sections
    nframes : number of frames to stylize. if -1, stylize all frames.
    """
    with torch.no_grad():
        # Load video
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

        set_prev = 0
        prev_cont_img = None
        pprev_cont_img = None
        prev_out_img = None
        frames = []
        count = 0
        # Loop over frames
        while success and (nframes == -1 or count < nframes):
            count += 1
            cont_img = Image.fromarray(cv2.cvtColor(cont_img_array,cv2.COLOR_BGR2RGB))
            memory_limit_image_resize(cont_img)
            cont_img_array = np.array(cont_img)
            cont_img_gray = cv2.cvtColor(cont_img_array,cv2.COLOR_BGR2GRAY)
            if seg_cap != None:
                cont_seg = Image.fromarray(cv2.cvtColor(cont_seg_array,cv2.COLOR_BGR2RGB))

            # Stylize frame with PhotoWCT
            out_img = stylize_image(stylization_module, smoothing_module, cont_img, styl_img, cont_seg,
                styl_seg, cuda, no_post, cont_seg_remapping, styl_seg_remapping)
            
            # Resize content, style
            cont_img = cont_img.resize(out_img.size)
            styl_img = styl_img.resize(out_img.size)

            # Run style transfer on the content image, style image, output image, previous output image,
            # previous content image, and second previous content image. Load each image into a tensor,
            # then take the training result tensor and unload into an image.
            if set_prev == 2:
                final_out = unloader(run_style_transfer(image_loader(cont_img),
                    image_loader(styl_img), image_loader(out_img), image_loader(prev_out_img),
                    image_loader(prev_cont_img), image_loader(pprev_cont_img)))
            elif set_prev == 1:
                final_out = unloader(run_style_transfer(image_loader(cont_img),
                    image_loader(styl_img), image_loader(out_img), image_loader(prev_out_img),
                    image_loader(prev_cont_img), image_loader(prev_cont_img), temporal_weight=0))
                set_prev = 2
            elif set_prev == 0:
                final_out = unloader(run_style_transfer(image_loader(cont_img),
                    image_loader(styl_img), image_loader(out_img), image_loader(out_img),
                    image_loader(cont_img), image_loader(cont_img), temporal_weight=0))
                set_prev = 1

            frames.append(np.array(final_out)[:,:,::-1].copy())
            
            # Set previous frame values
            prev_out_img = final_out
            pprev_cont_img = prev_cont_img
            prev_cont_img = cont_img

            # Load the next frame
            success, cont_img_array = cap.read()
            if seg_cap != None:
                seg_success, cont_seg_array = seg_cap.read()
                if not seg_success:
                    break
        if len(frames) < 1:
            return
        height, width, layers = frames[0].shape
        size = (width,height)
        # Save the video
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, size)
        for f in frames:
            out.write(f)
        out.release()

def video_stylization_artistic_optical_flow(stylization_module, smoothing_module, content_video_path, style_image_path,
        content_seg_video_path, style_seg_path, output_video_path, cuda, no_post, cont_seg_remapping=None,
        styl_seg_remapping=None, nframes=-1):
    """
    Stylize and save a video.
    Use gradient descent to train the output image based on content loss, style loss, and
    temporal loss. Slower than other methods but produces far more interesting results.
    Use a higher style weight for artistic style.
    stylization_module : the style model
    smoothing_module : post-processing smoothing module
    content_video_path : path of content video
    style_image_path : path of style image
    content_seg_video_path : path of content segmentation video
    style_seg_path : path of style segmentation
    output_video_path : place to store result video
    cuda : use GPU or not
    no_post : enable post_processing
    cont_seg_remapping : map segmentation to image sections
    styl_seg_remapping : map segmentation to style sections
    nframes : number of frames to stylize. if -1, stylize all frames.
    """
    with torch.no_grad():
        # Load the video
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

        set_prev = 0
        prev_cont_img = None
        pprev_cont_img = None
        prev_out_img = None
        frames = []
        count = 0
        while success and (nframes == -1 or count < nframes):
            count += 1
            cont_img = Image.fromarray(cv2.cvtColor(cont_img_array,cv2.COLOR_BGR2RGB))
            memory_limit_image_resize(cont_img)
            cont_img_array = np.array(cont_img)
            cont_img_gray = cv2.cvtColor(cont_img_array,cv2.COLOR_BGR2GRAY)
            if seg_cap != None:
                cont_seg = Image.fromarray(cv2.cvtColor(cont_seg_array,cv2.COLOR_BGR2RGB))
            
            # Stylize frame with PhotoWCT
            out_img = stylize_image(stylization_module, smoothing_module, cont_img, styl_img, cont_seg,
                styl_seg, cuda, no_post, cont_seg_remapping, styl_seg_remapping)
            
            # Resize content, style images
            cont_img = cont_img.resize(out_img.size)
            styl_img = styl_img.resize(out_img.size)

            # Run style transfer on the tensor-converted images with a higher than normal style weight.
            # Unload the output tensor into an image.
            if set_prev == 2:
                final_out = unloader(run_style_transfer(image_loader(cont_img),
                    image_loader(styl_img), image_loader(out_img), image_loader(prev_out_img),
                    image_loader(prev_cont_img), image_loader(pprev_cont_img), style_weight=1e6))
            elif set_prev == 1:
                final_out = unloader(run_style_transfer(image_loader(cont_img),
                    image_loader(styl_img), image_loader(out_img), image_loader(prev_out_img),
                    image_loader(prev_cont_img), image_loader(prev_cont_img), style_weight=1e6, temporal_weight=0))
                set_prev = 2
            elif set_prev == 0:
                final_out = unloader(run_style_transfer(image_loader(cont_img),
                    image_loader(styl_img), image_loader(out_img), image_loader(out_img),
                    image_loader(cont_img), image_loader(cont_img), style_weight=1e6, temporal_weight=0))
                set_prev = 1

            frames.append(np.array(final_out)[:,:,::-1].copy())
            
            # Save the previous frame values
            prev_out_img = final_out
            pprev_cont_img = prev_cont_img
            prev_cont_img = cont_img

            # Load the next frame
            success, cont_img_array = cap.read()
            if seg_cap != None:
                seg_success, cont_seg_array = seg_cap.read()
                if not seg_success:
                    break
        if len(frames) < 1:
            return
        height, width, layers = frames[0].shape
        size = (width,height)
        # Save the video
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, size)
        for f in frames:
            out.write(f)
        out.release()
