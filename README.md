# CS 583 Final Project

Stephen Hansen, Harshwardhan Pande, Trang Bao Ha

Requirements:

```
Anaconda
NVIDIA TITAN GPU + CUDA
Python 3.6 or greater, and pip

conda install pytorch=0.4.1 torchvision cuda91 -y -c pytorch
pip install scikit-umfpack
pip install -U setuptools
pip install cupy
pip install pynvrtc
conda install -c conda-forge ffmpeg
conda install -c conda-forge opencv
```

Env Variables:

```
export ANACONDA=/usr/local/bin/conda
export CUDA_PATH=/usr/local/cuda
export PATH=${ANACONDA}/bin:${CUDA_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${ANACONDA}/lib:${CUDA_PATH}/bin64:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=${CUDA_PATH}/include
```

If you plan to run any of the demo bash scripts, please install ImageMagick and axel as well:

```
sudo apt-get install -y axel imagemagick
```

Before running any scripts, make sure to download the PhotoWCT model by running the following:

```
./download_models.sh
```

We **highly** recommend using the colab notebook in `notebooks/FastVideoStyle.ipynb`. Just upload to Google Colab,
run the cells in the order specified (see instructions in notebook), and make sure that GPU runtime is enabled.
This will allow you to test the different methods relatively easily without needing to go through and set
up all of the dependencies manually.

Python usage:

```
python video_demo.py --fast --nframes 120 --content_video_path videos/video1.mp4 --style_image_path images/style1.png --output_video_path results/demoresultvideo1.avi
```

Flags:

```
--fast : use fast postprocessing smoothing module, produces frames 8x faster with minimal differences
--nframes N : only use the first N frames of the video. if not provided, will stylize entire video (will take a long time!)
--content_video_path : location of content video
--style_image_path : location of style image
--output_video_path : location of where the output stylized video will be stored
```

By default the module will do per-frame stylization. Please enable one of these optional flags to try different
stylization methods:

```
--general_flow : preserve pixels between frames where pixel color does not change
--color_mapping : map input pixel colors to output PhotoWCT model colors
--optical_flow : compute the warped forward and backward flows per frame and blend each together
--smart_optical_flow : use gradient descent to train a model for making a stylized image that resembles the warped previous frame. Warning: much slower than other methods.
--artistic_optical_flow : use gradient descent to train an artistic stylization model on each frame. Warning: much slower than other methods.
```

Source scripts:

```
video_demo.py : controls argument parser, routes args to appropriate method
process_stylization.py : implementations of all video methods
model.py : the gradient descent model for optical flow training and artistic stylization
```

-------------------------------

# ORIGINAL README

[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg)

## FastPhotoStyle

### License
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

<img src="https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/teaser.png" width="800" title="Teaser results"> 


### What's new
 
 | Date     | News |
 |----------|--------------|
 |2018-07-25| Migrate to pytorch 0.4.0. For pytorch 0.3.0 user, check out [FastPhotoStyle for pytorch 0.3.0](https://github.com/NVIDIA/FastPhotoStyle/releases/tag/f33e07f). |
 |          | Add a [tutorial](TUTORIAL.md) showing 3 ways of using the FastPhotoStyle algorithm.|
 |2018-07-10| Our paper is accepted by the ECCV 2018 conference!!! | 


### About

Given a content photo and a style photo, the code can transfer the style of the style photo to the content photo. The details of the algorithm behind the code is documented in our arxiv paper. Please cite the paper if this code repository is used in your publications.

[A Closed-form Solution to Photorealistic Image Stylization](https://arxiv.org/abs/1802.06474) <br> 
[Yijun Li (UC Merced)](https://sites.google.com/site/yijunlimaverick/), [Ming-Yu Liu (NVIDIA)](http://mingyuliu.net/), [Xueting Li (UC Merced)](https://sunshineatnoon.github.io/), [Ming-Hsuan Yang (NVIDIA, UC Merced)](http://faculty.ucmerced.edu/mhyang/), [Jan Kautz (NVIDIA)](http://jankautz.com/) <br>
European Conference on Computer Vision (ECCV), 2018 <br>


### Tutorial

Please check out the [tutorial](TUTORIAL.md).


