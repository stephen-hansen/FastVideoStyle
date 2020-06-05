import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import cv2
import numpy as np

imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image):
    if max(image.size) > imsize:
        size = imsize
    else:
        size = max(image.size)

    in_transform = transforms.Compose([
        transforms.Resize((size, int(1.5*size))),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    
    return image

def unloader(tensor):
    image = tensor.to("cpu").clone().detach()
    image  = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229,0.224,0.225)) + np.array((0.485,0.456,0.406))
    image = image.clip(0, 1)
    return image

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(tensor):
    _, n_filters, h, w = tensor.size()
    tensor = tensor.view(n_filters, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

def occlusion(prev_frame, curr_frame):
    pass

def warp(prev_frame, curr_frame):
    prev_image = prev_frame.cpu().clone()
    prev_image = prev_image.squeeze(0)
    prev_frame_array = np.array(unloader(prev_image))
    prev_frame_gray = cv2.cvtColor(prev_frame_array,cv2.COLOR_BGR2GRAY)
    curr_image = curr_frame.cpu().clone()
    curr_image = curr_image.squeeze(0)
    curr_frame_array = np.array(unloader(curr_image))
    curr_frame_gray = cv2.cvtColor(curr_frame_array,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, curr_frame_gray, None, 0.5, 5, 15, 3, 5, 1.1, 0)
    bflow = cv2.calcOpticalFlowFarneback(curr_frame_gray, prev_frame_gray, None, 0.5, 5, 15, 3, 5, 1.1, 0)
    height = prev_frame_array.shape[0]
    width = prev_frame_array.shape[1]
    xs = np.arange(width)
    ys = np.arange(height)
    grid = np.meshgrid(xs, ys)
    x_grid = grid[0]
    y_grid = grid[1]
    new_grid = np.copy(grid).astype(np.float64)
    new_grid[0] += flow[:,:,0]
    new_grid[1] += flow[:,:,1]
    new_grid = np.around(new_grid).astype(np.int64)
    mask = (new_grid[0] < 0) | (new_grid[0] >= width)
    new_x_grid = np.copy(new_grid[0])
    new_x_grid[mask] = x_grid[mask]
    mask = (new_grid[1] < 0) | (new_grid[1] >= height)
    new_y_grid =  np.copy(new_grid[1])
    new_y_grid[mask] = y_grid[mask]
    curr_frame_array[new_y_grid,new_x_grid] = prev_frame_array[y_grid,x_grid]
    return torch.from_numpy(curr_frame_array)

class TemporalLoss(nn.Module):
    def __init__(self, target_feature):
        super(TemporalLoss, self).__init__()
        self.target = target_feature.detach()

    def forward(self, input):
        D = input.numel()
        #warped = warp(self.target, input)
        self.loss = torch.sum((F.mse_loss(input, self.target)))/D

cnn = models.vgg19(pretrained=True).features.to(device).eval()
for param in cnn.parameters():
    param.requires_grad_(False)

normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they c
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

#content_layers_default = ['relu4_2']
#style_layers_default = ['relu1_1','relu2_1','relu3_1','relu4_1','relu5_1']
content_layers_default = ['relu_23']
style_layers_default = ['relu_2', 'relu_7', 'relu_12', 'relu_21', 'relu_30']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, prev_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []
    temporal_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        i += 1
        if isinstance(layer, nn.Conv2d):
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    target = model(prev_img).detach()
    temporal_loss = TemporalLoss(target)
    model.add_module('temporal_loss', temporal_loss)
    temporal_losses.append(temporal_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) or isinstance(model[i], TemporalLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses, temporal_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(content_img, style_img, input_img, prev_img, num_steps=300,
                       style_weight=20, content_weight=1, temporal_weight=200):
    print('Building the style transfer model...')
    model, style_losses, content_losses, temporal_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img, prev_img)
    optimizer = get_input_optimizer(input_img)
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            temporal_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            for tl in temporal_losses:
                temporal_score += tl.loss

            style_score *= style_weight
            content_score *= content_weight
            temporal_score *= temporal_weight

            loss = style_score + content_score + temporal_score
            loss.backward(retain_graph=True)

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss : {:4f} Temporal Loss : {:4f}'.format(
                    style_score.item(), content_score.item(), temporal_score.item()))
                print()

            return style_score + content_score + temporal_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img
