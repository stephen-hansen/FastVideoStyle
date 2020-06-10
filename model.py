"""
    model.py
    implements gradient descent training for artistic style
    and temporal loss training based on "Artistic style transfer
    for videos".

    the starting code was reused from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    and additional code was added to handle temporal loss.
"""
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

imsize = 270 if torch.cuda.is_available() else 128  # use small size if no gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image):
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

unloader = transforms.ToPILImage()

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

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

def occlusion(fflow, bflow, prev_bflow):
    """
    Given a forward flow, backwards flow, and previous backwards flow, compute
    the occlusion matrix to determine which pixels are occlusions and should be
    regenerated by the model (has no weight on the gradient descent loss calculation).
    """
    h = fflow.shape[0]
    w = fflow.shape[1]
    img = np.ones((h, w, 3))
    mask = (np.linalg.norm(fflow + bflow, axis=2) > 0.01*(np.linalg.norm(fflow, axis=2) + \
           np.linalg.norm(bflow, axis=2)) + 0.5) | (np.square(bflow[:,:,0] - prev_bflow[:,:,0]) + \
           np.square(bflow[:,:,1] - prev_bflow[:,:,1]) > 0.01*(np.linalg.norm(bflow, axis=2)) + 0.002)
    img[:,:,0] = np.invert(mask)
    img[:,:,1] = np.invert(mask)
    img[:,:,2] = np.invert(mask)
    img *= 255
    print(np.sum(mask))
    return img

def gen_flow(prev_frame_array, curr_frame_array):
    """
    Given two image arrays, compute the forward flow, warped forward flow,
    and backward flow for the images.
    """
    prev_frame_gray = cv2.cvtColor(prev_frame_array,cv2.COLOR_BGR2GRAY)
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
    new_grid[0] += bflow[:,:,0] # move all pixels backwards according to backwards flow
    new_grid[1] += bflow[:,:,1]
    new_grid = np.around(new_grid).astype(np.int64)
    mask = (new_grid[0] < 0) | (new_grid[0] >= width) # prevent out of bounds error
    new_x_grid = np.copy(new_grid[0])
    new_x_grid[mask] = x_grid[mask]
    mask = (new_grid[1] < 0) | (new_grid[1] >= height) # prevent out of bounds error
    new_y_grid =  np.copy(new_grid[1])
    new_y_grid[mask] = y_grid[mask]
    # We now have a warped grid
    # Plug the warped grid into the forward flow function
    fflow = np.copy(flow)
    # create the warped forward flow by applying forward flow on backwards-warped grid
    fflow[y_grid,x_grid] = flow[new_y_grid,new_x_grid]
    return flow, fflow, bflow

def warp(prev_out_array, flow):
    """
    Warp the previous image array according to the specified flow.
    """
    height = prev_out_array.shape[0]
    width = prev_out_array.shape[1]
    xs = np.arange(width)
    ys = np.arange(height)
    grid = np.meshgrid(xs, ys)
    x_grid = grid[0]
    y_grid = grid[1]
    new_grid = np.copy(grid).astype(np.float64)
    # Warp the meshgrid according to the flow.
    new_grid[0] += flow[:,:,0]
    new_grid[1] += flow[:,:,1]
    new_grid = np.around(new_grid).astype(np.int64)
    mask = (new_grid[0] < 0) | (new_grid[0] >= width) # prevent out of bounds error
    new_x_grid = np.copy(new_grid[0])
    new_x_grid[mask] = x_grid[mask]
    mask = (new_grid[1] < 0) | (new_grid[1] >= height) # prevent out of bounds error
    new_y_grid =  np.copy(new_grid[1])
    new_y_grid[mask] = y_grid[mask]
    new_frame_array = np.copy(prev_out_array)
    new_frame_array[new_y_grid,new_x_grid] = prev_out_array[y_grid,x_grid] # perform the warp
    return new_frame_array

class TemporalLoss(nn.Module):
    """
    Measure the MSE loss between the current image and
    the previous image, warped to the current, multiplied
    by the occlusion matrix.
    """
    def __init__(self, target, occlusion):
        super(TemporalLoss, self).__init__()
        self.target = target.detach()
        self.occlusion = occlusion.detach()

    def forward(self, input):
        # occlusion is 0 or 1 here, it's okay to multiply directly here
        self.loss = F.mse_loss(self.occlusion*input, self.occlusion*self.target)
        return input

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
# Using this ReLU layers as suggested by the "Artistic style transfer for videos" paper
#content_layers_default = ['conv_22']
#style_layers_default = ['conv_1', 'conv_6', 'conv_11', 'conv_20', 'conv_29']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, prev_img, prev_content_img,
                               pprev_content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    """
    Build the style model based on the VGG19 CNN, with specified
    normalization parameters, and using a style image, a content image,
    a previous output image, a previous content image, and a second previous
    content image.
    """
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    prev_content_img_arr = prev_content_img.cpu().clone()  # we clone the tensor to not do changes on it
    prev_content_img_arr = prev_content_img_arr.squeeze(0)
    prev_content_img_arr = np.array(unloader(prev_content_img_arr))
    pprev_content_img_arr = pprev_content_img.cpu().clone()  # we clone the tensor to not do changes on it
    pprev_content_img_arr = pprev_content_img_arr.squeeze(0)
    pprev_content_img_arr = np.array(unloader(pprev_content_img_arr))
    content_img_arr = content_img.cpu().clone()  # we clone the tensor to not do changes on it
    content_img_arr = content_img_arr.squeeze(0)
    content_img_arr = np.array(unloader(content_img_arr))
    # Compute forward, warped forward, and backward flows for previous frame and current frame
    flow, fflow, bflow = gen_flow(prev_content_img_arr, content_img_arr)
    # Compute same flows between previous frame and second previous frame
    flow2, fflow2, bflow2 = gen_flow(pprev_content_img_arr, prev_content_img_arr)
    # Determine the occlusion using the warped forward flow, the backwards flow, and previous backwards flow
    occ_img = occlusion(fflow, bflow, bflow2)
    occ_img = Image.fromarray(np.uint8(occ_img)).convert('RGB')
    occ_img = image_loader(occ_img) # Convert occ_img into a tensor
    prev_image = prev_img.cpu().clone()
    prev_image = prev_image.squeeze(0)
    prev_image = np.array(unloader(prev_image))
    prev_warped = warp(prev_image, flow) # Warp the previous frame with the calculated flow
    prev_warped = Image.fromarray(cv2.cvtColor(prev_warped[:,:,::-1], cv2.COLOR_BGR2RGB))
    prev_warped = image_loader(prev_warped) # Convert prev_warped into a tensor
    target = prev_warped.detach()
    target2 = occ_img.detach()

    # set up temporal loss. Use previously warped frame and occlusion matrix as targets.
    temporal_loss = TemporalLoss(target, target2).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    # Select the style and content layers, put them in nn.Sequential
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

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) or isinstance(model[i], TemporalLoss):
            break

    # Slice the sequential module
    model = model[:(i + 1)]

    return model, style_losses, content_losses, temporal_loss

def get_input_optimizer(input_img):
    """
    Using LBFGS to optimize as recommended in "Artistic style transfer for videos"
    """
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(content_img, style_img, input_img, prev_img, prev_content_img, 
                       pprev_content_img, num_steps=500, style_weight=1e4, 
                       content_weight=5e0, temporal_weight=2e2):
    """
    Train the model with the given content image, style image, input image, previous output image, previous content
    image, and second previous content image.
    """
    print('Building the style transfer model...')
    # Get the model
    model, style_losses, content_losses, temporal_loss = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, prev_img, prev_content_img, pprev_content_img)
    optimizer = get_input_optimizer(input_img)
    print('Optimizing..')
    run = [0]
    # Perform gradient descent
    with torch.enable_grad():
        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                temporal_loss(input_img)
                model(input_img)
                style_score = 0
                content_score = 0
                temporal_score = 0
                # Add in temporal loss
                temporal_score += temporal_loss.loss
                # Add style, content losses, to prevent complete warping into the previous frame.
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight
                temporal_score *= temporal_weight

                loss = style_score + content_score + temporal_score
                loss.backward()

                run[0] += 1
                # Print the loss every 50 runs
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss : {:4f} Temporal Loss : {:4f}'.format(
                        style_score.item(), content_score.item(), temporal_score.item()))
                    print()

                return style_score + content_score + temporal_score

            optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    image = input_img.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    return image # Return the image tensor
