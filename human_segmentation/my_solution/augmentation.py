import torch.utils.data as dt
import torch
import os
import random
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import rotate
from scipy.ndimage import zoom

to_img = ToPILImage()


def sp_noise(image,prob):
  
    # Add salt&pepper noise to image differently for every channel
    # Returns burred image
    # Input parameters:
    #   image - input PIL image
    #   prob - noise probability
    
    img = np.array(image)
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j][k] = 0
                elif rdn > thres:
                    output[i][j][k] = 255
                else:
                    output[i][j][k] = img[i][j][k]
    return to_img(output)

def sp_noise1(image,prob):
  
    # Adds salt&pepper noise to image
    # Returns blurred image
    # Input parameters:
    #   image - input PIL image
    #   prob - noise probability
    
    img = np.array(image)
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return to_img(output)


def gaussian_noise(image, std):
    # Adds gaussian noise to image
    # Returns blurred image
    # Input parameters:
    #   image - input PIL image
    #   std - noise variance
    img = np.array(image,np.uint8)
    output = np.clip((np.random.normal(0.0, std, img.shape).astype(np.uint8) + img), 0, 255).astype(np.uint8)            
    return to_img(output)


def random_shift(image, max_shift):
    # Shifts full image randomly
    # Returns shifted image
    # Input parameters:
    #   image - input PIL image
    #   max_shift - maximal shift value
    img = np.array(image, np.uint8)
    output = np.clip(np.random.randint( - max_shift, max_shift, size=1)[0] + img, 0, 255).astype(np.uint8)            
    return to_img(output)

def random_color(image):
    # Changes brightness of image's channels
    # Returns changed image
    # Input parameters:
    #   image - input PIL image
    img = np.swapaxes(np.array(image), 0, 2)
    rdn = np.random.uniform(size=3)
    for i in range(0, img.shape[0]):
       img[i] = np.around(img[i] * rdn[i])    
    return to_img( np.swapaxes(img, 0, 2) )

def color_shuffle(image):
    # Swap image's channels
    # Returns changed image
    # Input parameters:
    #   image - input PIL image
    img = np.swapaxes(np.array(image), 0, 2)
    np.random.shuffle(img)
    return to_img( np.swapaxes(img, 0, 2) )

def random_crop(image, mask, max_size):
    # Crops image and add black space
    # Returns changed image
    # Input parameters:
    #   image - input PIL image
    #   mask - input PIL image
    #   max_size - maximal size of cutted border
    img = np.array(image)
    msk = np.array(mask)

    new_img = np.zeros(img.shape, np.uint8)
    new_msk = np.zeros(msk.shape, np.uint8)

    shape1 = img.shape
    shape2 = msk.shape

    shift1 = np.random.randint(0, max_size, size=1)[0]
    shift2 = np.random.randint(0, max_size, size=1)[0]
    shift3 = np.random.randint(0, max_size, size=1)[0]
    shift4 = np.random.randint(0, max_size, size=1)[0]

    pos1 = 0
    pos2 = 0
    if shift1 + shift2 > 0:
        pos1 = np.random.randint(0,shift1 + shift2, size=1)[0]
    if shift3 + shift4 > 0:
        pos2 = np.random.randint(0,shift3 + shift4, size=1)[0]

    img = img[shift1: shape1[0] - shift2, shift3: shape1[1] - shift4, :]
    msk = msk[shift1: shape2[0] - shift2, shift3: shape2[1] - shift4, :]

    new_img[pos1:pos1+img.shape[0], pos2: pos2 + img.shape[1], :] = img
    new_msk[pos1:pos1+msk.shape[0], pos2: pos2 + msk.shape[1], :] = msk

    return to_img(new_img.astype(np.uint8)), to_img(new_msk.astype(np.uint8))


def random_zoom(image, mask, max_size):
    # Crops image and stretchs it
    # Returns changed image
    # Input parameters:
    #   image - input PIL image
    #   mask - input PIL image
    #   max_size - maximal size of cutted border
    
    img = np.array(image)
    msk = np.array(mask)

    shape1 = img.shape
    shape2 = msk.shape
    shift1 = np.random.randint(0, max_size, size=1)[0]
    shift2 = np.random.randint(0, max_size, size=1)[0]
    shift3 = np.random.randint(0, max_size, size=1)[0]
    shift4 = np.random.randint(0, max_size, size=1)[0]

    img = img[shift1: shape1[0] - shift2, shift3: shape1[1] - shift4, :]
    msk = msk[shift1: shape2[0] - shift2, shift3: shape2[1] - shift4, :]

    a = to_img(img).resize((shape1[1], shape1[0]),Image.BICUBIC)
    b = to_img(msk).resize( (shape2[1], shape2[0]) ,Image.BICUBIC)
    return a, b

def random_rotate(image, mask, max_degrees):
    # Rotates image
    # Returns changed image
    # Input parameters:
    #   image - input PIL image
    #   mask - input PIL image
    #   max_degrees - maximal value of angle

    angle = np.random.randint(-max_degrees, max_degrees)
    shape = image.size

    shift1 = np.random.randint(shape[0]/2 - 10, shape[0]/2 + 10, size=1)[0]

    shift2 = np.random.randint(shape[1]/2 - 10, shape[1]/2 + 10, size=1)[0]
    image = rotate(image, angle, resample=False, expand=False, center=(shift1, shift2))
    mask = rotate(mask, angle, resample=False, expand=False, center=(shift1, shift2))

    return image, mask
