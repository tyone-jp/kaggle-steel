import numpy as np
from PIL import Image
import pandas as pd

def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(rle, input_shape):
    height, width = input_shape[:2]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1

    return mask.reshape((height,width),order='F')

def build_masks(rles, input_shape):
    depth = len(rles)
    masks = np.zeros((*input_shape, depth))

    for i, rle in enumerate(rles):
        if type(rle) is str:
            masks[:, :, i] = rle2mask(rle, input_shape)
            
    return masks

def build_rles(masks):
    height, width, depth = masks.shape
    
    rles = [mask2rle(masks[:, :, i]) for i in range(depth)]

    return rles

def rle2maskResize(rle):
    if (pd.isnull(rle)) | (rle==''):
        return np.zeros((128,800),dtype=np.uint8)

    height=256
    width=1600
    mask=np.zeros(width*height,dtype=np.uint8)

    array=np.asarray([int(x) for x in rle.split()])
    starts=array[0::2]-1
    lengths=array[1::2]
    for index,start in enumerate(starts):
        mask[int(start):int(start+lengths[index])]=1

    return mask.reshape((height,width),order='F')[::2,::2]


