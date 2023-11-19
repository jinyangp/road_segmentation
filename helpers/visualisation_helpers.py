'''This file contains the implementation code for helper functions used to visualise the output of the model. These helper functions include making a prediction overlay with the image and many more.
'''
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def concatenate_seg_maps(gt_seg_map, pred_seg_map):
    
    """Visualise the actual segmentation map and the predicted segmentation map side by side 
    
    Args:
        gt_seg_map: numpy array, ground truth segmentation map
        pred_seg_map: numpy array, predicted segmentation map
    
    Returns:
        None
    """
    
    fig, axs = plt.subplots(1, 2, figsize = (8, 8))
    axs[0].imshow(gt_seg_map, cmap = 'gray')
    axs[1].imshow(pred_seg_map, cmap = 'gray')


def concatenate_img_and_segmaps(img, seg_map):
    
    """Visualise the actual segmentation map and the predicted segmentation map side by side 
    
    Args:
        img: numpy array, image
        pred_seg_map: numpy array, predicted segmentation map
    
    Returns:
        None
    """
    
    fig, axs = plt.subplots(1, 2, figsize = (8, 8))
    axs[0].imshow(img)
    axs[1].imshow(pred_seg_map, cmap = 'gray')

    
def segmap_to_colourmask(img_width, img_height, seg_map):
    
    """Converts segmentation map, a numpy array, into a colour mask 
    
    Args:
        img_width: int, width of predicted segmentation map
        img_height: int, height of predicted segmentation map
        seg_map: numpy array, segmentation map
        
    Returns:
        colour_mask: numpy array, pixel values of colour mask
    """
    
    colour_mask = np.zeros((img_height, img_width, 3), dtype = np.uint8)
    
    road_idxs = np.argwhere(seg_map == 1.)
    road_color = [255, 0, 0] # colour the road as red

    # Set the pixels corresponding to road class in the color mask
    colour_mask[road_idxs[:, 0], road_idxs[:, 1]] = road_color

    return colour_mask


def make_img_overlay(img, colour_mask):
    
    """Overlays colour mask onto the original image 
    
    Args:
        img: numpy array, pixel values of image
        colour_mask: numpy array, pixel values of colour mask
    Returns:
        overlayed_img: numpy array, pixel values of overlayed image
    """

    background = Image.fromarray(img, "RGB").convert("RGBA")
    overlay = Image.fromarray(colour_mask, "RGB").convert("RGBA")
    overlayed_img = Image.blend(background, overlay, 0.2)
    return overlayed_img
