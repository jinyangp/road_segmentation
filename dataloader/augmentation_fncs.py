'''This file contains the implementation code for the functions used to augment the dataset.
'''

import random
import numpy as np
import tensorflow as tf

'''Augmentation related to brightness, contrast and saturation'''
def adjust_brightness(img, gt):
    
    # delta should be in the range (-1,1), as it is added to the image in floating point representation, where pixel values are in the [0,1) range
    random_delta = random.uniform(-0.1, 0.25)
    img = tf.image.adjust_brightness(img, random_delta)
    
    return img, gt


def adjust_contrast(img, gt):
    
    # contrast_factor must be in the interval (-inf, inf).
    random_contrast = random.uniform(0., 0.5)
    img = tf.image.adjust_contrast(img, random_contrast)
    
    return img, gt


def adjust_saturation(img, gt):
    
    # saturation_factor must be in the interval [0, inf).
    random_saturation = random.uniform(0., 1.)
    img = tf.image.adjust_saturation(img, random_saturation)
    
    return img, gt


'''Augmentation related to rotation and flipping'''
def flip_left_right(img, gt):
    
    img = tf.image.flip_left_right(img)
    gt = tf.image.flip_left_right(gt)

    return img, gt


def flip_up_down(img, gt):
    
    img = tf.image.flip_up_down(img)
    gt = tf.image.flip_up_down(gt)

    return img, gt


def rotate_img(img, gt, k):
    
    img = tf.image.rot90(img, k)
    gt = tf.image.rot90(gt, k)
    
    return img, gt


def transpose_img(img, gt):
    
    img = tf.image.transpose(img)
    gt = tf.image.transpose(gt)
    
    return img, gt


'''Augmentation related to adding noise to image (Gaussian, Salt and Pepper, Cropped out black patches'''
def add_gaussian_noise(img, gt):
    
    # Generate gaussian distribution
    mean, sigma = 0., 0.25
    gaussian_noise = np.random.normal(mean, sigma, (img.shape[0],img.shape[1], img.shape[2]))
    gaussian_noise = tf.convert_to_tensor(gaussian_noise, np.float32)
    img = img + gaussian_noise
    
    return img, gt


def salt_and_pepper_noise(img, gt):
    
#     prob_salt, prob_pepper = 0.05, 0.05
#     random_values = tf.random.uniform(shape=img[0, ..., -1:].shape)
#     img = tf.where(random_values < prob_salt, 1., img)
#     img = tf.where(1 - random_values < prob_pepper, 0., img)
    
    # Generate binomial distribution to select pixels that will be corrupted
    num_trials = 100
    p_selected = 0.8
    p_salt_or_pepper = 0.5
    shape = img.shape

    # Create a stateless random generator and obtain a seed
    seed = tf.constant([1, 2], dtype=tf.int32)  # Can use any tuple of two integers as the seed

    # Generate binomial random variables without setting a seed
    mask_select = tf.cast(tf.random.stateless_binomial(shape=shape, seed=seed, counts=num_trials, probs=p_selected), dtype = tf.float32)
    mask_noise = tf.cast(tf.random.stateless_binomial(shape=shape, seed=seed, counts=num_trials, probs=p_salt_or_pepper), dtype = tf.float32)
    mask_select = tf.where(mask_select > num_trials*p_selected, 1.0, 0.0)
    mask_noise = tf.where(mask_noise > num_trials*p_salt_or_pepper, 1.0, 0.0)
    
    # Add corruption if pixel is selected otherwise return the original pixel
    img = img * (1. - mask_select) + mask_noise * mask_select
    
    return img, gt


def black_patch_noise(img, gt):
    
    # Specify patch parameters
    patch_size = [150, 150]  # height and width of the patch
    img_height, img_width = img.shape[0], img.shape[1]
    
    max_height_start = img_height - 1 - patch_size[0] # -1 since 0-indexed
    max_width_start = img_width - 1 - patch_size[1]
    
    height_start = random.randint(0, max_height_start)
    width_start = random.randint(0, max_width_start)
    
    start_position = [height_start, width_start]  # starting position of the patch

    patch_height_indices, patch_width_indices = tf.meshgrid(
    tf.range(start_position[0], start_position[0] + patch_size[0]),
    tf.range(start_position[1], start_position[1] + patch_size[1]),
    indexing='ij')
    
    indices_to_update = tf.stack([patch_height_indices, patch_width_indices], axis=-1)
    
    # Create masks
    img_mask = tf.scatter_nd(indices_to_update, tf.ones(tf.shape(indices_to_update)[:-1]), shape=tf.shape(img)[:-1])
    img_mask = tf.cast(img_mask, dtype=tf.bool)
    gt_mask = tf.scatter_nd(indices_to_update, tf.ones(tf.shape(indices_to_update)[:-1]), shape=tf.shape(gt)[:-1])
    gt_mask = tf.cast(gt_mask, dtype=tf.bool)
    
    # Update at specified indices
    augmented_img = tf.where(img_mask[..., tf.newaxis], 0.0, img)
    augmented_gt = tf.where(gt_mask[..., tf.newaxis], 0.0, gt)
    
    return augmented_img, augmented_gt


def cloud_noise(img, gt):
    
    # Rescale images back to 255. temporarily to generate cloud overlay
    img = img * 255.
    
    # Generate a random factor to determine opacity of clouds
    opacity = random.uniform(0.3, 0.7)  # Set a low opacity

    # Generate cloud overlay
    cloud_overlay = tf.ones_like(img, dtype=tf.float32) * opacity
    cloud_overlay = tf.random.normal(shape=tf.shape(img), mean=1.0, stddev=0.2, dtype=tf.float32) * cloud_overlay

    # Blend the original image with the cloud overlay
    img = img + cloud_overlay * (255. - img)

    # Clip values to the valid range [0, 255] and scale back to [0,1]
    img = tf.clip_by_value(img, 0, 255.)
    img = tf.cast(img, tf.float32) / 255.0

    return img, gt