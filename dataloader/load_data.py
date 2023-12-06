'''This file contains the implementation code for loading in the original and augmented dataset from the images directory.
'''

import os 
import tensorflow as tf
import numpy as np

from augmentation_fncs import *

# Function to determine class weights
def get_class_weights(imgs_dir, gts_dir):
    
    '''Implementation code to determine class weights for road and background.
        
    Args:
        imgs_dir: directory name where images are stored
        gts_dir: directory name where groundtruths are stored
        
    Returns:
        background_weight, road_weight: float, determined weights of each class
    '''
    
    dataset = create_dataset(imgs_dir, gts_dir, True)
    num_samples = len(dataset)
    
    road_pixel_prop = 0.
    bg_pixel_prop = 0.

    for sample_img, sample_gt in dataset:
    
        sample_gt = sample_gt.numpy()

        sample_gt = np.argmax(sample_gt, axis = -1)

        num_road_pixels = np.count_nonzero(sample_gt)
        num_bg_pixels = np.count_nonzero(sample_gt == 0.)
        ttal_pixels = sample_gt.shape[0]*sample_gt.shape[1]

        road_pixel_proportion = num_road_pixels/ttal_pixels
        bg_pixel_proportion = num_bg_pixels/ttal_pixels

        road_pixel_prop += road_pixel_proportion
        bg_pixel_prop += bg_pixel_proportion

    return road_pixel_prop/num_samples, bg_pixel_prop/num_samples

    
# Load images from filepath
def load_image(file_name):
    
    '''Implementation code to preprocess images. 
        
    Args:
        file_name: file name of image to be preprocessed
        
    Returns:
        tensor: tensor containing pixel values of processed image
    '''    

    raw = tf.io.read_file(file_name)
    tensor = tf.io.decode_image(raw)
    tensor = tf.cast(tensor, tf.float32)
    tensor = tf.cast(tensor, tf.float32) / 255.0
    
    return tensor


def load_gt(file_name, one_hot = False):
    
    '''Implementation code to preprocess groundtruth images. 
        
    Args:
        file_name: file name of groundtruth to be preprocessed
        
    Returns:
        tensor: tensor containing pixel values of processed groundtruth
    '''    
    
    raw = tf.io.read_file(file_name)
    tensor = tf.io.decode_image(raw)
    
    # Normalize pixel values to [0, 1]
    tensor = tf.cast(tensor, tf.float32)
    threshold = 1.0
    tensor = tf.where(tensor >= threshold, 1., 0.)
    
    if one_hot:
        tensor = tf.cast(tensor, dtype = tf.int32)
        tensor = tf.one_hot(tensor[:,:,0], 2, axis = -1)

    tensor = tf.cast(tensor, tf.float32)
    
    return tensor


def set_shapes(img, gt, one_hot = False):
    
    img.set_shape([400, 400, 3])
    
    if one_hot:
        gt.set_shape([400, 400, 2])
    else:
        gt.set_shape([400, 400, 1])
        
    return img, gt


def create_dataset(imgs_dir, gts_dir, one_hot = False):
    
    '''Implementation code to load in original dataset from file directory. 
        
    Args:
        imgs_dir: directory name where images are stored
        gts_dir: directory name where groundtruths are stored
        
    Returns:
        dataset: tf.data.Dataset instance, an instance of the original dataset
    '''    
        
    img_filenames = [os.path.join(imgs_dir, file_name) for file_name in os.listdir(imgs_dir)]
    gt_filenames = [os.path.join(gts_dir, file_name) for file_name in os.listdir(gts_dir)]
    
    dataset = tf.data.Dataset.from_tensor_slices((img_filenames, gt_filenames))
    dataset = dataset.map(lambda img, gt: (load_image(img), load_gt(gt, one_hot)))
    dataset = dataset.map(lambda img, gt: set_shapes(img, gt, one_hot))
    return dataset
                          

# Function to generate the augmented dataset given the original dataset
def create_augmented_dataset(dataset, num_images):

    '''Implementation code to augment the originally loaded dataset.
        
    Args:
        dataset: original dataset to be augmented
        num_images: List[int] of (num_brightness, num_rotation, num_noise)
        
    Returns:
        augmented_dataset: tf.data.Dataset instance, an instance of the augmented dataset
    '''    
        
    # Create a copy of the original dataset
    images = np.array([img for img, _ in dataset])
    gts = np.array([gt for _, gt in dataset])

    augmented_dataset = tf.data.Dataset.from_tensor_slices((images, gts))
    
    # Get number of augmented samples for each augmentation categories
    num_brightness, num_rotation, num_noise = num_images[0], num_images[1], num_images[2]
    
    # Brightness, contrast and saturation (2 different brightness, 2 different contrast, 2 different saturations)
    for i in range(num_brightness):

        brightness_augmented_ds = dataset.map(adjust_brightness)
        augmented_dataset = augmented_dataset.concatenate(brightness_augmented_ds)

        contrast_augmented_ds = dataset.map(adjust_contrast)
        augmented_dataset = augmented_dataset.concatenate(contrast_augmented_ds)

        saturation_augmented_ds = dataset.map(adjust_saturation)
        augmented_dataset = augmented_dataset.concatenate(saturation_augmented_ds)
    

    # Apply rotation to images (1 flip_left_right, 1 flip_up_down, 1 flip 90 degree, 1 flip 270 degree, 1 transpose)
    for i in range(num_rotation):

        flip_lr_ds = dataset.map(flip_left_right)
        augmented_dataset = augmented_dataset.concatenate(flip_lr_ds)

        flip_updown_ds = dataset.map(flip_up_down)
        augmented_dataset = augmented_dataset.concatenate(flip_updown_ds)

        rotated_90_ds = dataset.map(lambda img, gt: rotate_img(img, gt, 1))
        augmented_dataset = augmented_dataset.concatenate(rotated_90_ds)

        rotated_270_ds = dataset.map(lambda img, gt: rotate_img(img, gt, 3))
        augmented_dataset = augmented_dataset.concatenate(rotated_270_ds)

        transposed_ds = dataset.map(transpose_img)
        augmented_dataset = augmented_dataset.concatenate(transposed_ds)
    

    # Add noise to image (3 Gaussian noise, 3 cloud noise, 3 random black patches)
    for i in range(num_noise):

        gaussian_augmented_ds = dataset.map(add_gaussian_noise)
        augmented_dataset = augmented_dataset.concatenate(gaussian_augmented_ds)

        cloud_noise_augmented_ds = dataset.map(cloud_noise)
        augmented_dataset = augmented_dataset.concatenate(cloud_noise_augmented_ds)

        black_patch_augmented_ds = dataset.map(black_patch_noise)
        augmented_dataset = augmented_dataset.concatenate(black_patch_augmented_ds)
                    
    return augmented_dataset
