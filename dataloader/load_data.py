'''This file contains the implementation code for all functions pertaining to the creation and loading in of dataset used to train or evaluate the model.
'''

import os 
import tensorflow as tf
import numpy as np
from patchify import patchify

from augmentation_fncs import *

# Function to determine class weights
def get_class_weights(imgs_dir, gts_dir):
    
    '''Implementation code to determine the proportion of pixels labelled as road or background. This is achieved by calculating the proportion of pixels labelled as road and background in the provided groundtruth images. Then, the proportions are swapped and the inverse is taken as the weight to give greater importance to the 'road' class. This is due to the dataset being imbalanced with the majority of pixels being classified as the 'background' class.
        
    Args:
        imgs_dir: directory name where images are stored
        gts_dir: directory name where groundtruths are stored
        
    Returns:
        road_proportion, background_proportion: float, determined proportions of each class
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
    
    '''Implementation code to read in and preprocess an image from the directory. 
        
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
    
    '''Implementation code to read in and preprocess groundtruth image from the directory. 
        
    Args:
        file_name: file name of groundtruth to be preprocessed
        one_hot: boolean, boolean flag to determine if groundtruths should be one hot encoded. The groundtruths only need to be one hot encoded if weighted IOU loss is used.
        
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


def generate_patches(ds, patch_size, overlap, one_hot = False, padding = False):

    '''Implementation code to generate image patches for both images and groundtruths from a loaded tf.Dataset. To be used when training.
        
    Args:
        ds: tf.data.Dataset, containing images and groundtruths
        patch_size: shape of patches of (patch_height, patch_width, patch_channels)
        overlap: int, number of overlapping pixels per patch
        one_hot: boolean, whether groundtruth images are one hot encoded
        padding: boolean, whether the images are padded
        
    Returns:
       patched_imgs_ds: tf.data.Dataset, containing patched images and groundtruths
    '''  
    
    img_patches_arr = []
    gt_patches_arr = []
    
    height, width = patch_size[0], patch_size[1]
    img_patch_size = (height, width, 3)
    
    # Iterate through images in the dataset
    for img, gt in ds:
        
        img = img.numpy()
        gt = gt.numpy()

        # Add padding to image to ensure all pixels covered
        if padding:
            pad_height = patch_size[0] - (img.shape[0] % (patch_size[0] - overlap))
            pad_width = patch_size[1] - (img.shape[1] % (patch_size[1] - overlap))
            img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
            gt = np.pad(gt,((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        
        if one_hot:
            gt_patch_size = (height, width, 2)
        else:
            gt_patch_size = (height, width, 1)
  
        # Visualise the patches as a grid, at row i and column j
        img_patches = patchify(img, img_patch_size, step = img_patch_size[0] - overlap)
        gt_patches = patchify(gt, gt_patch_size, step = gt_patch_size[0] - overlap)
    
        num_rows = img_patches.shape[0]
        num_cols = img_patches.shape[1]
        
        for i in range(num_rows):
            for j in range(num_cols):
                
                cur_img_patch = img_patches[i,j, 0]
                img_patches_arr.append(cur_img_patch)
                cur_gt_patch = gt_patches[i,j, 0]
                gt_patches_arr.append(cur_gt_patch)
    
    patched_imgs_ds = tf.data.Dataset.from_tensor_slices((np.array(img_patches_arr), np.array(gt_patches_arr)))
    return patched_imgs_ds
 
def generate_patches_imgs(ds, patch_size, overlap, padding = False):

    '''Implementation code to generate image patches from a loaded tf.Dataset. This function is similar to generate_patches but the dataset received in this function only contains images. To be used when generating submissions.
        
    Args:
        ds: tf.data.Dataset, containing images and groundtruths
        patch_size: shape of patches of (patch_height, patch_width, patch_channels)
        overlap: int, number of overlapping pixels per patch
        padding: boolean, whether the images are padded
        
    Returns:
       patched_imgs_ds: tf.data.Dataset, containing patched images
    '''  
    
    img_patches_arr = []
    
    height, width = patch_size[0], patch_size[1]
    img_patch_size = (height, width, 3)
    
    # Iterate through images in the dataset
    for img in ds:
        
        img = img.numpy()
    
        # Add padding to image to ensure all pixels covered
        if padding:
            pad_height = patch_size[0] - (img.shape[0] % (patch_size[0] - overlap))
            pad_width = patch_size[1] - (img.shape[1] % (patch_size[1] - overlap))
            img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
              
        # Visualise the patches as a grid, at row i and column j
        img_patches = patchify(img, img_patch_size, step = img_patch_size[0] - overlap)
        
        num_rows = img_patches.shape[0]
        num_cols = img_patches.shape[1]
        
        for i in range(num_rows):
            for j in range(num_cols):
                
                cur_img_patch = img_patches[i,j, 0]
                img_patches_arr.append(cur_img_patch)
    
    patched_imgs_ds = tf.data.Dataset.from_tensor_slices(np.array(img_patches_arr))
    return patched_imgs_ds
    
def reconstruct_image(patches, patch_size, original_size, overlap, padding = False):
    
    '''Reconstruct images from generated patches
    
    Args:
        patches: nd.array, of shape (num_patches, patch_height, patch_width, patch_channel)
        patch_size: nd.array, shape of each patch
        original_size: nd.array, shape of original image
        overlap: int, number of overlapping pixels per patch
        padding: boolean, if the image is padded
        
    Returns:
        combined_patches_arr: nd.array, of shape same as original size if padding was applied else shape of patches combined containing reconstructed image's pixel values
    
    '''
    original_height, original_width, original_channels = original_size[0], original_size[1], original_size[2]
    patch_height, patch_width, patch_channels  = patch_size[0], patch_size[1], patch_size[2]
    
    # Get array to store reconstructed image
    reconstructed_arr = np.zeros(shape = original_size, dtype = np.float32)
    
    num_rows = original_height // patch_height
    num_cols = original_width // patch_width
    
    if padding:
        num_rows += 1
        num_cols += 1
        
    # Array to store combined patches
    combined_patches_height = patch_height * num_rows
    combined_patches_width = patch_width * num_cols
    combined_patches_arr = np.zeros(shape = (combined_patches_height, combined_patches_width, original_channels), dtype = np.float32)
    
    # Combine the patches
    for i in range(num_rows):
        
        # if no padding, reconstructed image will be smaller than original
        # if have padding, reconstructed image will be larger than original
        row_start = i*patch_height
        row_end = (i+1)*patch_height

        for j in range(num_cols):
            
            col_start = j*patch_width
            col_end = (j+1)*patch_width
            
            cur_patch_idx = (i*num_rows)+j
            
            combined_patches_arr[row_start:row_end, col_start:col_end,:] = patches[cur_patch_idx]
    
    if padding:
        return combined_patches_arr[:original_height, :original_width,:]
    
    return combined_patches_arr


# Function to generate the augmented dataset given the original dataset
def create_augmented_dataset(dataset, num_images):

    '''Implementation code to augment the originally loaded dataset.
        
    Args:
        dataset: original dataset to be augmented
        num_images: List[int] of (num_brightness, num_rotation, num_noise). This indicates number of samples augmented by using augmentation techniques related to brightness, rotation and adding noise.
        
    Returns:
        augmented_dataset: tf.data.Dataset instance, an instance of the augmented dataset
    '''    
        
    # Create a copy of the original dataset
    images = np.array([img for img, _ in dataset])
    gts = np.array([gt for _, gt in dataset])

    augmented_dataset = tf.data.Dataset.from_tensor_slices((images, gts))
    
    # Get number of augmented samples for each augmentation categories
    num_brightness, num_rotation, num_noise = num_images[0], num_images[1], num_images[2]
    
    # Brightness, contrast and saturation
    for i in range(num_brightness):

        brightness_augmented_ds = dataset.map(adjust_brightness)
        augmented_dataset = augmented_dataset.concatenate(brightness_augmented_ds)

        contrast_augmented_ds = dataset.map(adjust_contrast)
        augmented_dataset = augmented_dataset.concatenate(contrast_augmented_ds)

        saturation_augmented_ds = dataset.map(adjust_saturation)
        augmented_dataset = augmented_dataset.concatenate(saturation_augmented_ds)
    

    # Apply rotation to images
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
    

    # Add noise to images
    for i in range(num_noise):

        gaussian_augmented_ds = dataset.map(add_gaussian_noise)
        augmented_dataset = augmented_dataset.concatenate(gaussian_augmented_ds)

        cloud_noise_augmented_ds = dataset.map(cloud_noise)
        augmented_dataset = augmented_dataset.concatenate(cloud_noise_augmented_ds)

        black_patch_augmented_ds = dataset.map(black_patch_noise)
        augmented_dataset = augmented_dataset.concatenate(black_patch_augmented_ds)
                    
    return augmented_dataset
