'''This python script generates the submission csv file.'''

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

sys.path.append('attention_unet')
sys.path.append('u_net')
sys.path.append('dataloader')
sys.path.append('helpers')

from attention_unet.cbam_unet import *
from u_net.unet import *
from dataloader.load_data import *
from helpers.mask_to_submission import *

# Directory paths
train_imgs_dir = os.path.join(os.getcwd(), 'datasets', 'train', 'image')
train_gts_dir = os.path.join(os.getcwd(), 'datasets', 'train', 'groundtruth')
test_imgs_dir = os.path.join(os.getcwd(), 'datasets', 'test', 'image')
test_output_dir = 'submission_results'

def get_imgs_from_dir(img_dir):
    
    '''Returns a tf dataset from the directory of images'''
    # List only files with a specific extension (e.g., '.png')
    files = tf.io.gfile.glob(f"{test_imgs_dir}/*.png")
    # Exclude directories
    imgs = [file for file in files if not tf.io.gfile.isdir(file)]
    #imgs = [os.path.join(img_dir, file_name) for file_name in os.listdir(img_dir)]
    imgs_ds = tf.data.Dataset.from_tensor_slices(imgs)
    imgs_ds = imgs_ds.map(load_image)
    
    return imgs_ds
    

def generate_submissions(trial_no):
    
    # load in images
    imgs_ds = get_imgs_from_dir(test_imgs_dir)
    # Convert to a format that can be passed into model
    imgs_ds = imgs_ds.batch(1)

    # Get class weights
    road_pixel_prop, bg_pixel_prop = get_class_weights(train_imgs_dir, train_gts_dir)
    road_weight = 1./road_pixel_prop
    bg_weight = 1./bg_pixel_prop
       
    # Prepare model
    input_shape = (None, 400, 400, 3)
    model = None
    
    if trial_no == 1:
        model = UNet(input_shape, bg_weight, road_weight)
    
    elif trial_no == 2 or trial_no == 3:
        model = CBAM_UNet(input_shape, bg_weight, road_weight, 'weighted_bce')
    
    elif trial_no == 4:
        model = CBAM_UNet(input_shape, bg_weight, road_weight, 'weighted_iou')
    
    # Path to trained weights
    # TODO: Change this filepath to directory where trained weights are at
    # TODO: In README instructions, ask them to put file at base directory
    inputs = tf.keras.Input(shape=input_shape[1:])
    model.call(inputs)
    model.built = True
    
    weights_filepath = os.path.join(os.getcwd(), 'results', f'trial_{trial_no}', f'trial{trial_no}_weights.h5')
    model.load_weights(weights_filepath)
    
    # Forward feed each batch into model
    img_no = 1
    for imgs_batch in imgs_ds:
        
        preds = model(imgs_batch)
        
        # Prepare model's output to binary semantic map
        preds_binary = model.get_binary_mask(preds)
        
        # Save predicted semantic maps into a file directory
        for pred in preds_binary:
            pred = pred.numpy()
            pred_arr = np.squeeze(pred)  # Remove singleton dimension
            # Convert to uint8 for Pillow
            pred_image = (pred_arr * 255).astype(np.uint8)
            # Convert to Pillow Image
            pred_image = Image.fromarray(pred_image, mode='L')
            # Save the image
            output_filename = 'submission_results/satImage_' + '%.3d' % img_no + '.png'
            pred_image.save(output_filename)
            img_no += 1
                    
    # Use semantic maps to generate submission csv file
    test_output_fns = tf.io.gfile.glob(f"{test_output_dir}/*.png")
    # Exclude directories
    test_output_fns = [fn for fn in test_output_fns if not tf.io.gfile.isdir(fn)]
    
    submission_filename = 'submission.csv'
    masks_to_submission(submission_filename, *test_output_fns)
    
generate_submissions(4)