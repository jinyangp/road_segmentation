'''This file contains the implementation code for the UNet model.

This model references the following papers:
1. U-Net Architecture: https://arxiv.org/pdf/1505.04597.pdf

Side note: (Additional papers to be implemented in the future)
1. CBAM-UNet++ Architecture: https://ieeexplore.ieee.org/document/9622008
'''

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Conv2DTranspose, Concatenate, Input, Lambda
from keras import Model

from layers import *

class UNet(Model):
    
    
    def __init__(self, input_shape, num_classes = 2, **kwargs):
    
        """Initialises the CBAM-UNet model.
        """
        
        super(CBAM_UNet, self).__init__(**kwargs)
        
        # Instantiate basic parameters of the model
        self._input_shape = input_shape
        self.num_classes = num_classes
        
        # Instantiate custom layers
        self.encoder_convblock_depth1 = Encoder_ConvBlock(64, 3, 1, True)
        self.encoder_convblock_depth2 = Encoder_ConvBlock(128, 3, 1, True)
        self.encoder_convblock_depth3 = Encoder_ConvBlock(256, 3, 1, True)
        self.encoder_convblock_depth4 = Encoder_ConvBlock(512, 3, 1, True)
        self.encoder_convblock_depth5 = Encoder_ConvBlock(1024, 3, 1, True)
        
        self.convtranspose_depth5 = Conv2DTranspose(filters = 512, kernel_size = 2, strides = 2, padding = 'same')
        self.convtranspose_depth4 = Conv2DTranspose(filters = 256, kernel_size = 2, strides = 2, padding = 'same')
        self.convtranspose_depth3 = Conv2DTranspose(filters = 128, kernel_size = 2, strides = 2, padding = 'same')
        self.convtranspose_depth2 = Conv2DTranspose(filters = 64, kernel_size = 2, strides = 2, padding = 'same')
        
        self.decoder_convblock_depth4 = Decoder_ConvBlock(512, 3, 1, True)
        self.decoder_convblock_depth3 = Decoder_ConvBlock(256, 3, 1, True)
        self.decoder_convblock_depth2 = Decoder_ConvBlock(128, 3, 1, True)
        
        self.output_convblock = Output_ConvBlock(64, 3, 1, True)
        
    
    # Override inherited summary() function
    def summary(self):
        model_inputs = Input(shape = self._input_shape[1:])
        model_outputs = self.call(model_inputs)
        model = Model(inputs = model_inputs, outputs = model_outputs)
        return model.summary()
    
    
    def call(self, inputs, num_classes=2):

        '''Implementation code for UNet model.

        In each convolutional block, 2 2D Convolutional layers with a kernel size of 3x3 and the number of filters per layer
        given as an argument to the function.

        Args:
            x: Input features of shape (height, width, channel)
            num_filters: int, number of filters used in convolutional layer

        Returns:
            output of the convolutional block
        '''

        # Initial shape: (400, 400, 3)

        # -----------------------------------
        # Downsampling part - Encoder portion
        # Store x at each iteration of downsampling in another variable for skip connection later on
        x = inputs
        x_0 = x
        # shape: (None, 400, 400, 3)
        
        x = self.encoder_convblock_depth1(x)
        x_1 = x
        # shape: (None, 400, 400, 64)
        
        x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)
        # shape: (None, 200, 200, 64)
        
        x = self.encoder_convblock_depth2(x)
        x_2 = x
        # shape: (None, 200, 200, 128)
        x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)
        # shape: (None, 100, 100, 128)
        
        x = self.encoder_convblock_depth3(x)
        x_3 = x
        # shape: (None, 100, 100, 256)
        x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)
        # shape: (None, 50, 50, 256)

        x = self.encoder_convblock_depth4(x)
        x_4 = x
        # shape: (None, 50, 50, 512)
        x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)
        # shape: (None, 25, 25, 1024)

        x = self.encoder_convblock_depth5(x)
        # shape: (None, 25, 25, 1024)

        # -----------------------------------
        # Upsampling part - Decoder portion

        # No. of channels x1/2 but dimensions x2
        x = self.convtranspose_depth5(x)
        x = Concatenate(axis=-1)([x_4, x])
        x = self.decoder_convblock_depth4(x)
        # shape: (None, 50, 50, 512)
        
        x = self.convtranspose_depth4(x)
        x = Concatenate(axis = -1)([x_3, x])
        x = self.decoder_convblock_depth3(x)
        # shape: (None, 100, 100, 256)
        
        x = self.convtranspose_depth3(x)
        x = Concatenate(axis = -1)([x_2, x])
        x = self.decoder_convblock_depth2(x)
        # shape: (None, 200, 200, 128)
        
        x = self.convtranspose_depth2(x)
        x = Concatenate(axis=-1)([x_1, x])
        # shape: (None, 400, 400, 128)
        
        # -----------------------------------
        # Final portion: Further convolve and finally a 1x1 convolution to get segmentation map

        x = self.output_convblock(x)
        # shape: (None, 400, 400, 1)
      
        return x
        
    
    def compute_loss(self, y_true, y_pred):
        
        '''Implementation code to calculate Ioss for model. 
        
        Args:
            y_true: tensor of shape (batch_size, 400, 400, 1), containing pixel values of groundtruth image
            y_pred: tensor of shape (batch_size, 400, 400, 1), containing pixel values of prediction from model

        Returns:
            loss
        '''
        
        # binary cross entropy loss
        predictions = tf.reshape(y_pred, [-1])
        targets = tf.reshape(y_true, [-1])

        # Apply binary cross entropy loss
        # TODO: Determine optimal class weights
        # TODO: BCE should be averaged over each sample in the batch, rather than done as a whole
        # TODO: Investigate more on accuracy and Dice/IoU
        bce = tf.keras.losses.BinaryCrossentropy()(targets, predictions)
        
        class_weights = [0.7, 0.3]
        weighted_bce = class_weights*bce
        return tf.reduce_mean(weighted_bce)
        
    
    def compute_metrics(self, y_true, y_pred):
        
        '''Implementation code to calculate desired metrics of model. 
        
        The following losses are considered:
        - Accuracy
        - IoU
        - Dice/F1 (In binary semantic segmentation, Dice == IoU)
        
        Args:
            y_true: tensor of shape (batch_size, 400, 400, 1), containing pixel values of groundtruth image
            y_pred: tensor of shape (batch_size, 400, 400, 1), containing pixel values of prediction from model

        Returns:
            accuracy, iou, dice
        '''
        
        ttal_acc, ttal_iou, ttal_dice = 0., 0., 0.
        
        for i in range(y_true.shape[0]):
            
            # Flattten segmentation maps into 1D array
            y_true_sample = tf.reshape(y_true[i], [-1])
            y_pred_sample = tf.reshape(y_pred[i], [-1])

            # Compute accuracy
            intersection = tf.reduce_sum(tf.cast(y_true_sample, tf.float32) * tf.cast(y_pred_sample, tf.float32))
            height, width = y_true.shape[1], y_true.shape[2]
            accuracy = intersection/ (height*width)
            ttal_acc += accuracy
            
            # Compute IoU
            iou = (intersection + 1.) / (tf.reduce_sum(tf.cast(y_true_sample, tf.float32)) + 
            tf.reduce_sum(tf.cast(y_pred_sample, tf.float32)) - intersection + 1.)
            ttal_iou += iou
            
            # Compute dice
            dice = (2*intersection)/ ((tf.reduce_sum(tf.cast(y_true_sample, tf.float32))) + (tf.reduce_sum(tf.cast(y_pred_sample, tf.float32))))
            ttal_dice += dice
        
        return (ttal_acc/y_true.shape[0]), (ttal_iou/y_true.shape[0]), (ttal_dice/y_true.shape[0])
