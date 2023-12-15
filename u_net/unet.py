'''This file contains the implementation code for the UNet model.

This model references the following papers:
1. U-Net: Convolutional Networks for Biomedical Image Segmentation, https://arxiv.org/pdf/1505.04597.pdf

This model serves as a baseline to the CBAM-UNet model implemented.
'''

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Conv2DTranspose, Concatenate, Input, Lambda
from keras import Model

from layers import *
from losses import *

class UNet(Model):
    
    
    def __init__(self, input_shape, background_weight, road_weight, num_classes = 2, **kwargs):
    
        """Initialises the UNet model.
        """
        
        super(UNet, self).__init__(**kwargs)
        
        # Instantiate basic parameters of the model
        self._input_shape = input_shape
        self.num_classes = num_classes
        self.background_weight = background_weight
        self.road_weight = road_weight

        # Instantiate loss function
        self.loss_fnc = Weighted_bce(self.background_weight, self.road_weight)
        
        # Instantiate custom layers
        self.encoder_convblock_depth1 = Encoder_ConvBlock(1, 64, 3, 1)
        self.encoder_convblock_depth2 = Encoder_ConvBlock(2, 128, 3, 1)
        self.encoder_convblock_depth3 = Encoder_ConvBlock(3, 256, 3, 1)
        self.encoder_convblock_depth4 = Encoder_ConvBlock(4, 512, 3, 1)
        self.encoder_convblock_depth5 = Encoder_ConvBlock(5, 1024, 3, 1)
        
        self.convtranspose_depth5 = Conv2DTranspose(filters = 512, kernel_size = 2, strides = 2, padding = 'same')
        self.convtranspose_depth4 = Conv2DTranspose(filters = 256, kernel_size = 2, strides = 2, padding = 'same')
        self.convtranspose_depth3 = Conv2DTranspose(filters = 128, kernel_size = 2, strides = 2, padding = 'same')
        self.convtranspose_depth2 = Conv2DTranspose(filters = 64, kernel_size = 2, strides = 2, padding = 'same')
        
        self.decoder_convblock_depth4 = Decoder_ConvBlock(4, 512, 3, 1)
        self.decoder_convblock_depth3 = Decoder_ConvBlock(3, 256, 3, 1)
        self.decoder_convblock_depth2 = Decoder_ConvBlock(2, 128, 3, 1)
        
        self.output_convblock = Output_ConvBlock(64, 3, 1)
    
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
        
        '''Implementation code to calculate binary cross entropy loss for the CBAM U-Net model. 
        
        Args:
            output shape can be of shape (batch_size, 400, 400, 1) or (batch_size, 400, 400, 2) depending on the loss function used
            y_true: tensors containing pixel values of groundtruth image
            y_pred: tensor containing pixel values of predictions

        Returns:
            loss
        '''        
            
        return self.loss_fnc.compute_loss(y_true, y_pred)


    def compute_metrics(self, y_true, y_pred):
        
        '''Implementation code to calculate desired metrics of CBAM model. 
        
        The following losses are considered:
        - Accuracy
        - IoU
        - Dice/F1 (In binary semantic segmentation, Dice == IoU)
        
        Args:
            output shape can be of shape (batch_size, 400, 400, 1) or (batch_size, 400, 400, 2) depending on the loss function used
            y_true: tensors containing pixel values of groundtruth image
            y_pred: tensor containing pixel values of predictions
        
        Returns:
            accuracy, iou, dice
        '''
        
        iou = self.loss_fnc.metrics.compute_iou(y_true, y_pred)
        dice = self.loss_fnc.metrics.compute_dice(y_true, y_pred)
        acc = self.loss_fnc.metrics.compute_acc(y_true, y_pred)
        
        return acc, iou, dice
    
    
    def get_binary_mask(self, y_pred):
        
        '''Converts the output to a binary semantic mask.
        
        Args:
            y_pred: tensor containing predictions, of shape (num_samples, 400, 400, 1)
            
        Returns:
            y_pred_binary: tensors containing 0 or 1, representing class 0 or 1, of shape (num_samples, 400, 400, 1)
        '''
        
        # TODO: Update documentations to (batch_size, height, width, channel) when describing shape to generalise for input shape
        y_pred_binary = tf.where(y_pred > 0.5, 1.0, 0.0)
            
        return y_pred_binary