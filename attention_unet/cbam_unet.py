'''This file contains the implementation code for the CBAM-UNet model.

This model references the following papers:
1. CBAM: https://arxiv.org/pdf/1807.06521.pdf
2. U-Net Architecture: https://arxiv.org/pdf/1505.04597.pdf

This model uses the U-Net Architecture as the backbone and incorporates Attention mechanism and skip connections to allow for better performance.

Side note: (Additional papers to be implemented in the future)
1. CBAM-UNet++ Architecture: https://ieeexplore.ieee.org/document/9622008
'''

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Conv2DTranspose, Concatenate, Input, Lambda
from keras import Model

from cbam import *
from layers import *

class CBAM_UNet(Model):
    
    
    def __init__(self, input_shape, num_classes = 2, **kwargs):
    
        """Initialises the CBAM-UNet model.
        """
        
        super(CBAM_UNet, self).__init__(**kwargs)
        
        # Instantiate basic parameters of the model
        self._input_shape = input_shape
        self.num_classes = num_classes
        
        # Instantiate CBAM modules
        self.cbam_depth1 = CBAM_Module(64)
        self.cbam_depth2 = CBAM_Module(128)
        self.cbam_depth3 = CBAM_Module(256)
        self.cbam_depth4 = CBAM_Module(512)
        
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
    
    
    def apply_att_and_crop(self, x, depth, new_size):

        '''Implementation code for applying attention and concatenating it to upsampled feature map

        Args:
            x: Input features of shape (height, width, channel) from downsampled portion
            depth: Depth in U-Net model, to decide which CBAM_Module to apply
            new_size: Desired size of cropped image, to match with the upsampled size from deeper layers

        Returns:
            cropped features from downsmpled portion with attention applied
        '''

        # Apply attention
        if depth == 4:
            x_att_applied = self.cbam_depth4(x)
        elif depth == 3:
            x_att_applied = self.cbam_depth3(x)
        elif depth == 2:
            x_att_applied = self.cbam_depth2(x)
        else:
            x_att_applied = self.cbam_depth1(x)
       
        x_att_shape = tf.shape(x_att_applied)
        
        # Calculate the crop sizes for height and width
        h_crop_size = (x_att_shape[1] - new_size[1]) // 2
        w_crop_size = (x_att_shape[2] - new_size[2]) // 2

        # Calculate the starting and ending indices for cropping along height
        h_start = h_crop_size
        h_end = h_start + new_size[1]

        # Calculate the starting and ending indices for cropping along width
        w_start = w_crop_size
        w_end = w_start + new_size[2]

        # Return the cropped image tensor
        return x_att_applied[:, h_start:h_end, w_start:w_end, :]

    
    def call(self, inputs, num_classes=2):

        '''Implementation code for CBAM UNet model.

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
        
        # x = self.conv3x3_encoder_block(x, 64)
        x = self.encoder_convblock_depth1(x)
        x_1 = x
        # shape: (None, 400, 400, 64)
        
        x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)
        # shape: (None, 200, 200, 64)
        
        # x = self.conv3x3_encoder_block(x, 128)
        x = self.encoder_convblock_depth2(x)
        x_2 = x
        # shape: (None, 200, 200, 128)
        x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)
        # shape: (None, 100, 100, 128)
        
        # x = self.conv3x3_encoder_block(x, 256)
        x = self.encoder_convblock_depth3(x)
        x_3 = x
        # shape: (None, 100, 100, 256)
        x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)
        # shape: (None, 50, 50, 256)

        # x = self.conv3x3_encoder_block(x, 512)
        x = self.encoder_convblock_depth4(x)
        x_4 = x
        # shape: (None, 50, 50, 512)
        x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)
        # shape: (None, 25, 25, 1024)

        # x = self.conv3x3_encoder_block(x, 1024)
        x = self.encoder_convblock_depth5(x)
        # shape: (None, 25, 25, 1024)

        # -----------------------------------
        # Upsampling part - Decoder portion

        # At each upsampling layer,
        # 1. Apply up convolution of kernel size 2x2 on x
        # 2. Apply attention on x_i and crop to desired size
        # 3. Concatenate with 1. and apply 3x3 convolutional block (with no max pooling to downsample)
        
        # No. of channels x1/2 but dimensions x2
        #x = Conv2DTranspose(filters = 512, kernel_size = 2, strides = 2, padding = 'same')(x)
        x = self.convtranspose_depth5(x)
        attention_x_4 = self.apply_att_and_crop(x_4, 4, tf.shape(x_4))       
        x = Concatenate(axis=-1)([attention_x_4, x])
        #x = self.conv3x3_decoder_block(x, 512)
        x = self.decoder_convblock_depth4(x)
        # shape: (None, 50, 50, 512)
        
        #x = Conv2DTranspose(filters = 256, kernel_size = 2, strides = 2, padding = 'same')(x)
        x = self.convtranspose_depth4(x)
        attention_x_3 = self.apply_att_and_crop(x_3, 3, tf.shape(x))
        x = Concatenate(axis=-1)([attention_x_3, x])
        #x = self.conv3x3_decoder_block(x, 256)
        x = self.decoder_convblock_depth3(x)
        # shape: (None, 100, 100, 256)
        
        #x = Conv2DTranspose(filters = 128, kernel_size = 2, strides = 2, padding = 'same')(x)
        x = self.convtranspose_depth3(x)
        attention_x_2 = self.apply_att_and_crop(x_2, 2, tf.shape(x))
        x = Concatenate(axis=-1)([attention_x_2, x])
        #x = self.conv3x3_decoder_block(x, 128)
        x = self.decoder_convblock_depth2(x)
        # shape: (None, 200, 200, 128)
        
        # x = Conv2DTranspose(filters = 64, kernel_size = 2, strides = 2, padding = 'same')(x)
        x = self.convtranspose_depth2(x)
        attention_x_1 = self.apply_att_and_crop(x_1, 1, tf.shape(x))
        x = Concatenate(axis=-1)([attention_x_1, x])
        # shape: (None, 400, 400, 128)
        
        # -----------------------------------
        # Final portion: Further convolve and finally a 1x1 convolution to get segmentation map
        #x = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same')(x)
        # shape: (None, 400, 400, 64)
        #x = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same')(x)
        # shape: (None, 400, 400, 64)
        #x = Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid')(x)
        
        x = self.output_convblock(x)
        # shape: (None, 400, 400, 1)
      
        #threshold = 0.5
        #x = Lambda(lambda val: tf.where(val >= threshold, 1.0, 0.0))(x)
        
        return x
        
    
    def compute_loss(self, y_true, y_pred):
        
        '''Implementation code to calculate IoU loss for the CBAM model. 
        
        Args:
            y_true: tensor of shape (batch_size, 400, 400, 1), containing pixel values of groundtruth image
            y_pred: tensor of shape (batch_size, 400, 400, 1), containing pixel values of prediction from model

        Returns:
            iou_loss: float
        '''
        
        # intersection = tf.reduce_sum(y_true * y_pred)
        # union = tf.reduce_sum(y_true + y_pred) - intersection
        # return 1. - (intersection + 1.)/ (union + 1.)
    
        scores = []
        
        for i in range(y_true.shape[0]):
            # Flattten segmentation maps into 1D array
            y_true_sample = tf.reshape(y_true[i], [-1])
            y_pred_sample = tf.reshape(y_pred[i], [-1])

            # Get number of pixels that are predicted correctly (1*1 = 1 otherwise 0)
            intersection = tf.reduce_sum(tf.cast(y_true_sample, tf.float32) * tf.cast(y_pred_sample, tf.float32))
            
            # Compute iou score which = intersection/union
            score = (intersection + 1.) / (tf.reduce_sum(tf.cast(y_true_sample, tf.float32)) + tf.reduce_sum(tf.cast(y_pred_sample, tf.float32)) - intersection + 1.)

            scores.append(1. - score)
            
        # Return iou loss where IoU = 1 for perfect score hence, an iou loss of 0
        scores = tf.convert_to_tensor(scores, dtype=tf.float32)
        return tf.reduce_mean(scores)

    
    
    def compute_metrics(self, y_true, y_pred):
        
        '''Implementation code to calculate desired metrics of CBAM model. 
        
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
        
        return (ttal_acc/y_true.shape[0]), (iou/y_true.shape[0]), (dice/y_true.shape[0])