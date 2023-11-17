'''This file contains the implementation code for the CBAM-UNet model.

This model references the following papers:
1. CBAM: https://arxiv.org/pdf/1807.06521.pdf
2. U-Net Architecture: https://arxiv.org/pdf/1505.04597.pdf

This model uses the U-Net Architecture as the backbone and incorporates Attention mechanism and skip connections to allow for better performance.

Side note: (Additional papers to be implemented in the future)
1. CBAM-UNet++ Architecture: https://ieeexplore.ieee.org/document/9622008
'''


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Conv2DTranspose, Concatenate
from keras import Model

from cbam import *

class CBAM_UNet(Model):
    
    
    def __init__(self, num_classes = 2):
    
        """Initialises the CBAM-UNet model.
        """
        super(CBAM_UNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.cbam_depth1 = CBAM_Module()
        self.cbam_depth2 = CBAM_Module()
        self.cbam_depth3 = CBAM_Module()
        self.cbam_depth4 = CBAM_Module()
     
    
    def conv3x3_encoder_block(self, x, num_filters):

        '''Implementation code a convolutional 3x3 block.

        In each convolutional block, 2 2D Convolutional layers with a kernel size of 3x3 and the number of filters per layer given as an argument to the function.

        Args:
            x: Input features of shape (height, width, channel)
            num_filters: int, number of filters used in convolutional layer

        Returns:
            output of the convolutional block
        '''

        for i in range(2):
            x = Conv2D(filters = num_filters, kernel_size = 3, strides = 1, padding = 'valid')(x)
            x = ReLU()(x)
        x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x)

        return x
    
    
    def conv3x3_decoder_block(self, x, num_filters):
        
        '''Implementation code a convolutional 3x3 block.

        In each convolutional block, 2 2D Convolutional layers with a kernel size of 3x3 and the number of filters per layer given as an argument to the function.

        Args:
            x: Input features of shape (height, width, channel)
            num_filters: int, number of filters used in convolutional layer

        Returns:
            output of the convolutional block
        '''
        for i in range(2):
            x = Conv2D(filters = num_filters, kernel_size = 3, strides = 1, padding = 'same')(x)
            x = ReLU()(x)
        
        return x
    
    
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
            x_att_applied = self.cbam_depth4()(x) 
        elif depth == 3:
            x_att_applied = self.cbam_depth3()(x)
        elif depth == 2:
            x_att_applied = self.cbam_depth2()(x)
        else:
            x_att_applied = self.cbam_depth1()(x)
        
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
        x = self.conv3x3_encoder_block(x, 64)
        # shape: ()

        x_1 = x
        x = self.conv3x3_encoder_block(x, 128)
        # shape: ()

        x_2 = x
        x = self.conv3x3_encoder_block(x, 256)
        # shape: ()

        x_3 = x
        x = self.conv3x3_encoder_block(x, 512)
        # shape: ()

        x_4 = x
        x = self.conv3x3_encoder_block(x, 1024)

        # -----------------------------------
        # Upsampling part - Decoder portion

        # At each upsampling layer,
        # 1. Apply up convolution of kernel size 2x2 on x
        # 2. Apply attention on x_i and crop to desired size
        # 3. Concatenate with 1. and apply 3x3 convolutional block (with no max pooling to downsample)
        
        # No. of channels x1/2 but dimensions x2
        x = Conv2DTranspose(filters = 512, kernel_size = 2, strides = 1)(x)
        attention_x_4 = self.apply_att_and_crop(x_4, 4, tf.shape(x))
        x = Concatenate(axis=-1)([attention_x_4, x])
        x = self.conv3x3_decoder_block(x, 512)
        # shape: ()
        
        x = Conv2DTranspose(filters = 256, kernel_size = 2, strides = 1)(x)
        attention_x_3 = self.apply_att_and_crop(x_3, 3, tf.shape(x))
        x = Concatenate(axis=-1)([attention_x_3, x])
        x = self.conv3x3_decoder_block(x, 256)
        # shape: ()
        
        x = Conv2DTranspose(filters = 128, kernel_size = 2, strides = 1)(x)
        attention_x_2 = self.apply_att_and_crop(x_2, 2, tf.shape(x))
        x = Concatenate(axis=-1)([attention_x_2, x])
        x = self.conv3x3_decoder_block(x, 128)
        
        x = Conv2DTranspose(filters = 64, kernel_size = 2, strides = 1)(x)
        attention_x_1 = self.apply_att_and_crop(x_1, 1, tf.shape(x))
        x = Concatenate(axis=-1)([attention_x_1, x])
        
        # -----------------------------------
        # Final portion: Further convolve and finally a 1x1 convolution to get segmentation map
        x = Conv2D(filters = 64, kernel_size = 3, stride = 1, padding = 'same')(x)
        x = Conv2D(filters = 64, kernel_size = 3, stride = 1, padding = 'same')(x)
        x = Conv2D(filters = self.num_classes, kernel_size = 1)(x)
            
        return x
        
        
        
        
        
        
    

