'''This file contains the implementation code for custom layers for the CBAM-UNet model.
'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Conv2DTranspose, Layer, Lambda


class Encoder_ConvBlock(Layer):
    
    '''Convolutional block for the encoder (downsampling portion) of the model. 
    '''
    
    def __init__(self, depth_no, num_filters, kernel_size, strides, **kwargs):
        
        super(Encoder_ConvBlock, self).__init__(**kwargs)
        
        # Encoder block is a Layer itself with nested layers
        # First, give the encoder block a name of its own
        self.depth_no = depth_no
        self._name = f'encoder_depth{self.depth_no}'
        
        self.conv_layer1 = Conv2D(filters = num_filters, kernel_size = kernel_size, strides = strides, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')
        
        self.conv_layer2 = Conv2D(filters = num_filters, kernel_size = kernel_size, strides = strides, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')
    
    
    def call(self, x):
            
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)            
        return x

    
class Decoder_ConvBlock(Layer):
    
    '''Convolutional block for the decoder (upsampling portion) of the model.
    '''
    
    def __init__(self, depth_no, num_filters, kernel_size, strides, **kwargs):
        
        super(Decoder_ConvBlock, self).__init__(**kwargs)
        
        # Decoder block is a Layer itself with nested layers
        # First, give the encoder block a name of its own
        self.depth_no = depth_no
        self._name = f'decoder_depth{self.depth_no}'
        
        self.conv_layer1 = Conv2D(filters = num_filters, kernel_size = kernel_size, strides = strides, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')
        
        self.conv_layer2 = Conv2D(filters = num_filters, kernel_size = kernel_size, strides = strides, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')
      
        
    def call(self, x):
        
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        
        return x

    
class Output_ConvBlock(Layer):
    
    '''Convolutional block after the decoder portion of the model (final convolution layers).
    '''
    
    def __init__(self, num_filters, kernel_size, strides, output_num_filters, **kwargs):
        
        super(Output_ConvBlock, self).__init__(**kwargs)
        
        self._name = 'output_block'
        
        self.conv_layer1 = Conv2D(filters = num_filters, kernel_size = kernel_size, strides = strides, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')
        
        self.conv_layer2 = Conv2D(filters = num_filters, kernel_size = kernel_size, strides = strides, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')
        
        self.conv1d_layer = Conv2D(filters = output_num_filters, kernel_size = 1, activation = 'sigmoid')
        
    
    def call(self, x):
        
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv1d_layer(x)
        
        return x