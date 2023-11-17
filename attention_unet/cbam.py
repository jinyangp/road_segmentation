'''This file contains the implementation code for the Convolutional Block Attention Module (CBAM).


The code references the following papers:
1. CBAM: https://arxiv.org/pdf/1807.06521.pdf
'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense,\
                                    Add, Concatenate, Multiply


class CBAM_Module(Layer):


    def __init__(self, reduction_rate = 0.3):
        
        """Initialises the CBAM module.
        """
            
        super(CBAM_Module, self).__init__(**kwargs)
        self.reduction_rate = reduction_rate
        self.mlp_hidden_layer = Dense(units = num_channels/reduction_ratio,
                                   activation = 'relu')
        self.mlp_output_layer = Dense(units = num_channels)


    def compute_channel_att(self, x):

        """Implements the channel attention component of CBAM.

        Args:
            x: Input features F of shape (height, width,channel)
            reduction_ratio: Ratio to reduce no. of parameters in shaped MLP. A hyperparameter to be tuned later

        Returns:
            channel_att_map: Channel attention map of shape (channel, 1, 1)

        Notes:
        1. In literature, feature map is passed in as (channel, height, width) but in Tensorflow implementaiton, feature maps are handled as (height, width, channel) so have to perform a transpose operation on input.
        2. Channel attention map is multiplied element-wise to each pixel in each of the channel
        """

        # Make shape consistent with literature
        # TF handles array as (height,width, channel) but literature states (channel, height, width)
        x = tf.transpose(x, perm=[2, 0, 1])
        # shape: (channel, height, width)

        # Apply max and average pooling
        max_pool_x = GlobalMaxPooling2D()(x)
        avg_pool_x = GlobalAveragePooling2D()(x)
        #  shape: (channel, )

        # Shared MLP
        num_channels = tf.shape(avg_pool_x).numpy()[0]

        # Pass pooled tensors into shared MLP
        max_pool_x = self.mlp_hidden_layer()(max_pool_x)
        # shape: (channel/reduction_ratio, )
        max_pool_x = self.mlp_output_layer()(max_pool_x)
        # shape: (channel, )

        avg_pool_x = self.mlp_hidden_layer()(avg_pool_x)
        # shape: (channel/reduction_ratio, )
        avg_pool_x = self.mlp_output_layer()(avg_pool_x)
        # shape: (channel, )

        # Perform element wise summation
        x = Add()([max_pool_x, avg_pool_x])
        # shape: (channel, )

        # Pass through sigmoid activation
        x = tf.nn.sigmoid(x)
        # shape: (channel, )

        return x


    def compute_spatial_att(self, x):

        """Implements the spatial attention component of CBAM.

        Args:
            x: Input features of shape (channel, height, width) after applying channel attention

        Returns:
            spatial_att_map: Spatial attention map of shape (height, width)
        """

        # Apply max pooling across the channels
        # Here, 4 dimensions as accounting for the batch size as well
        max_pool_x = tf.nn.max_pool(x, ksize = (1,1,1,3), strides = (1,1,1,3), padding = 'VALID')
        # shape: (1, height, width)
        avg_pool_x = tf.nn.avg_pool(x, ksize = (1,1,1,3), strides = (1,1,1,3), padding = 'VALID')
        # shape: (1, height, width)

        # Concatenate the inputs channel wise
        x = Concatenate()([max_pool_x, avg_pool_x])
        # shape: (2, height, width)

        # Transpose the matrix to get (height, width, 2)
        x = tf.transpose(x, perm = [1,2,0])

        # Perform 7x7 convolution layer
        x = Conv2D(filters = 1, kernel_size = 7, strides = 1, padding = 'same')(x)
        # shape: (height, width, 1)

        x = tf.nn.sigmoid(x)
        # shape: (height, width, 1)

        return x


        def call(self, x):

            """Passes input features through the CBAM module

            Args:
                x: Input features of shape (height,width,channel)

            Returns:
                x: Input features with attention applied of shape (height,width,channel)
            """

            # Get dimensions of the input
            num_channels = tf.shape(avg_pool_x).numpy()[2]

            # Compute channel attention
            channel_att = self.compute_channel_att(x)

            # Apply channel attention
            # Idea: Multiply each value in the vector channel_att with the corresponding channel in the image
            # Broadcast channel_att in the height and width dimensions
            channel_attn_reshaped = tf.reshape(channel_attn, (1, 1, -1))
            # shape: (1,1, channel)

            x = Multiply()([x, channel_attn_reshaped])
            # shape: (height, width, channel)

            # Compute spatial attention
            spatial_att = self.compute_spatial_att()

            # Apply spatial attention
            # Broadcast spatial_att in the channel dimension
            spatial_att = tf.expand_dims(spatial_att, axis=-1)
            # shape: (height, width, 1)
            spatial_att = tf.tile(spatial_att, multiples=[1, 1, num_channels])
            # shape: (height, width, channel)

            # Return input features with attention applied
            x = Multiply()([x, spatial_att])

            return x