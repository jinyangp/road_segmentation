'''This file contains the implementation code for computing evaluation metrics for the model (including iou, dice and accuracy).
'''

import tensorflow as tf
import tensorflow.keras.backend as K

class Metric:
    
    '''Base class for functions used to calculate evaluation metrics'''    
    def compute_intersection(self, y_true, y_pred):
        
        return tf.cast(tf.reduce_sum(y_true * y_pred), dtype = tf.float32)

    
    def compute_union(self, y_true, y_pred):

        intersection = self.compute_intersection(y_true, y_pred)
        return tf.cast(tf.reduce_sum(y_true), dtype = tf.float32) + tf.cast(tf.reduce_sum(y_pred), dtype = tf.float32) - intersection


    def iou(self, y_true, y_pred):

        epsilon = 1e-7

        intersection = self.compute_intersection(y_true, y_pred)
        union = self.compute_union(y_true, y_pred)
        
        return (intersection + epsilon)/ (union + epsilon)


    def dice(self, y_true, y_pred):

        epsilon = 1e-7

        intersection = self.compute_intersection(y_true, y_pred)
        numerator = 2.*intersection
        denominator = self.compute_union(y_true, y_pred) + intersection
        
        return (numerator + epsilon)/ (denominator + epsilon)
    

class BCE_metric(Metric):
    
    '''Class for functions used to calculate evaluation metric for model using bce loss'''
    def __init__(self, background_weight, road_weight):
        super(BCE_metric, self).__init__()
        self.background_weight = background_weight
        self.road_weight = road_weight

        
    def flatten_arrs(self, y_true, y_pred):
        
        # Flattten segmentation maps into 1D array
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        return y_true, y_pred

    
    def compute_iou(self, y_true, y_pred):
            
        y_true, y_pred = self.flatten_arrs(y_true, y_pred)
        
        return self.iou(y_true, y_pred)
    
    
    def compute_dice(self, y_true, y_pred):
        
        y_true, y_pred = self.flatten_arrs(y_true, y_pred)
        
        return self.dice(y_true, y_pred)
        
    
    def compute_acc(self, y_true, y_pred):
    
        y_true, y_pred = self.flatten_arrs(y_true, y_pred)
        
        # Threshold pixels to get predicted class
        y_true = tf.where(tf.greater(y_true, 0.5), 1., 0.)
        y_pred = tf.where(tf.greater(y_pred, 0.5), 1., 0.)
        
        correct_pixels = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))     
        ttal_pixel = y_true.shape[0]
        accuracy = correct_pixels/ ttal_pixel
        
        return accuracy

        
