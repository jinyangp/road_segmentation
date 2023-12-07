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


class IOU_metric(Metric):
    
    '''Class for functions used to calculate evaluation metric for model using iou loss'''
    def __init__(self, background_weight, road_weight):
        super(IOU_metric, self).__init__()
        self.background_weight = background_weight
        self.road_weight = road_weight
        
        
    def flatten_arrs(self, y_true, y_pred):

        y_true_background = K.flatten(y_true[...,0])
        y_true_road = K.flatten(y_true[...,1:])

        y_pred_background = K.flatten(y_pred[...,0])
        y_pred_road = K.flatten(y_pred[...,1:])

        return y_true_background, y_true_road, y_pred_background, y_pred_road


    def compute_iou(self, y_true, y_pred, background_weight = 0.5, road_weight = 0.5):

        y_true_background, y_true_road, y_pred_background, y_pred_road = self.flatten_arrs(y_true, y_pred)

        background_iou = self.iou(y_true_background, y_pred_background)
        road_iou = self.iou(y_true_road, y_pred_road)

        return (background_weight*background_iou) + (road_weight*road_iou)
    

    def compute_dice(self, y_true, y_pred, background_weight = 0.5, road_weight = 0.5):

        y_true_background, y_true_road, y_pred_background, y_pred_road = self.flatten_arrs(y_true, y_pred)

        background_dice = self.dice(y_true_background, y_pred_background)
        road_dice = self.dice(y_true_road, y_pred_road)

        return (background_weight*background_dice) + (road_weight*road_dice)
    

    def compute_acc(self, y_true, y_pred):

        total_acc = 0.0

        # Compute accuracy
        for i in range(y_true.shape[0]):
            y_true_sample = y_true[i]
            y_pred_sample = y_pred[i]

            y_pred_sample = tf.argmax(y_pred_sample, axis=-1)
            y_true_sample = tf.argmax(y_true_sample, axis=-1)

            correct_pixel = tf.reduce_sum(tf.cast(tf.equal(y_true_sample, y_pred_sample), tf.float32))
            height, width = y_true.shape[1], y_true.shape[2]
            accuracy = correct_pixel / (height * width)
            total_acc += accuracy

        return total_acc / y_true.shape[0]
            
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

        