'''This file contains the implementation code for the different loss functions to be used to train the model.
'''

import tensorflow as tf
import tensorflow.keras.backend as K

from metrics import *

class Loss:
    
    def __init__(self, background_weight, road_weight):
        self.background_weight = background_weight
        self.road_weight = road_weight

        
class Weighted_iou(Loss):
    
    def __init__(self, background_weight, road_weight):
        super(Weighted_iou, self).__init__(background_weight, road_weight)
        self.metrics = IOU_metric(background_weight, road_weight)
    
    def compute_loss(self, y_true, y_pred):
        
        '''Implementation code to calculate weighted iou loss. 
        
        Args:
            y_true: tensor of shape (batch_size, 400, 400, 2), containing pixel values of groundtruth image. Each channel corresponds to one segmented class.
            y_pred: tensor of shape (batch_size, 400, 400, 2), containing pixel values of prediction from model. Each channel corresponds to one segmented class.

        Returns:
            loss
        '''    
            
        # add a constant to ensure that loss is positive when applying class weights
        constant = 10.
        return (1. - self.metrics.compute_iou(y_true, y_pred, self.background_weight, self.road_weight)) + 10.
    

class Weighted_bce(Loss):
    
    def __init__(self, background_weight, road_weight):
        super(Weighted_bce, self).__init__(background_weight, road_weight)
        self.metrics = BCE_metric(background_weight, road_weight)
        
    def compute_loss(self, y_true, y_pred):
        
        '''Implementation code to calculate weighted binary cross entropy loss.
        
        Args:
            y_true: tensor of shape (batch_size, 400, 400, 1), containing pixel values of groundtruth image
            y_pred: tensor of shape (batch_size, 400, 400, 1), containing pixel values of prediction from model

        Returns:
            loss
        '''    
            
        class_weights = [self.background_weight, self.road_weight] 
        num_samples = y_true.shape[0]
        loss = 0.
        
        for i in range(num_samples):
            # binary cross entropy loss
            predictions = tf.reshape(y_pred[i], [-1])
            targets = tf.reshape(y_true[i], [-1])

            # Apply binary cross entropy loss
            bce = tf.keras.losses.BinaryCrossentropy()(targets, predictions)
            weighted_bce = class_weights*bce
            loss += tf.reduce_mean(weighted_bce)
        
        return loss/num_samples

