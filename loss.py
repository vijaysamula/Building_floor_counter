
""" Here the loss function Quantile loss is defined"""

import torch.nn as nn 
import torch




def criterion(pred, target):
    """
    this is Quantile loss function
    input arguments are pred and target 
    hyper parameters are Quantiles q1,q2,q3
    
    returns loss 
    """
    ## Other Losses
    # Normal loss
    #loss = (((input[0] - target)/torch.exp(input[1]))**2+input[1]).mean()
    # Laplace loss
    #loss = (torch.abs((input[0] - target)/torch.exp(input[1]))+input[1]).mean()
    # t-distribution loss
    #nu = 3
    #loss = ((nu + 1)/2*torch.log(1+((input[0] - target)/torch.exp(input[1]))**2/nu)+input[1]).mean()
    # print(input, target)

    ## Quanile Loss
    q1 = 0.05
    q2 = 0.5
    q3 = 0.95
    

    ## Keras quantile loss, https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/    
    #e = y_p-y    
    #return tf.keras.backend.mean(tf.keras.backend.maximum(q*e, (q-1)*e))
    
    ## Quantile Loss
    ## for q1, q2, q3
    #print(input.shape)
    #print(input[:,0:1].shape)
    #print(target.shape)
    e1 = pred - target # !!! if input[:,0]  -> shape = (1000,)
    
    eq1 = torch.max(q1*e1, (q1-1)*e1)
    eq2 = torch.max(q2*e1, (q2-1)*e1)
    eq3 = torch.max(q3*e1, (q3-1)*e1)
    
    #eq1 = torch.max(0.05*e1, (0.05-1)*e1)
    #eq2 = torch.max(0.5*e2, (0.5-1)*e2)
    #eq3 = torch.max(0.95*e3, (0.95-1)*e3)

    loss = (eq1 + eq2 + eq3).mean()

    return loss

