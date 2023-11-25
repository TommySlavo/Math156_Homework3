### Losses
import numpy as np
class MSE:
    def loss(self, predictions, y):

        l = np.mean((y - predictions)**2)
        return l
    def loss_gradient(self, predictions, y):
        grad = -2*(y - predictions)/y.shape[1]
        return grad.T
    
class Cross_Entropy:
    
    def loss(self, predictions, y):

        l = -np.sum(y*np.log(predictions + 1*np.exp(-6)))
        #compute here the cross entropy loss l, you may want to add 1e-6 to the argument of the logarithm for numercal stability
        return l
    def loss_gradient(self, predictions, y):
        grad = -(y / (predictions))
        #compute here the cross entropy loss l as computed before, the result, grad, should be of dimension n x m, where m is the number of points in the batch
        return grad.T
