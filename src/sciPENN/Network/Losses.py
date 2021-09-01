from torch import tensor, abs as torch_abs, logical_not, log, clamp
from torch.nn import Softplus

class no_loss(object):
    def __init__(self, device):
        self.device = device
    
    def __call__(self, outputs, target):
        return tensor(0., device = self.device)

class mse_quantile(object):
    def __init__(self, device, quantiles):
        self.mse = mse_loss(reduce = False)
        
        if quantiles is not None:
            assert type(quantiles) == list
            assert min(quantiles) > 0.
            assert max(quantiles) < 1.
            
            self.qloss = quantile_loss(device, quantiles)
            self.loss_call = self.multi_call
        else:
            self.loss_call = self.single_call
    
    def __call__(self, yhat, y, bools):
        return self.loss_call(yhat, y, bools)
    
    def multi_call(self, yhat, y, bools):
        mse = self.mse(yhat[0], y)
        qloss = self.qloss(yhat[1], y)
        
        loss = mse + qloss
        loss = loss * bools
        
        return loss.mean()
    
    def single_call(self, yhat, y, bools):
        mse = self.mse(yhat, y)
        mse = mse * bools
        
        return mse.mean()
    
class quantile_loss(object):
    def __init__(self, device, quantiles):        
        self.q = tensor(quantiles, device = device)
        
    def __call__(self, pred, truth):
        bias = pred - truth[:, :, None]
        
        I_over = bias.detach() > 0.
        q_weight = I_over * (1 - self.q) + logical_not(I_over) * self.q
        
        q_loss = torch_abs(bias) * q_weight
        
        return q_loss.mean(axis = 2)
    
class mse_loss(object):
    def __init__(self, reduce = True):
        self.reduce = reduce
        
    def __call__(self, yhat, y):
        SEs = (yhat - y)**2
        
        if self.reduce:
            return SEs.mean()
        
        return SEs
        
def cross_entropy(outputs, target):
    indices = range(len(outputs))
    return -(log(outputs[indices, target])).mean()