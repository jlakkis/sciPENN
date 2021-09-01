from torch import cat
from torch.nn import Module, Linear, BatchNorm1d, PReLU, Dropout

class Input_Block(Module):
    def __init__(self, in_units, out_units, dropout_inrate, dropout_outrate):
        super(Input_Block, self).__init__()
        
        self.bnorm_in = BatchNorm1d(in_units)
        self.dropout_in = Dropout(dropout_inrate)

        self.dense = Linear(in_units, out_units)
        self.bnorm_out = BatchNorm1d(out_units)
        self.act = PReLU()
        self.dropout_out = Dropout(dropout_outrate)
            
    def forward(self, x_new):
        x_new = self.bnorm_in(x_new)
        x_new = self.dropout_in(x_new)
        
        x = self.dense(x_new)
        x = self.bnorm_out(x)
        x = self.act(x)
        x = self.dropout_out(x)
        
        return x
    
class FF_Block(Module):
    def __init__(self, hidden_units, dropout_rate):
        super(FF_Block, self).__init__()
        
        self.dense = Linear(hidden_units, hidden_units)
        self.bnorm = BatchNorm1d(hidden_units)
        self.act = PReLU()
        self.Dropout = Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.dense(x)
        x = self.bnorm(x)
        x = self.act(x)
        x = self.Dropout(x)
        
        return x

class LambdaLayer(Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
        
    def forward(self, x):
        return self.lambd(x)

class Dual_Forward(Module):
    def __init__(self, MSE_layer, Quantile_layer):
        super(Dual_Forward, self).__init__()
        self.MSE_layer = MSE_layer
        self.Quantile_layer = Quantile_layer
        
    def forward(self, decoded):
        return self.MSE_layer(decoded), self.Quantile_layer(decoded)