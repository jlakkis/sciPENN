from math import log, exp
from numpy import inf, zeros, zeros_like as np_zeros_like, arange, asarray, empty
from pandas import concat
from anndata import AnnData

from torch import cat, no_grad, randn, zeros_like, zeros as torch_zeros, ones, argmax
from torch.nn import Module, Linear, Sequential, RNNCell, Softplus, Parameter, Softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from .Layers import Input_Block, FF_Block, LambdaLayer, Dual_Forward

class sciPENN_Model(Module):
    def __init__(self, p_mod1, p_mod2, loss1, loss2, quantiles, categories):
        super(sciPENN_Model, self).__init__()
        
        h_size, drop_rate = 512, 0.25
        
        self.RNNCell = RNNCell(h_size, h_size)
        
        self.input_block = Input_Block(p_mod1, h_size, drop_rate, drop_rate)
        
        self.skip_1 = FF_Block(h_size, drop_rate)
        self.skip_2 = FF_Block(h_size, drop_rate)
        self.skip_3 = FF_Block(h_size, drop_rate)
        
        MSE_output = Linear(h_size, p_mod2)
        
        if len(quantiles) > 0:
            quantile_layer = []
            quantile_layer.append(Linear(h_size, p_mod2 * len(quantiles)))
            quantile_layer.append(LambdaLayer(lambda x: x.view(-1,  p_mod2, len(quantiles))))
            quantile_layer = Sequential(*quantile_layer)

            self.mod2_out = Dual_Forward(MSE_output, quantile_layer)
            
        else:
            self.mod2_out = MSE_output
                
        if categories is not None:
            self.celltype_out = Sequential(Linear(h_size, len(categories)), Softmax(1))
            self.forward = self.forward_transfer
            
            self.categories_arr = empty((len(categories), ), dtype = 'object')
            for cat in categories:
                self.categories_arr[categories[cat]] = cat
                
        else:
            self.forward = self.forward_simple
            self.categories_arr = None
            
        self.quantiles = quantiles
        self.loss1, self.loss2 = loss1, loss2
        
    def forward_transfer(self, x):
        x = self.input_block(x)
        h = self.RNNCell(x, zeros_like(x))
        
        x = self.skip_1(x)
        h = self.RNNCell(x, h)
        
        x = self.skip_2(x)
        h = self.RNNCell(x, h)
        
        x = self.skip_3(x)
        h = self.RNNCell(x, h)
        
        return {'celltypes': self.celltype_out(h.detach()), 'modality 2': self.mod2_out(h), 'embedding': h}
    
    def forward_simple(self, x):
        x = self.input_block(x)
        h = self.RNNCell(x, zeros_like(x))
        
        x = self.skip_1(x)
        h = self.RNNCell(x, h)
        
        x = self.skip_2(x)
        h = self.RNNCell(x, h)
        
        x = self.skip_3(x)
        h = self.RNNCell(x, h)
        
        return {'celltypes': None, 'modality 2': self.mod2_out(h), 'embedding': h}
    
    def train_backprop(self, train_loader, val_loader, 
                 n_epoch = 10000, ES_max = 30, decay_max = 10, decay_step = 0.1, lr = 10**(-3)):
        optimizer = Adam(self.parameters(), lr = lr)
        scheduler = StepLR(optimizer, step_size = 1, gamma = decay_step)
        
        patience = 0
        bestloss = inf
        
        if self.categories_arr is None:
            get_correct = lambda x: 0
        else:
            get_correct = lambda outputs: (argmax(outputs['celltypes'], axis = 1) == celltypes).sum()
        
        for epoch in range(n_epoch):
            with no_grad():
                running_loss, rtype_acc = 0., 0.
                self.eval()
                
                for batch, inputs in enumerate(val_loader):
                    mod1, mod2, protein_bools, celltypes = inputs
                    outputs = self(mod1)
                    
                    n_correct = get_correct(outputs)
                    mod2_loss = self.loss2(outputs['modality 2'], mod2, protein_bools)

                    rtype_acc += n_correct
                    running_loss += mod2_loss.item() * len(mod2)
                    
                if self.categories_arr is None:
                    print(f"Epoch {epoch} prediction loss = {running_loss/len(val_loader):.3f}")
                else:
                    print(f"Epoch {epoch} prediction loss = {running_loss/len(val_loader):.3f}, validation accuracy = {rtype_acc/len(val_loader):.3f}")

                patience += 1
                
                if bestloss/1.005 > running_loss:
                    bestloss, patience = running_loss, 0

                if (patience + 1) % decay_max == 0:
                    scheduler.step()
                    print(f"Decaying loss to {optimizer.param_groups[0]['lr']}")

                if (patience + 1) > ES_max:
                    break

            self.train()
            for batch, inputs in enumerate(train_loader):
                optimizer.zero_grad()
                
                mod1, mod2, protein_bools, celltypes = inputs    
                outputs = self(mod1)
                
                mod1_loss = self.loss1(outputs['celltypes'], celltypes)
                mod2_loss = self.loss2(outputs['modality 2'], mod2, protein_bools)
                
                loss = mod1_loss + mod2_loss
                loss.backward()
                    
                optimizer.step()
                
    def impute(self, impute_loader, requested_quantiles, denoise_genes, proteins):
        imputed_test = proteins.copy()

        for quantile in requested_quantiles:
            imputed_test.layers['q' + str(round(100 * quantile))] = np_zeros_like(imputed_test.X)
            
        self.eval()
        start = 0
        
        for mod1, bools, celltypes in impute_loader:
            end = start + mod1.shape[0]
            
            with no_grad():
                outputs = self(mod1)

            if len(self.quantiles) > 0:
                mod2_impute, mod2_quantile = outputs['modality 2']
            else:
                mod2_impute = outputs['modality 2']

            imputed_test.X[start:end] = self.fill_predicted(imputed_test.X[start:end], mod2_impute, bools)

            for quantile in requested_quantiles:
                index = [i for i, q in enumerate(self.quantiles) if quantile == q][0]
                q_name = 'q' + str(round(100 * quantile))
                imputed_test.layers[q_name][start:end] = mod2_quantile[:, : , index].cpu().numpy()

            start = end
        
        return imputed_test
    
    def embed(self, impute_loader, test_loader, cells_train, cells_test):
        if cells_test is not None:
            embedding = AnnData(zeros(shape = (len(cells_train) + len(cells_test), 512)))
            embedding.obs = concat((cells_train, cells_test), join = 'inner')
        else:
            embedding = AnnData(zeros(shape = (len(cells_train), 512)))
            embedding.obs = cells_train
        
        self.eval()
        start = 0
        
        for mod1, bools, celltypes in impute_loader:
            end = start + mod1.shape[0]
            outputs = self(mod1)
            
            embedding[start:end] = outputs['embedding'].detach().cpu().numpy()
            
            start = end
            
        if cells_test is not None:
            for mod1 in test_loader:
                end = start + mod1.shape[0]
                outputs = self(mod1)

                embedding[start:end] = outputs['embedding'].detach().cpu().numpy()

                start = end
            
        return embedding
                    
    def fill_predicted(self, array, predicted, bools):
        bools = bools.cpu().numpy()
        return (1. - bools) * predicted.cpu().numpy() + array
    
    def predict(self, test_loader, requested_quantiles, denoise_genes, proteins, cells):
        imputed_test = AnnData(zeros(shape = (len(cells), len(proteins.var))))
        imputed_test.obs = cells
        imputed_test.var.index = proteins.var.index
        
        if self.categories_arr is not None:
            celltypes = ['None'] * len(cells)
               
        for quantile in requested_quantiles:
            imputed_test.layers['q' + str(round(100 * quantile))] = np_zeros_like(imputed_test.X)
            
        self.eval()
        start = 0
        
        for mod1 in test_loader:
            end = start + mod1.shape[0]
            
            with no_grad():
                outputs = self(mod1)
            
            if self.categories_arr is not None:
                predicted_types = argmax(outputs['celltypes'], axis = 1).cpu().numpy()
                celltypes[start:end] = self.categories_arr[predicted_types].tolist()
            
            if len(self.quantiles) > 0:
                mod2_impute, mod2_quantile = outputs['modality 2']
            else:
                mod2_impute = outputs['modality 2']

            imputed_test.X[start:end] = mod2_impute.cpu().numpy()

            for quantile in requested_quantiles:
                index = [i for i, q in enumerate(self.quantiles) if quantile == q][0]
                q_name = 'q' + str(round(100 * quantile))
                imputed_test.layers[q_name][start:end] = mod2_quantile[:, : , index].cpu().numpy()

            start = end
        
        if self.categories_arr is not None:
            imputed_test.obs['transfered cell labels'] = celltypes
        
        return imputed_test