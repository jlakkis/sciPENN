from numpy import setdiff1d, intersect1d
from scipy.sparse import issparse
from os.path import isfile

from torch.cuda import is_available
from torch.nn import MSELoss
from torch import load as torch_load, save

from .Utils import build_dir
from .Preprocessing import preprocess
from .Data_Infrastructure.DataLoader_Constructor import build_dataloaders
from .Network.Model import sciPENN_Model
from .Network.Losses import cross_entropy, mse_quantile, no_loss

class sciPENN_API(object):
    def __init__(self, gene_trainsets, protein_trainsets, gene_test = None, gene_list = [], select_hvg = True, train_batchkeys = None, test_batchkey = None, type_key = None, cell_normalize = True, log_normalize = True, gene_normalize = True, min_cells = 30, min_genes = 200, 
                 batch_size = 128, val_split = 0.1, use_gpu = True):

        if use_gpu:
            print("Searching for GPU")
            
            if is_available():
                print("GPU detected, using GPU")
                self.device = 'cuda'
                
            else:
                print("GPU not detected, falling back to CPU")
                self.device = 'cpu'
                
        else:
            print("Using CPU")
            self.device = 'cpu'
            
        preprocess_args = (gene_trainsets, protein_trainsets, gene_test, train_batchkeys, test_batchkey, type_key,
                 gene_list, select_hvg, cell_normalize, log_normalize, gene_normalize, min_cells, min_genes)

        genes, proteins, genes_test, bools, train_keys, categories = preprocess(*preprocess_args)

        self.proteins = proteins
        self.train_genes = genes.var.copy()
        self.train_cells = genes.obs.copy()
        self.type_key = type_key
        self.categories = categories
        
        if genes_test is not None:
            self.test_cells = genes_test.obs.copy()
        else:
            self.test_cells = None
            
        if categories is not None:
            celltypes = proteins.obs[type_key]
        else:
            celltypes = None
        
        dataloaders = build_dataloaders(genes, proteins, genes_test, bools, train_keys, val_split, batch_size, self.device, celltypes, categories)
        self.dataloaders = {key: loader for key, loader in zip(['train', 'val', 'impute', 'test'], dataloaders)}
        
    def train(self, quantiles = [0.1, 0.25, 0.75, 0.9], n_epochs = 10000, ES_max = 12, decay_max = 6, 
              decay_step = 0.1, lr = 10**(-3), weights_dir = "prediction_weights", load = True):
    
        self.quantiles = quantiles
        protein_loss = mse_quantile(self.device, quantiles)
        if self.categories is not None:
            type_loss = cross_entropy
        else:
            type_loss = no_loss(self.device)
        
        p_mod1, p_mod2 = self.train_genes.shape[0], self.proteins.shape[1]
        model_params = {'p_mod1': p_mod1, 'p_mod2': p_mod2, 'loss1': type_loss, 'loss2': protein_loss, 'quantiles': quantiles, 'categories': self.categories}

        self.model = sciPENN_Model(**model_params)
        self.model.to(self.device)
        
        build_dir(weights_dir)
        path = weights_dir + "/sciPENN_Weights"
        
        if load and isfile(path):
            self.model.load_state_dict(torch_load(path))
            
        else:
            train_params = (self.dataloaders['train'], self.dataloaders['val'], n_epochs, ES_max, decay_max, decay_step, lr)
        
            self.model.train_backprop(*train_params)
            save(self.model.state_dict(), path)
        
    def impute(self, requested_quantiles = 'all', denoise_genes = True):
        if requested_quantiles == 'all':
            requested_quantiles = self.quantiles
        else:
            assert type(requested_quantiles) == list
            
        return self.model.impute(self.dataloaders['impute'], requested_quantiles, denoise_genes, self.proteins)
        
    def predict(self, requested_quantiles = 'all', denoise_genes = True):
        assert self.test_cells is not None
        
        if requested_quantiles == 'all':
            requested_quantiles = self.quantiles
        else:
            assert type(requested_quantiles) == list
            
        return self.model.predict(self.dataloaders['test'], requested_quantiles, denoise_genes, self.proteins, self.test_cells)
    
    def embed(self,):
        if self.test_cells is not None:
            loaders = self.dataloaders['impute'], self.dataloaders['test']
        else:
            loaders = self.dataloaders['impute'], None
        return self.model.embed(*loaders, self.train_cells, self.test_cells)