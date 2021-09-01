from numpy import arange, setdiff1d
from numpy.random import choice

class batchSampler:
    def __init__(self, indices, train_keys, bsize, shuffle = False):
        self.indices = indices
        self.train_keys = train_keys
        self.bsize = bsize
        self.shuffle = shuffle
        
    def __iter__(self):
        if self.shuffle:
            indices = choice(self.indices, size = len(self.indices), replace = False)
        else:
            indices = self.indices
        
        minibatch_idx = []
        bool_idx = []
        
        for idx in indices:
            minibatch_idx.append(idx)
            bool_idx.append(sum([int(idx >= x) for x in self.train_keys]))
            
            if len(minibatch_idx) >= self.bsize:
                yield minibatch_idx, bool_idx
                minibatch_idx, bool_idx = [], []
                
        if minibatch_idx:
            yield minibatch_idx, bool_idx
        
    def __len__(self):
        return len(self.indices)
    
def build_trainSamplers(adata, n_train, bsize = 128, val_frac = 0.1):
    num_val = round(val_frac * len(adata))
    assert num_val >= 1
    
    idx = arange(len(adata))
    val_idx = choice(idx, num_val, replace = False)

    train_indices, val_indices = setdiff1d(idx, val_idx).tolist(), val_idx.tolist()
    
    train_sampler = batchSampler(train_indices, n_train, bsize, shuffle = True)
    val_sampler = batchSampler(val_indices, n_train, bsize)
    
    return train_sampler, val_sampler

def build_testSampler(adata, train_keys, bsize = 128):
    indices = list(range(len(adata)))
    
    return batchSampler(indices, train_keys, bsize)