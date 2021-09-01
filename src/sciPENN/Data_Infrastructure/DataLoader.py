from torch import tensor, long

def tensor_loader(*adatas, protein_boolean, sampler, device, celltypes = None, categories = None):
    if (celltypes is not None) and (categories is not None):
        return tensor_loader_trihead(*adatas, protein_boolean = protein_boolean, sampler = sampler, device = device, celltypes = celltypes, categories = categories)
    
    return tensor_loader_dualhead(*adatas, protein_boolean = protein_boolean, sampler = sampler, device = device)

class tensor_loader_dualhead:
    def __init__(self, *adatas, protein_boolean, sampler, device):
        assert all([adata.shape[0] == adatas[0].shape[0] for adata in adatas])
        
        self.arrs = [adata.X for adata in adatas]
        self.bools = protein_boolean
        
        self.sampler = sampler
        self.device = device
        
    def __iter__(self):
        for idxs, bool_set in self.sampler:
            arrays = [arr[idxs] for arr in self.arrs]
            arrays = [tensor(arr, device = self.device) for arr in arrays]
            arrays.append(tensor(self.bools[bool_set], device = self.device))
            arrays.append(None)
            
            yield arrays
    
    def __len__(self):
        return len(self.sampler)
    
class tensor_loader_trihead:
    def __init__(self, *adatas, protein_boolean, sampler, device, celltypes, categories):
        assert all([adata.shape[0] == adatas[0].shape[0] for adata in adatas])
        
        self.arrs = [adata.X for adata in adatas]
        self.celltypes, self.categories = celltypes, categories
        self.bools = protein_boolean
        
        self.sampler = sampler
        self.device = device
        
    def __iter__(self):
        for idxs, bool_set in self.sampler:
            arrays = [arr[idxs] for arr in self.arrs]
            arrays = [tensor(arr, device = self.device) for arr in arrays]
            arrays.append(tensor(self.bools[bool_set], device = self.device))
            arrays.append(tensor([self.categories[cat] for cat in self.celltypes[idxs]], device = self.device, dtype = long))
                        
            yield arrays
    
    def __len__(self):
        return len(self.sampler)
    
    
class tensor_loader_basic:
    def __init__(self, adata, batch_size, device):
        self.arr = adata.X.copy()
        self.batch_size = batch_size
        self.device = device
        
    def __iter__(self):
        batch = []
        
        for idx in range(len(self.arr)):
            batch.append(idx)
            
            if len(batch) == self.batch_size:
                yield tensor(self.arr[batch], device = self.device)
                batch = []
                
        if batch:
            yield tensor(self.arr[batch], device = self.device)
            
    def __len__(self):
        return len(self.arr)