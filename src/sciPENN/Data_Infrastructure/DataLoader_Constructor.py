from .Samplers import build_trainSamplers, build_testSampler
from .DataLoader import tensor_loader, tensor_loader_basic

def build_dataloaders(gene_train, protein_train, gene_test, bools, train_keys, val_frac, batch_size, device, celltypes = None, categories = None):
    sampler_train, sampler_val = build_trainSamplers(gene_train, train_keys, batch_size, val_frac)
    indexer_test = build_testSampler(gene_train, train_keys, batch_size)

    DS_train = tensor_loader(gene_train, protein_train, protein_boolean = bools, sampler = sampler_train, device = device, celltypes = celltypes, categories = categories)
    DS_val = tensor_loader(gene_train, protein_train, protein_boolean = bools, sampler = sampler_val, device = device, celltypes = celltypes, categories = categories)
    DS_impute = tensor_loader(gene_train, protein_boolean = bools, sampler = indexer_test, device = device, celltypes = celltypes, categories = categories)
    
    if gene_test is not None:
        DS_test = tensor_loader_basic(gene_test, batch_size, device = device)
    else:
        DS_test = None
    
    return DS_train, DS_val, DS_impute, DS_test