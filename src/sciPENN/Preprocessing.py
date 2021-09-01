from tqdm import tqdm
from time import sleep
from numpy import intersect1d, setdiff1d, quantile, unique, asarray, zeros
from numpy.random import choice, seed
from scipy.sparse import issparse

from anndata import AnnData
import scanpy as sc

sc.settings.verbosity = 0

def preprocess(gene_trainsets, protein_trainsets, gene_test = None, train_batchkeys = None, test_batchkey = None, type_key = None, gene_list = [], select_hvg = True, cell_normalize = True, log_normalize = True, gene_normalize = True, min_cells = 0, min_genes = 0):
    assert type(gene_trainsets) == list
    assert type(protein_trainsets) == list
    assert all([sum(x.obs.index == y.obs.index)/len(x) == 1. for x, y in zip(gene_trainsets, protein_trainsets)])
    
    if type_key:
        assert all([type_key in x.obs.columns for x in protein_trainsets])
        categories, i = {}, 0
        
        for dataset in protein_trainsets:
            for celltype in dataset.obs[type_key]:
                if celltype not in categories:
                    categories[celltype] = i
                    i += 1
     
    else:
        categories = None
            
    if train_batchkeys is not None:
        assert type(train_batchkeys) == list
        for i in range(len(gene_trainsets)):
            key = train_batchkeys[i]
            gene_trainsets[i].obs['batch'] = ['DS-' + str(i + 1) + ' ' + x for x in gene_trainsets[i].obs[key]]
            protein_trainsets[i].obs['batch'] = ['DS-' + str(i + 1) + ' ' + x for x in protein_trainsets[i].obs[key]]
    else:
        for i in range(len(gene_trainsets)):
            gene_trainsets[i].obs['batch'] = 'DS-' + str(i + 1)
            protein_trainsets[i].obs['batch'] = 'DS-' + str(i + 1)
        
    if gene_test is not None:
        if test_batchkey is not None:
            gene_test.obs['batch'] = ['DS-Test ' + x for x in gene_test.obs[test_batchkey]]
        else:
            gene_test.obs['batch'] = 'DS-Test'
                    
        
    gene_set = set(gene_list)
    
    if min_genes:
        print("\nQC Filtering Training Cells")
        
        for i in range(len(gene_trainsets)):
            cell_filter = (gene_trainsets[i].X > 10**(-8)).sum(axis = 1) >= min_genes
            gene_trainsets[i] = gene_trainsets[i][cell_filter].copy()
            protein_trainsets[i] = protein_trainsets[i][cell_filter].copy()
        
        if gene_test is not None:
            print("QC Filtering Testing Cells")
            
            cell_filter = (gene_test.X > 10**(-8)).sum(axis = 1) >= min_genes
            gene_test = gene_test[cell_filter].copy()
        
    if min_cells:
        print("\nQC Filtering Training Genes")
        
        for i in range(len(gene_trainsets)):
            bools = (gene_trainsets[i].X > 10**(-8)).sum(axis = 0) >= min_cells            
            genes = gene_trainsets[i].var.index[asarray(bools)[0]]
            genes = asarray(genes).reshape((-1,))
            features = set(genes)
            features.update(gene_set)
            features = list(features)
            features.sort()

            gene_trainsets[i] = gene_trainsets[i][:, features].copy()

        if gene_test is not None:
            print("QC Filtering Testing Genes")

            bools = (gene_test.X > 10**(-8)).sum(axis = 0) >= min_cells
            genes = gene_test.var.index[asarray(bools)[0]]
            genes = asarray(genes).reshape((-1,))
            features = set(genes)
            features.update(gene_set)
            features = list(features)
            features.sort()

            gene_test = gene_test[:, features].copy()
        
    for i in range(len(gene_trainsets)):
        gene_trainsets[i].layers["raw"] = gene_trainsets[i].X.copy()
        protein_trainsets[i].layers["raw"] = protein_trainsets[i].X.copy()
    if gene_test is not None:
        gene_test.layers["raw"] = gene_test.X.copy()
            
    if cell_normalize:
        print("\nNormalizing Training Cells")
        
        [sc.pp.normalize_total(x) for x in gene_trainsets]
        [sc.pp.normalize_total(x) for x in protein_trainsets]

        if gene_test is not None:
            print("Normalizing Testing Cells")
            sc.pp.normalize_total(gene_test, key_added = "scale_factor")

    if log_normalize:
        print("\nLog-Normalizing Training Data")
        
        [sc.pp.log1p(x) for x in gene_trainsets]
        [sc.pp.log1p(x) for x in protein_trainsets]
        
        if gene_test is not None:
            print("Log-Normalizing Testing Data")
            sc.pp.log1p(gene_test)
    
    gene_train = gene_trainsets[0]
    gene_train.obs['Dataset'] = 'Dataset 1'
    protein_trainsets[0].obs['Dataset'] = 'Dataset 1'
    for i in range(1, len(gene_trainsets)):
        gene_trainsets[i].obs['Dataset'] = 'Dataset ' + str(i + 1)
        protein_trainsets[i].obs['Dataset'] = 'Dataset ' + str(i + 1)
        gene_train = gene_train.concatenate(gene_trainsets[i], join = 'inner', batch_key = None)
        
    if gene_test is not None:
        genes = intersect1d(gene_train.var.index, gene_test.var.index)
        gene_train = gene_train[:, genes].copy()
        gene_test = gene_test[:, genes].copy()
        
    if select_hvg:
        print("\nFinding HVGs")
        
        if gene_test is not None:
            tmp = gene_train.concatenate(gene_test, batch_key = None).copy()
        else:
            tmp = gene_train.copy()
        
        if not cell_normalize or not log_normalize:
            print("Warning, highly variable gene selection may not be accurate if expression is not cell normalized and log normalized")
            
        if len(tmp) > 10**5:
            seed(123)
            idx = choice(range(len(tmp)), 10**5, False)
            tmp = tmp[idx].copy()
            
        sc.pp.highly_variable_genes(tmp, min_mean = 0.0125, max_mean = 3, min_disp = 0.5, 
                              n_bins = 20, subset = False, batch_key = 'batch', n_top_genes = 1000)
        hvgs = tmp.var.index[tmp.var['highly_variable']].copy()
        tmp = None
        
        gene_set.update(set(hvgs))
        gene_set = list(gene_set)
        gene_set.sort()
        gene_train = gene_train[:, gene_set].copy()
        if gene_test is not None:
            gene_test = gene_test[:, gene_set].copy()
        
    make_dense(gene_train)
    [make_dense(x) for x in protein_trainsets]
    if gene_test is not None:
        make_dense(gene_test)
    
    if gene_normalize:
        patients = unique(gene_train.obs['batch'].values)

        print("\nNormalizing Gene Training Data by Batch")
        sleep(1)

        for patient in tqdm(patients):
            indices = [x == patient for x in gene_train.obs['batch']]
            sub_adata = gene_train[indices].copy()
            sc.pp.scale(sub_adata)

            gene_train[indices] = sub_adata.X.copy()

        print("\nNormalizing Protein Training Data by Batch")
        sleep(1)
        
        for i in range(len(protein_trainsets)):
            patients = unique(protein_trainsets[i].obs['batch'].values)
            for patient in tqdm(patients):
                indices = [x == patient for x in protein_trainsets[i].obs['batch']]
                sub_adata = protein_trainsets[i][indices].copy()
                sc.pp.scale(sub_adata)

                protein_trainsets[i][indices] = sub_adata.X.copy()
                
        if gene_test is not None:
            patients = unique(gene_test.obs['batch'].values)

            print("\nNormalizing Gene Testing Data by Batch")
            sleep(1)
            
            for patient in tqdm(patients):
                indices = [x == patient for x in gene_test.obs['batch']]
                sub_adata = gene_test[indices].copy()
                sc.pp.scale(sub_adata)

                gene_test[indices] = sub_adata.X.copy()

        train_keys, curr_break, proteins = [], len(protein_trainsets[0]), [set(protein_trainsets[0].var.index)]
        protein_train = protein_trainsets[0].copy()
        for i in range(1, len(protein_trainsets)):
            protein_train = protein_train.concatenate(protein_trainsets[i], join = 'outer', fill_value = 0., 
                                                      batch_key = None)
            
            proteins.append(set(protein_trainsets[i].var.index))
            train_keys.append(curr_break)
            curr_break += len(protein_trainsets[i])
        
        bools = asarray([[int(x in prot_set) for x in protein_train.var.index] for prot_set in proteins])
        
        for i in range(len(bools)):
            protein_train.var['Dataset ' + str(i + 1)] = [bool(x) for x in bools[i]]
            
    return gene_train, protein_train, gene_test, bools, train_keys, categories

def make_dense(anndata):
    if issparse(anndata.X):
        tmp = anndata.X.copy()
        anndata.X = tmp.copy().toarray()