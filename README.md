# sciPENN

sciPENN (**s**ingle **c**ell **i**mputation **P**rotein **E**mbedding **N**eural **N**etwork) is a deep learning computational tool that is useful for analyses of CITE-seq data. sciPENN can be used to:

1. Predict proteins in a query scRNA-seq dataset from a reference CITE-seq dataset.
2. Integrate the scRNA-seq and CITE-seq data into a shared latent space.
3. Combine multiple CITE-seq datasets with different protein panels by imputing missing proteins for each CITE-seq dataset.
4. Transfer cell-type labels from a reference CITE-seq dataset to a query scRNA-seq dataset.

## Reproducibility

To find code to reproduce the results we generated in that paper, please visit this separate [github repository](https://github.com/jlakkis/sciPENN_codes), which provides all code (including that for other methods) necessary to reproduce our results.

## Installation

Recomended installation procedure is as follows. 

1. Install [Anaconda](https://www.anaconda.com/products/individual) if you do not already have it. 
2. Create a conda environment, and then activate it as follows in terminal.

```
$ conda create -n scipennenv
$ conda activate scipennenv
```

3. Install an appropriate version of python.

```
$ conda install python==3.7
```

4. Install nb_conda_kernels so that you can change python kernels in jupyter notebook.

```
$ conda install nb_conda_kernels
```

5. Finally, install sciPENN.

```
$ pip install sciPENN
```

Now, to use sciPENN, always make sure you activate the environment in terminal first ("conda activate scipennenv"). And then run jupyter notebook. When you create a notebook to run sciPENN, make sure the active kernel is switched to "scipennenv"

## Usage

A [tutorial jupyter notebook](https://drive.google.com/drive/folders/1iY4s76UYNMFvF6v3XN4JxD9gM77NIxoH?usp=sharing), together with a dataset, is publicly downloadable.

## Software Requirements

- Python >= 3.7
- torch >= 1.6.1
- scanpy >= 1.7.1
- pandas >= 1.1.5
- numpy >= 1.20.1
- scipy >= 1.6.1
- tqdm >= 4.59.0
- anndata >= 0.7.5
- numba <= 0.50.0