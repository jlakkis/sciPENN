import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sciPENN",
    version="0.9.6",
    author="Justin Lakkis",
    author_email="jlakks@gmail.com",
    description="A package for integrative and predictive analysis of CITE-seq data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jlakkis/sciPENN",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=['torch>=1.6.1', 'numba<=0.50.0', 'scanpy>=1.7.1', 'pandas>=1.1.5', 'numpy>=1.20.1', 'scipy>=1.6.1', 'tqdm>=4.59.0', 'anndata>=0.7.5'],
    python_requires=">=3.7",
)