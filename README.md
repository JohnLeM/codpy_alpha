# Overview

Codpy is a kernel based, open source software library for high performance numerical computation, relying on the [RKHS](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) theory.
It contains a set of core tools that we use for machine Learning, statistics and numerical simulations. As a machine learning platform, it enjoys some interesting properties :

 * It is a numerically efficient machine learning platform. We provide benchmark tools to compare codpy to other popular machine learning platforms.
 * It is a white box method. Any learning machine has access to worst-error bounds computations. These allow to compute confidence levels of prediction on any test set. Moreover, reproducibility properties of kernel methods allow to fully understand and explain obtained results.
 * Each learning machine has access to all classical differential operators. These properties allow us to use this library with any PDE (partial differential equations) approach.
 * Each learning machine has access to optimal transport tools, much needed for statistics.


## Technical requirement

This version of the library is CPUs based, and is tested on

 * windows / amd64 platforms  

## Directory structure

Once installed (see below), navigate to ```<path\to\python39>\Lib\site-packages\codpy```. The directory structure should be

* ```codpy```
    *   ```docs```  
        *   codpy-book.pdf is the codpy reference manual.  
        *   *.ipynb are jupyter notebooks to reproduce the example of codpy book.  
    *   ```com```  : low level tools and interface.  
    *   ```pred``` : Wrappers to a number of prediction machines : kernels, neural networks, and more.  
    *   ```data``` : Wrappers to data set handling  
    *   ```proj``` : some examples of applications  
        *   BTC_predictor.py    : an example of time serie prediction.  
        *   clustering.py       : benchmarks of clustering methods.  
        *   housing_prices.py   : benchmarks for the venerable Boston house price data set.  
        *   mnist_codpy.py      : benchmarks for the MNIST data set.  
        *   radon.py            : an application for medical imagery.  
        *   reordering.py       : illustration of optimal transport tools.  
    *  README.md        : this document  
    *  __init__.py      : codpy loader  
    *  include.py       : called by __init__  

# Installation 

Note: this installation process has been tested on
 * windows / amd64 platform 

## prerequisite

### Minimum installation

* [python3.9.7](https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe): a valid python python3.9.7 installation.

*NOTE* : Python installation differs from one machine to another. The python root folder is denoted "\<path/to/python39>" in the rest of this document. The software Everything (or another equivalent) can be of great help finding files.



### Dev installations

For information, we list the softwares that we are using for our dev configuration :
* [GitHub Desktop](https://desktop.github.com)
* [R](https://www.r-project.org): use any CRAN mirror for download
* [RStudio](https://rstudio.com): see the download link, then choose the free version
* [MiKTEX](https://miktex.org): see the download tab
* [Everything](https://www.voidtools.com/downloads/)
* [Visual Studio Code](https://code.visualstudio.com)

Those installations should be fine using the latest (64 bits) version and the default settings for each software .

*Note* Once R and RStudio are installed, open the latter.
In the console, enter "*install.packages("rmarkdown")*" to install [RMarkdown](https://rmarkdown.rstudio.com/index.html).

## Cloning repo

Download the codpy repo at [codpy alpha](https://github.com/JohnLeM/codpy_alpha) to your location <path/to/codpyrepo>

If you don't have access to this repo, [Contact Us](mailto:jean-marc.mercoer@mpg-partners.com) to have access, as you should have received an invitation to join the codpy repository.

## Installation

### prerequisite

We suppose that there is a valid python installation on the host machine. The reader can 
* either use its main python environment ```<path/to/python39>```
* or create a virtual python environment ```<path/to/venv>```, a good practice that we describe in the rest of this section.

First open a command shell ```cmd```,  create a virtual environment and activate it.

```
python -m venv .\venv
.\venv\Scripts\activate
```
*NOTE* : In the rest of the installation procedure, we consider a virtual environment <path/to/venv>. One can replace with <path/to/python39> if a main environment installation is desired, for dev purposes for instance.

### pip install codpy

Open a command shell ```cmd```, and pip install codpy

```
pip install codpy
```
or from the local repository

```
pip install <path/to/codpyrepo>/dist/codpy-XXXX.whl
```
The installation procedure might take some minutes depending on your internet connection.

### Test codpy

open a python shell and import codpy
```
python
```
```
import codpy
```

# Testing with Visual Studio Code

You can your visual studio installation.

 - With Visual Studio Code, open the ```<path/to/codpyrepo>``` folder and select for instance the file  ```<path/to/codpyrepo>/proj/clustering.py```

 - Select your python interpreter (Shift+P) 


- Hit F5. If everything works, you should have some figures.
