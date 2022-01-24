# Installation 

This document aims to describe the codpy installation process.

Note: this installation process has been tested on
 * windows / amd64 platform 

## prerequisite

### Minimum installation

* <img src="README.img\python.jpg" width="40"> [python3.9.7](https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe): a valid python python3.9.7 installation.
* <img src="README.img\logo_github.png" width="40"> [GitHub Desktop](https://desktop.github.com)

*NOTE* : Python installation differs from one machine to another. The python root folder is denoted "\<path/to/python39>" in the rest of this document. The software Everything (or another equivalent) can be of great help finding files.



### Dev installations
For a dev install, we recommend downloading the following softwares :
* <img src="README.img\logo_R.png" width="40"> [R](https://www.r-project.org): use any CRAN mirror for download
* <img src="README.img\logo_RStudio.png" width="40"> [RStudio](https://rstudio.com): see the download link, then choose the free version
* <img src="README.img\logo_MiKTEX.png" width="40"> [MiKTEX](https://miktex.org): see the download tab
* <img src="README.img\logo_Everything.png" width="40"> [Everything](https://www.voidtools.com/downloads/)
* <img src="README.img\logo_VSC.png" width="40"> [Visual Studio Code](https://code.visualstudio.com)

Those installations should be fine using the latest (64 bits) version and the default settings for each software .
In case some issues are encountered, here are the detailed versions used as of January 2021 :
* R : [4.0.3](https://cran.rstudio.com)
* RStudio: [1.4.1103](https://rstudio.com/products/rstudio/release-notes/)
* Miktex: [21.1](https://miktex.org/download)

 <img src="README.img\logo_Rmd.png" width="40"> RMarkdown
Once R and RStudio are installed, open the latter.
In the console, enter "*install.packages("rmarkdown")*" to install [RMarkdown](https://rmarkdown.rstudio.com/index.html).

## Cloning repo

Download the codpy repo at [codpy alpha](https://github.com/JohnLeM/codpy_alpha) to your location <path/to/codpyrepo>

If you don't have access to this repo, [Contact Us](mailto:jean-marc.mercoer@mpg-partners.com) to have access, as you should have received an invitation to join the codpy repository (from the Github account JohnLeM).

<img src="README.img\Invitation.png" width="200">


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
pip install <path/to/codpyrepo>/dist/codpy-0.0.1-cp39-cp39-win_amd64.whl
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

 - With Visual Studio Code, open the ```<path/to/codpyrepo>``` folder and select for instance the file  ```<path/to/codpyrepo>/codpy/clustering/clustering.py```

<img src="README.img\example_launch.png" width="700">


 - Select your python interpreter (Shift+P) <img src="README.img\interpreter.png" width="300">


- Hit F5. If everything works, you should have some figures, the first one being :

<img src="README.img\example_worked.png" width="500">


