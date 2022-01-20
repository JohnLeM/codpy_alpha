# CodPy installation (Windows Only)

CodPy is a RKHS based methods implementation. 

## Python installation

NOTE : you might already have some versions of Python (be it Python itself or via Anaconda for instance) installed on your computer. It is still important to download the version that is described here. 

Also, it is easier to install Python at the root, but if for any reason (you already installed it, or you just prefer somewhere else) you install it in another folder, make sure to remember the path (or use Everything to find *python.exe* in a folder named *Python37*).

## Installation
First, download [Python 3.7.7](https://www.python.org/downloads/release/python-377/) (select *Windows x86-64 executable installer*).

Choose the customized installation, and in the custom parameters :
* install Python, "at the root" ("*C:\Python*") if you are able to do so.
* make sure to "*add Python to the PATH*"
* make sure pip will be installed by typing: "*pip*"

## Environment variables
We now need to update the environment variables.

* open CMD as administrator and use the following instructions:
    * $setx /M path "%path%;C:\Python\Lib\site-packages"
    * $setx /M path "%path%;C:\Python\Library\bin"

# Codpy Installation
* Unzip the key "ssh_key.zip" that you received by e-mail to the folder where CodPy will be cloned.
* Open the command line (CMD) and open the directory, where unzipped "ssh_key" is saved. 
* Clone  the CodPy Library via 
  * $git clone git@github.com:codpy2020/CodPy_Test.git
  * Enter the password that you received with SSH key;
* Find Current_Directory using the following instruction in the CMD:
  * $cd 
* Install CodPy via 
  * $pip install "Current_Directory\CodPy_Test"

## Direct Test in CMD:
* open CMD and open python:
  * $python
  
  Use the following instructions:
    * import os, sys
    * sys.path.append(os.path.join(sys.exec_prefix, "Lib","site-packages","codpy"))
    * import codpy.codpy as cd
    * dir(cd)

# Additional software installations
We recommend downloading the following softwares :
* [R](https://www.r-project.org) and [RStudio](https://rstudio.com)
* [MiKTEX](https://miktex.org)
* [Visual Studio Code](https://code.visualstudio.com)

## RMarkdown
Once R and RStudio are installed, open the latter.
In the console, enter "*install.packages("rmarkdown")*" to install [RMarkdown](https://rmarkdown.rstudio.com/index.html).
