# Basic softwares installation (Windows Only)

This document list the softwares any member or person working with MPG Partners, and more specifically CodPy, might need. It details the installation process for Windows only.

A note for Python aficionados : the current process is still quite hand-crafted, there will be some work to automize it a bit.

# Direct installations
We recommend downloading the following softwares :
* <img src="README.img\logo_R.png" width="40"> [R](https://www.r-project.org): use any CRAN mirror for download
* <img src="README.img\logo_RStudio.png" width="40"> [RStudio](https://rstudio.com): see the download link, then choose the free version
* <img src="README.img\logo_MiKTEX.png" width="40"> [MiKTEX](https://miktex.org): see the download tab
* <img src="README.img\logo_Everything.png" width="40"> [Everything](https://www.voidtools.com/downloads/)
* <img src="README.img\logo_github.png" width="40"> [GitHub Desktop](https://desktop.github.com)
* <img src="README.img\logo_VSC.png" width="40"> [Visual Studio Code](https://code.visualstudio.com)

Those installations should be fine using the latest (64 bits) version and the default settings for each software .
In case some issues are encountered, here are the detailed versions used as of January 2021 :
* R : [4.0.3](https://cran.rstudio.com)
* RStudio: [1.4.1103](https://rstudio.com/products/rstudio/release-notes/)
* Miktex: [21.1](https://miktex.org/download)

# <img src="README.img\logo_intel.png" width="40"> Intel® oneAPI Math Kernel Library
Codpy uses math tools from the [Intel® oneAPI Math Kernel Library](https://software.intel.com/content/www/us/en/develop/articles/oneapi-standalone-components.html#onemkl).
Make sure you only download the MKL component, as such :

<img src="README.img\MKL.png" width="700">

# <img src="README.img\logo_Rmd.png" width="40"> RMarkdown
Once R and RStudio are installed, open the latter.
In the console, enter "*install.packages("rmarkdown")*" to install [RMarkdown](https://rmarkdown.rstudio.com/index.html).

# <img src="README.img\logo_Python.png" width="40">  Python

NOTE : you might already have some versions of Python (be it Python itself or via Anaconda for instance) installed on your computer. It is still important to download the version that is described here. 

Also, it is easier to install Python at the root, but if for any reason (you already installed it, or you just prefer somewhere else) you install it in another folder, make sure to remember the path (or use Everything to find *python.exe* in a folder named *Python37*). We will deal a lot with paths in this section and will refer to them as "path containing this file" so you can find them with Everything, and write the path that worked for us as an example in parenthesis.

## Installation
First, download [Python 3.7.7](https://www.python.org/downloads/release/python-377/) (select *Windows x86-64 executable installer*).

<img src="README.img\Python_version.png" width="700">

Choose the customized istallation, and in the custom parameters :
* install Python, "at the root" ("*C:\Python*") if you are able to do so.
* make sure to "*add Python to the PATH*"
* make sure pip will be installed by typing: "*pip*"

## Librairies
Open the command prompt (*invite de commande*) by typing "cmd" in the Windows search bar.
Then :

* you can check that pip is installed by typing "*pip*"
* install pandas by typing: "*pip install pandas*"
* install numpy by typing: "*pip install numpy*" (might indicate that numpy is already installed)
* install matplotlib by typing: "*-m pip install -U matplotlib*"
Don't hesitate to use *Everything* to check if "pandas", "numpy" and "matplotlib" are installed (just look for their names and make sure *Everything* finds some files).

<img src="README.img\Everything_query.png" width="500">

## GitHub & Codpy

By now, you should have received an invitation to join the team repository (from Jean-Mark a.k.a JohnleM).

<img src="README.img\Invitation.png" width="700">

View the invitation : this will open a github tab.
Click on code, and choose to open with GitHub Desktop.

<img src="README.img\Open_github.png" width="700">

You are now on GitHub Desktop.

* Select *Fetch origin* : you should now be able to "clone" codpy on the repository you want.
* We recommend opening a codpy file in your Documents, and cloning in it, but it's you choice.

<img src="README.img\github_desktop.png" width="700">

Then :

* Access *codpy>codpy* : you will find two "codpy" files. Copy both of them.
* Find the python directory (you can look for *python.exe* in the Everything software) and access the "site-packages" file. Path shoud be *Python>Python37>Lib>site-packages*
* Paste the two copied files.

<img src="README.img\Transfert_codpy.png" width="700">

## Environment variables
We now need to update the environment variables.

* In the Windows search bar, look for "*Edit the system environment variables*" (typing "environment" or "variable" should suffice)
* In the new window, select "Environment variables..."

Now, in the second frame ("*System variables*"), create two new variables :

* Call the first one "*PYTHONHOME*" with as value the folder where you installed Python ("*C:\Python\Python37*")
* Call the second one "*PYTHONPATH*" with three different values (you can either separate the paths with a semicolumn, or add them separately) which are all subfolders of the one you just used :
    * the DLLs subfolder ("*C:\Python\Python37\DLLs*")
    * the Lib subfolder ("*C:\Python\Python37\Lib*")
    * the Lib\site-packages subfolder ("*C:\Python\Python37\Lib\site-packages*")
    
<img src="README.img\Variables_environnement_bis.png" width="700">

Select the "*Path*" variable and, as values, add the following paths (use Everything to find them):

* Path to the Python37 folder ("*C:\Python\Python37*")
* Path to the folder containing the *mkl_rt.1.dll* file ("*C:\Program Files (x86)\Intel\oneAPI\mkl\2021.1.1\redist\intel64*")
* "*C:\Program Files (x86)\Intel\oneAPI\compiler\2021.1.1\windows\redist\intel64_win\compiler*" (**A VERIFIER**)
* "Path to the folder containing the *libiomp5md.dll* file *C:\Users\barry\OneDrive\Documents\codpy\codpy\dlls*"
      
<img src="README.img\Variables_environnement.png" width="700">

Close all windows.

# Testing with Visual Studio Code

You can now check the installation worked.
With Visual Studio Code, open the *Example1D2D.py* file found in the *\codpy\codpy\apps\Examples1D2D* folder.

<img src="README.img\example_launch.png" width="700">

Hit F5, then "Python File - Debug the currently active Python file", or just Ctrl + F5.
If everything works, you should have this output :

<img src="README.img\example_worked.png" width="700">

Close all windows.
