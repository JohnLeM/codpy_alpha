@echo off
REM CODPY Python path configuration. set PythonDIR to your python 2 or 3 install path; e.g. the folder with python.exe in it.

setx /M path "%path%;C:\informatique\python\Library\bin;C:\informatique\python\Lib\site-packages;C:\informatique\python\DLLs"
setx /M PYTHONPATH "C:\informatique\python\Lib\site-packages\codpy"

REM set PythonDIR=C:\informatique\python
REM set PATH=%PythonDIR%;%PythonDIR%\Scripts;%PATH%
REM set PYTHONPATH=%PythonDIR%\Lib;%PythonDIR%\Lib\site-packages;%PythonDIR%\DLLs;


