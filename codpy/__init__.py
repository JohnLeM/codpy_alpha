import sys,os,ctypes
def mkl_path():
    codpy_path = os.path.dirname(__file__)
    if codpy_path not in sys.path: 
        sys.path.append(codpy_path)
    mkl_path = os.path.dirname(sys.executable)
    mkl_path = os.path.join(mkl_path,"Library","bin")
    os.environ["PATH"] += os.pathsep + mkl_path
    mkl_path = os.path.join(mkl_path,"mkl_rt.2.dll")
    return mkl_path
hllDll = ctypes.WinDLL(mkl_path())
import common_include