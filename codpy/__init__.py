print("init entrypoint") 
import sys,os
codpy_path = os.path.dirname(__file__)
print(codpy_path)
if codpy_path not in sys.path: 
    print("codpy_path added") 
    sys.path.append(codpy_path)
mkl_path = os.path.dirname(os.path.dirname(sys.executable))
mkl_path = os.path.join(mkl_path,"Library","bin")
if mkl_path not in sys.path: 
    print("mkl_path added") 
    sys.path.append(mkl_path)
print(sys.path)
import common_include
print("codpylib_path added") 
