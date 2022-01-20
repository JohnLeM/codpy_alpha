import sys,os
codpy_path = os.path.dirname(__file__)
if codpy_path not in sys.path: 
    sys.path.append(codpy_path)
mkl_path = os.path.dirname(os.path.dirname(sys.executable))
mkl_path = os.path.join(mkl_path,"Library","bin")
os.environ["PATH"] += os.pathsep + mkl_path
import common_include