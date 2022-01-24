import os, sys
from pathlib import Path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path: sys.path.append(parentdir)
data_path = os.path.dirname(__file__)
from common_include import * 


def standard_rescale(x):
    from sklearn.preprocessing import StandardScaler        
    return StandardScaler().fit_transform(x)

def df_standard_rescale(x):
    from sklearn.preprocessing import StandardScaler        
    return pd.DataFrame(data = StandardScaler().fit(x).transform(x.values), index = x.index)

def df_standard_normalize(x):
    from sklearn.preprocessing import Normalizer
    return pd.DataFrame(data = Normalizer().fit(x).transform(x.values), index = x.index)


def tensor_vectorize(fun, x):
    N,D = x.shape[0],x.shape[1]
    E = len(fun(x[0]))
    out = np.zeros((N, E))
    for n in range(0, N):
        out[n] = fun(x[n])
    return out.reshape(N,E,1)

def matrix_vectorize(fun, x):
    N,D = x.shape[0],x.shape[1]
    out = np.zeros((N, 1))
    for n in range(0, N):
        out[n,0] = fun(x[n])
    return out

################### data_blob_generator ##########################################

class data_random_generator(data_generator):

    def __init__(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        super().__init__(D,Nx,Ny,Nz,**kwargs)
        self.fun = kwargs.get("fun",None)
        self.nabla_fun = kwargs.get("nabla_fun",None)
        self.type = kwargs.get("type","sto")
        self.X_min = kwargs.get("X_min",-1.)
        self.X_max = kwargs.get("X_max",1.)
        self.Z_min = kwargs.get("Z_min",-1.5)
        self.Z_max = kwargs.get("Z_max",1.5)
        self.types = kwargs.get("types",["sto","sto","sto"])
        self.seeds = kwargs.get("seeds",[42,35,52])

    def get_raw_data(self,Nx, Ny, Nz, D,**kwargs):
        import numpy as np
        def cartesian_product(*arrays):
            la = len(arrays)
            dtype = np.result_type(*arrays)
            arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
            for i, a in enumerate(np.ix_(*arrays)):
                arr[...,i] = a
            return arr.reshape(-1, la)        

        def get_array(N,D,min,max,**kwargs):
            import itertools as it
            type = kwargs.get("type","sto")
            seed = kwargs.get("seed",0)
            if type == "sto": 
                if seed : np.random.seed(seed)
                return np.random.uniform(min, max, size = (N,D))
            v = [np.arange(start=min,stop=max+0.00001,step = (max-min) / ( int(N**(1/D) + 1.)) )]
            v = cartesian_product(*(v * D))
            if D==1: return np.asarray(v).reshape(len(v),D)
            v = np.asarray(v)
            return v

        x = get_array(Nx,D,self.X_min,self.X_max,**{**{'type':self.types[0],'seed':self.seeds[0]},**kwargs})
        y = get_array(Ny,D,self.X_min,self.X_max,**{**{'type':self.types[1],'seed':self.seeds[1]},**kwargs})
        z = get_array(Nz,D,self.Z_min,self.Z_max,**{**{'type':self.types[2],'seed':self.seeds[2]},**kwargs})

        Nx = len(x)
        Ny = len(y)
        Nz = len(z)

        Nx,Ny,Nz = len(x), len(y), len(z)
        if self.fun != None: 
            fy = matrix_vectorize(self.fun,y)
            fx = matrix_vectorize(self.fun,x)
            fz = matrix_vectorize(self.fun,z)
            if self.nabla_fun != None: 
                nabla_fx = tensor_vectorize(self.nabla_fun,x)
                nabla_fz = tensor_vectorize(self.nabla_fun,z)
                return(x,y,z,fx,fy,fz,nabla_fx,nabla_fz,Nx,Ny,Nz) 
            else:
                return(x,y,z,fx,fy,fz,Nx, Ny, Nz)
        else:
            return(x,y,z,Nx,Ny,Nz) 

    def get_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        if (D*Nx*Ny*Nz):
         if self.fun != None: 
             if self.nabla_fun != None: x,y,z,fx,fz,nabla_fx,nabla_fz,Nx,Ny,Nz = self.get_raw_data(Nx,Ny,Nz,D,**kwargs)
             else: x,y,z,fx,fy,fz,Nx, Ny, Nz = self.get_raw_data(Nx,Ny,Nz,D,**kwargs)
            
        return  x, fx, y, fy, z, fz
    def copy(self):
        return self.copy_data(data_random_generator())


class data_blob_generator(data_generator):
    def get_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples = Nx+Nz,
        n_features = D,
        centers = Ny, #Ny
        cluster_std=1,
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=1)
        x, z, fx, fz = train_test_split(X, y, train_size=Nx / (Nx+Nz), random_state=42)
        return  x, fx, x, fx, z, fz
    def copy(self):
        return self.copy_data(data_blob_generator())

class data_moon_generator(data_generator):
    def get_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_moons
        X, y = make_moons(Nx+Nz , noise=.07, random_state=1)
        x, z, fx, fz = train_test_split(X, y, train_size=Nx / (Nx+Nz), random_state=42)
        return  x, fx, [], [], z, fz
    def copy(self):
        return self.copy_data(data_moon_generator())


class data_circles_generator(data_generator):
    def get_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=Nx+Nz, noise=0.05, factor=0.6)
        x, z, fx, fz = train_test_split(X, y, train_size=Nx / (Nx+Nz), random_state=42)
        return  x, fx, [], [], z, fz
    def copy(self):
        return self.copy_data(data_moon_generator())

################### MNIST_data_generator ##########################################

class MNIST_data_generator(data_generator):
    tfx, tffx, tfz, tffz = [],[],[],[]
    def get_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        if (len(self.tfx)*len(self.tfz) == 0):
            import tensorflow as tf
            (self.tfx, self.tffx), (self.tfz, self.tffz) = tf.keras.datasets.mnist.load_data()
            self.tfx, self.tfz = self.tfx / 255.0, self.tfz / 255.0
            self.tfx, self.tfz = self.tfx.reshape(len(self.tfx),-1), self.tfz.reshape(len(self.tfz),-1)
        return (self.tfx.copy(), self.tffx.copy(), self.tfx.copy(), self.tffx.copy(),self.tfz.copy(), self.tffz.copy())

    def copy(self):
        return self.copy_data(MNIST_data_generator())

################### raw_data ##########################################
class raw_data_generator(data_generator):
    x_data  = []
    @abc.abstractmethod
    def get_raw_data(self,**kwargs):
        pass
    def get_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        if len(self.x_data) == 0:
            self.x_data = self.get_raw_data(**kwargs)
        return self.x_data, [], self.x_data, [], self.x_data, []

################### credit_data ##########################################

class german_credit_data_generator(raw_data_generator):
    def get_raw_data(self,**kwargs):
        import pandas as pd
        x_data = pd.read_csv(r"https://raw.githubusercontent.com/SaravananJaichandar/Credit-Risk-Model/master/german_credit_data.csv", index_col=0)
        x_data= x_data.dropna()
        x_data=hot_encoder(x_data,cat_cols=['Sex','Housing','Saving accounts','Checking account','Purpose'])
        return df_standard_rescale(x_data)

################### credit_card_data ##########################################
class credit_card_data_generator(raw_data_generator):
    def get_raw_data(self,**kwargs):
        import pandas as pd
        x_data = pd.read_csv(os.path.join(data_path, "CC_GENERAL.csv"), index_col=0)
        x_data['MINIMUM_PAYMENTS'].fillna(x_data['MINIMUM_PAYMENTS'].mean(skipna=True), inplace=True)
        x_data['CREDIT_LIMIT'].fillna(x_data['CREDIT_LIMIT'].mean(skipna=True), inplace=True)
        return df_standard_rescale(x_data)
    def copy(self):
        return self.copy_data(credit_card_data_generator())
################### labeled data ##########################################
class credit_card_fraud_data_generator(raw_data_generator):
    def get_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import train_test_split
        x,fx,y,fy,z,fz = super().get_data(**kwargs)
        if (Nz == -1): Nz = len(z)
        if (Nx == -1): Nx = len(x)

        rob_scaler = RobustScaler()
        x["Time"] = rob_scaler.fit_transform(x['Time'].values.reshape(-1,1))
        x["Amount"] = rob_scaler.fit_transform(x['Amount'].values.reshape(-1,1))


        frauds = x[x["Class"]==1]
        no_frauds = x[x["Class"]==0 ]

        Nx = max(Nx - len(frauds)/2, len(frauds) / 2)

        x, z = train_test_split(no_frauds, train_size=Nx / (Nx+Nz), random_state=42)
        x_train_fraud, z_test_fraud= train_test_split(frauds, train_size=.5, random_state=42)
        x = pd.concat([x,x_train_fraud])
        z = pd.concat([z,z_test_fraud])

        fx = x['Class']
        x = x.drop(['Class'],axis=1)
        fz = z['Class']
        z = z.drop(['Class'],axis=1)
        return x,fx,y,fy,z,fz
    def get_raw_data(self,**kwargs):
        from os.path import exists
        url = '1pzqrcrtz1XLXGx7BKR9v4U_d7SZbEK1o'
        path_to_file = os.path.join(data_path,"creditcardfraud.csv")

        def unzip_file(path_to_file,extract_path = None,**kwargs):
            import zipfile
            if extract_path is None: extract_path = path_to_file
            with zipfile.ZipFile(path_to_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

        def download_api_file(file_id, path_to_file, method = None, **kwargs):
            import googleapiclient
            request = googleapiclient.drive_service.files().get_media(fileId=file_id)
            fh = googleapiclient.io.BytesIO()
            downloader = googleapiclient.MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print("Download %d%%." % int(status.progress() * 100) )       

        def download_url_file(url, path_to_file, method = None, **kwargs):
            import requests
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(path_to_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        # If you have chunk encoded response uncomment if
                        # and set chunk_size parameter to None.
                        #if chunk: 
                        f.write(chunk)
                if method is not None: method(path_to_file=path_to_file,**kwargs)

        def kaggle_api_file(data_set, path_to_file, **kwargs):
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(data_set, path_to_file, **kwargs)

        if not exists(path_to_file): kaggle_api_file(data_set="mlg-ulb/creditcardfraud",path_to_file = path_to_file, unzip = True)
        data = pd.read_csv(path_to_file)
        return data
    def copy(self):
        return self.copy_data(credit_card_fraud_data_generator())
################" Iris " #####################################""
class iris_data_generator(data_generator):
    def get_data(self, D=0,Nx=0,Ny=0,Nz=0, **kwargs):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import label_binarize
        iris = datasets.load_iris()
        proba = kwargs.get('get_proba', 'False')
        X = iris.data 
        y = iris.target
        x, z, fx, fz = train_test_split(X, y, train_size=Nx / (Nx+Nz), random_state=42)
        return x, fx, x, fx, z, fz
    def get_feature_names(self):
        from sklearn import datasets
        return datasets.load_iris().feature_names
    def copy(self):
        return self.copy_data(iris_data_generator())
################" default_pred " #####################################""
class loan_default_data_generator(data_generator):
    def get_data(self, D=0,Nx=0,Ny=0,Nz=0, **kwargs):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing
        df = self.get_raw_data()
        df['due_date'] = pd.to_datetime(df['due_date'])
        df['effective_date'] = pd.to_datetime(df['effective_date'])
        df['dayofweek'] = df['effective_date'].dt.dayofweek
        df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
        df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
        df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
        df.groupby(['education'])['loan_status'].value_counts(normalize=True)
        df[['Principal','terms','age','Gender','education']].head()

        Feature = df[['Principal','terms','age','Gender','weekend']]
        Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
        Feature.drop(['Master or Above'], axis = 1,inplace=True)
        X = Feature
        y = df['loan_status'].values
        X= preprocessing.StandardScaler().fit(X).transform(X)
        if (Nz == -1): Nz = len(X)
        if (Nx == -1): Nx = len(X)
        x, z, fx, fz = train_test_split(X, y, train_size=Nx / (Nx+Nz), random_state=42)
        return x, fx, x, fx, z, fz
    def get_feature_names(self):
        from sklearn import datasets
        return datasets.load_iris().feature_names
    def get_raw_data(self,**kwargs):
        data = pd.read_csv(os.path.join(data_path,"loandata.csv"))
        return data
    def copy(self):
        return self.copy_data(loan_default_data_generator())        
################" airplane " #####################################""
class airplane_data_generator(data_generator):
    def get_data(self, D=0,Nx=0,Ny=0,Nz=0, **kwargs):
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        tensor = kwargs.get('tensor', False)
        sc = MinMaxScaler()
        dataset = self.get_raw_data()
        dataset = sc.fit_transform(dataset)
        train_set, test_set = self.train_test_set(dataset)
        x, fx, z, fz = self.supervised_learning_form(train_set, test_set)
        if tensor:
            x = x.reshape(x.shape[0], x.shape[1], 1)
            z = z.reshape(z.shape[0], z.shape[1], 1)
            fz = fz.reshape(fz.shape[0], 1)
            fz = sc.inverse_transform(fz)
        return x, fx, [], [], z, fz
        
    def train_test_set(self, dataset, percentage_train = 0.75):
        l = len(dataset)
        train_set_end_index = int(l * 0.67)
        test_set_start_index = train_set_end_index + 1
        train_set = dataset[:test_set_start_index]
        test_set = dataset[test_set_start_index:]
        return train_set, test_set

    def supervised_learning_form(self, train_dataset, test_dataset):
        train_x = train_dataset[:-1]
        train_y = train_dataset[1:]
        test_x = test_dataset[:-1]
        test_y = test_dataset[1:]
        return train_x, train_y, test_x, test_y

    def get_raw_data(self,**kwargs):
        data = pd.read_csv(os.path.join(data_path,'airline-passengers.csv'),usecols = [1], header = 0)[:-1]
        return data.values

    def copy(self):
        return self.copy_data(airplane_data_generator())
#############################################################################
class titanic_data_generator(data_generator):
    def get_data(self, D=0,Nx=0,Ny=0,Nz=0, **kwargs):
        import tensorflow_datasets as tfds
        ds = tfds.load(name="titanic", split=tfds.Split.TRAIN, batch_size=-1 )
        train = tfds.as_numpy(tfds.load(name="mnist", split=tfds.Split.TRAIN, batch_size=-1)) 
        test = tfds.as_numpy(tfds.load(name="mnist", split=tfds.Split.TEST, batch_size=-1))
        x, fx = train["image"], train["label"] 
        z, fz = test["image"], test["label"]
        return x, fx, [], [], z, fz
    def copy(self):
        return self.copy_data(titanic_data_generator())


################" Boston Housing prices " #####################################""
class Boston_data_generator(data_generator):
    x_raw, fx_raw, z_raw, fz_raw = [],[],[],[]
    def set_raw_data(self, **kwargs):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        x, fx = datasets.load_boston(return_X_y=True)
        a = np.arange(len(x))
        np.random.seed(42)
        np.random.shuffle(a)
        x = x[a]
        fx = fx.reshape((len(fx),1))
        fx = fx[a]
        Boston_data_generator.x_raw, Boston_data_generator.fx_raw = pd.DataFrame(x),pd.DataFrame(fx)

    def get_data(self, D=0,Nx=0,Ny=0,Nz=0, **kwargs):
        if len(Boston_data_generator.x_raw) == 0:
            self.set_raw_data()
        length = len(Boston_data_generator.x_raw)
        return Boston_data_generator.x_raw[0:Nx], Boston_data_generator.fx_raw[0:Nx], Boston_data_generator.x_raw[0:Nx], Boston_data_generator.fx_raw[0:Nx], Boston_data_generator.x_raw, Boston_data_generator.fx_raw

    def get_feature_names(self):
        from sklearn import datasets
        return datasets.load_boston().feature_names
    def copy(self):
        return self.copy_data(Boston_data_generator())

################"Housing prices " #####################################""
class housing_data_generator(data_generator):
    cols = ['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea',
    'BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','1stFlrSF','LowQualFinSF','GrLivArea',
    'BsmtFullBath','BsmtHalfBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt',
    'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
    'MiscVal','MoSold','YrSold','2ndFlrSF','FullBath','SalePrice']
    cols_cat = ['MSSubClass']
    x_raw, fx_raw, z_raw, fz_raw = [],[],[],[]


    def get_data(self,D=0,Nx=0,Ny=0,Nz=0,**kwargs):
        if len(housing_data_generator.x_raw) == 0:
            self.get_hp_csvdata("hp_train.csv","hp_test.csv")
        return housing_data_generator.x_raw, housing_data_generator.fx_raw, housing_data_generator.x_raw, housing_data_generator.fx_raw, housing_data_generator.z_raw, housing_data_generator.fz_raw


    def copy(self):
        return self.copy_data(housing_data_generator())


    def get_hp_csvdata(self,file_train_name = "hp_train.csv",file_test_name = "hp_train.csv"):
        import os, sys
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_train = pd.read_csv(os.path.join(dir_path, file_train_name))[self.cols].dropna(axis = 1)
        data_test = pd.read_csv(os.path.join(dir_path, file_test_name))[self.cols].dropna(axis = 1)
        final_cols = data_train.columns.intersection(data_test.columns)
        print(final_cols)

        data_train = data_train[final_cols]
        data_test = data_test[final_cols]


        housing_data_generator.x_raw = data_train.drop(['SalePrice'], axis=1)
        housing_data_generator.fx_raw = data_train["SalePrice"]
        housing_data_generator.z_raw = data_test.drop(['SalePrice'], axis=1)
        housing_data_generator.fz_raw = data_test["SalePrice"]

        housing_data_generator.x_raw = hot_encoder(housing_data_generator.x_raw,self.cols_cat)
        housing_data_generator.z_raw = hot_encoder(housing_data_generator.z_raw,self.cols_cat)
################" SP500_data " #####################################""
class SP500_data_generator(raw_data_generator):
    def get_raw_data(self,**kwargs):
        import pandas as pd
        df = pd.read_csv(os.path.join(data_path, "snp.csv"), index_col=0)
        df["Return"]= (df["Adj Close"]-df["Adj Close"].shift(1))/df["Adj Close"].shift(1)
        df["C/V"] = df["Adj Close"]/df["Volume"]
        df["abs_OC"] = abs(df["Open"]-df["Adj Close"])
        # df = df.drop(["Volume", "Open", "High", "Low", "Close", "Adj Close"], axis=1) ??
        return df.dropna()
    def copy(self):
        return self.copy_data(SP500_data_generator())
################### stock_data ##########################################
class company_stock_movements_data_generator(raw_data_generator):
    def get_raw_data(self,**kwargs):
        import pandas as pd
        df = pd.read_csv(os.path.join(data_path, "company-stock-movements-2010-2015-incl.csv"), index_col=0)
        return df_standard_normalize(df)
    def copy(self):
        return self.copy_data(company_stock_movements_data_generator())
################### SP500_Yahoo ##########################################
class SP500_Yahoo_generator(raw_data_generator):
    def get_raw_data(self,**kwargs):
        import pandas as pd
        df = pd.read_csv(os.path.join(data_path, "SP500Fresh_vol.csv"), index_col=0)
        return df_standard_normalize(df)
    def copy(self):
        return self.copy_data(SP500_Yahoo_generator())


class data_plots():
    def corr(x, y, **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np
        coef = np.corrcoef(x, y)[0][1]
        label = r'$\rho$ = ' + str(round(coef, 2))
        ax = plt.gca()
        ax.annotate(label, xy = (0.2, 0.95), size = 10, xycoords = ax.transAxes)

    def distribution_plot1D(x, ax, **kwargs):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        title = kwargs.get('title',"")
        suptitle = kwargs.get('suptitle',"")
        markersize = kwargs.get('markersize',3)
        fmt = kwargs.get('fmt','-bo')
        markerfacecolor = kwargs.get('markerfacecolor','r')
        ax.title.set_text(suptitle)
        sns.histplot(x, kde = True, ax = ax)
    
    def scatter_plot(x, ax, **kwargs):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.scatterplot(data = x, ax = ax)
    
    def scatter_plot_x_fx(self, x, fx, title = None):
        import matplotlib.pyplot as plt
        dot_size = 50
        cmap = 'viridis'
        fig, ax = plt.subplots(figsize=(9,7))
        ax.set_title(title, fontsize=18, fontweight='demi')
        plt.scatter(x[:, 0], x[:, 1], c=fx, s=dot_size, cmap=cmap)

    def heatmap(x, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        f_names = kwargs.get('f_names',"")
        title = kwargs.get('title',"")
        df = pd.DataFrame(x, columns=f_names).corr()
        #df.style.background_gradient(cmap='coolwarm')
        mask = np.triu(np.ones_like(df, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        if ax != None:
            sns.heatmap(df, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, ax=ax).set_title(title)
        else:
            sns.heatmap(df, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True).set_title(title)
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        plt.show()

    def density_scatter(xfx):
        import seaborn as sns
        g = sns.PairGrid(xfx, diag_sharey=False)
        g.map_upper(sns.scatterplot, s=15)
        g.map_upper(data_plots.corr)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot, lw=2)
        plt.show()     
    
    def roc(self, fz, f_z, **kwargs):
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        from itertools import cycle
        import numpy as np
        binarize = kwargs.get('binarize',True)
        title = kwargs.get('title'," ")
        classes = np.unique(fz)
        lw = 2
        if binarize == True:
            fz = label_binarize(fz, classes = classes)
        fpr = dict(); tpr = dict(); roc_auc = dict()
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(fz[:, i], f_z[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        #micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(fz.ravel(), f_z.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # FP rate
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        #average and AUC
        mean_tpr /= len(classes)
        fpr["macro"] = all_fpr; tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i in range(len(classes)):
            plt.plot(fpr[i], tpr[i], lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC to multi-class' + " " + title)
        plt.legend(loc="lower right")
        plt.show()

class shelf_data_generator(raw_data_generator):
    def get_raw_data(self, **kwargs):
        import pandas as pd
        feature_wt = 2
        x = np.array(pd.read_csv(os.path.join(data_path, "shelf.csv")))
        x[:,0] = x[:,0] * feature_wt
        return x
    def get_all_data(self):
        return np.array(pd.read_csv(os.path.join(data_path, "shelf.csv")))
    def copy(self):
        return self.copy_data(shelf_data_generator())


if __name__ == "__main__":
    def my_fun(x):
        import numpy as np
        from math import pi
        D = len(x)
        res = 1.
        for d in range(0,D):
            res *= np.cos(4 * x[d] * pi) 
        for d in range(0,D):
            res += x[d]
        return res

    D,Nx,Ny,Nz=2,1000,500,2000
    data_random_generator_ = data_random_generator(fun = my_fun,types = ["sto","sto","cart"])
    x, fx, y, fy, z, fz =  data_random_generator_.get_data(D=D,Nx=Nx,Ny=Ny,Nz=Nz)
    multi_plot([(x,fx),(z,fz)],plotD,mp_title="x,f(x)  and z, f(z)",projection="3d")

    D,Nx,Ny,Nz= -1,-1,-1,-1
    dp = data_plots()
    x, fx, y, fy, z, fz = Boston_data_generator().get_data(D = D,Nx= Nx, Ny = Ny, Nz = Nz)
    f_names = Boston_data_generator().get_feature_names()
    #multi_plot(x.T,dp.distribution_plot1D,mp_title="x,f(x)  and z, f(z)", f_names = f_names)
    #multi_plot(x.T,dp.scatter_plot,mp_title="scatter plot", f_names = f_names)
    #dp.heatmap(x, title= "Correlation matrix", f_names = f_names)
    #xfx = pd.DataFrame(x, index = np.reshape(fx,(len(fx))), columns = f_names)
    #dp.density_scatter(xfx)
    scenarios_list = [ (-1, -1, -1, -1)]
    set_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2, 1e-8 ,map_setters.set_mean_distance_map)
    from decision_tree_predictors import *
    from common_include import *
    from clustering import *
    from predictors import *
    DT_param = {
    'validator_compute': ['discrepancy_error'],
    'binarize':True,
    'title': 'Decision Tree'}
    scenarios = scenario_generator()
    scenarios.run_scenarios(scenarios_list,iris_data_generator(), DecisionTreeClassifier(set_kernel = set_kernel), data_accumulator(), **DT_param)
    f_z = DecisionTreeClassifier().get_proba(scenarios)
    fz = scenarios.accumulator.get_fzs()[0]
    dp = data_plots()
    dp.roc(fz, f_z, **DT_param)

    import tensorflow as tf
    tf_param = {'epochs': 10,
    'batch_size':16,
    'validation_split':0.1,
    'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'optimizer':tf.keras.optimizers.Adam(0.001),
    'activation':['relu',''],
    'layers':[128,3],
    'metrics':[tf.keras.metrics.SparseCategoricalAccuracy()],
    'get_proba':True,
    'validator_compute': ['discrepancy_error'],
    'binarize':True,
    'title': 'tensorflow'}
    scenarios = scenario_generator()
    scenarios.run_scenarios(scenarios_list,iris_data_generator(), tfClassifier(set_kernel = set_kernel), data_accumulator(), **tf_param)
    f_z = scenarios.accumulator.get_f_zs()[0]
    fz = scenarios.accumulator.get_fzs()[0]
    dp = data_plots()
    dp.roc(fz, f_z, **tf_param)

    torch_param = {'epochs': 128,
    'layers': [128],
    'batch_size': 16,
    'loss': nn.CrossEntropyLoss(),
    'activation': nn.ReLU(),
    'optimizer': torch.optim.Adam,
    "datatype": "long",
    "prediction": "labeled",
    "out_layer": 3,
    'get_proba':True,
    'validator_compute': ['discrepancy_error'],
    'binarize':True,
    'title': 'Pytorch'}

    scenarios = scenario_generator()
    scenarios.run_scenarios(scenarios_list,iris_data_generator(), PytorchClassifier(set_kernel = set_kernel), data_accumulator(), **torch_param)
    f_z = scenarios.accumulator.get_f_zs()[0]
    fz = scenarios.accumulator.get_fzs()[0]
    dp = data_plots()
    dp.roc(fz, f_z, **torch_param)

    codpy_param = {
    'get_proba':True,
    'validator_compute': ['discrepancy_error'],
    'binarize':True,
    'title': 'Codpy'}
    scenarios = scenario_generator()
    scenarios.run_scenarios(scenarios_list,iris_data_generator(), label_codpy_extrapolator(set_kernel = set_kernel), data_accumulator(), **codpy_param)
    f_z = scenarios.accumulator.get_f_zs()[0]
    fz = scenarios.accumulator.get_fzs()[0]
    dp = data_plots()
    dp.roc(fz, f_z, **codpy_param)




   