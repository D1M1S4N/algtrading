import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import MetaTrader5 as mt
from datetime import timedelta,datetime
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import multiprocessing

class Standardizer():
    def __init__(self):
        self.mu = None
        self.sd = None

        #scaler = Normalizer()
        #data_=scaler.fit_transform(data.values[:,:1])
        #print(data_[0])
        #data__=scaler.inverse_transform(data_)
        #print(data__[1])

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu
    
def raw_data_to_df(raw_data):
    index=[]
    data_rows=[]
    for i,j in raw_data.items():
        for _ in range(len(j)):
            index.append(i)
            data_rows.append(j[_])
    
    dataframe=pd.DataFrame(data_rows,index=index,columns=['Sentido','Entrada','Posición','TP largo','TP corto','Data5m'])
    return dataframe

def load_data(direc):
    # action=(0,1) para facilitar el trabajo de preprocesamiento, 0 (corto), 1 (largo)
    dataframe=pd.read_csv(direc)
    dataframe=dataframe.set_index(dataframe.columns[0])
    dataframe.index.names=['']
    return dataframe

def make_tensors(args):
    fila,seq_length,scaler=args
    dia=fila.name.split('-')
    fila=fila.values
    pos,action=int(fila[2]),int(fila[0])
    data=pd.DataFrame(mt.copy_ticks_range("EURUSD", datetime(int(dia[0]),int(dia[1]),int(dia[2])),datetime(int(dia[0]),int(dia[1]),int(dia[2]))+timedelta(days=1), mt.COPY_TICKS_ALL))
    data = scaler.fit_transform(data['ask'][pos-seq_length-1:pos+1].values)
    data=scaler.fit_transform(data)
    return torch.tensor(data),torch.tensor(action)

def train_test_dataloaders(data,nucleos,dir,seq_length=1000,pred_length=1,batch_size=4,train_size=0.7,shuffle=True,mode='create'):
    scaler=Standardizer()
        
    if mode=='create':
        train,test=[],[]

        for i in tqdm(range(len(data)),total=len(data)):
            x,y=make_tensors((data.iloc[i],seq_length,scaler))
            train.append(x)
            test.append(y)

        np.save(dir[:-4]+'_train.npy',train,allow_pickle=False)
        np.save(dir[:-4]+'_test.npy',test,allow_pickle=False)

        training_data=TensorDataset(torch.stack(train),torch.stack(test))

    elif mode=='load':
        train=np.load(dir[:-4]+'_train.npy')
        test=np.load(dir[:-4]+'_test.npy')

        training_data=TensorDataset(torch.tensor(train),torch.tensor(test))

    train_size = int(train_size * len(training_data))
    test_size = len(training_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(training_data, [train_size, test_size]) #separo los datasets

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader,test_dataloader

if __name__ == '__main__':
    import time
    #------Cuda/Cpu setUp------#
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device\n")

    #---------Mt5 Init--------#
    mt.initialize()
    if not mt.initialize():
        raise ValueError(f'initialize() failed, error code = {mt.last_error()}')
    
    mt.login("86492341",password="XrOl*1Cq",server="MetaQuotes-Demo")

    #----------Data----------#
    name="EURUSD"
    dataframe1=load_data(f"data/datasets/dataframe_{name}_2020-01-01_2024-01-01.csv")
    dataframe2=load_data(f"data/datasets/dataframe_{name}_2024-01-01_2024-12-24.csv")
    #dataframe3=load_data("data/datasets/dataframe_2022-01-01_2023-01-01.csv")
    #dataframe4=load_data("data/datasets/dataframe_2023-01-01_2024-01-01.csv")

    resultado = pd.concat([dataframe1,dataframe2])
    #resultado = pd.concat([resultado,dataframe3])
    #resultado = pd.concat([resultado,dataframe4])

    assert(len(resultado)==(len(dataframe1)+len(dataframe2)))

    resultado.to_csv(f"data/datasets/dataframe_{name}_2020-01-01_2024-12-24.csv")

    #train_dataloader,test_dataloader=train_test_dataloaders(dataframe,5,'data/datasets/dataframe_2023-01-01_2024-01-01.npy',mode='create')