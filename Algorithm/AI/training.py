import MetaTrader5 as mt
from torch import nn
import torch
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
from AI.transformers import TransformerModel  # Importar desde módulo local, no Hugging Face
from AI.helpers import train_model
import pandas as pd
from datetime import timedelta,datetime
import ast

class Standardizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def transform(self,x):
        normalized_x = (x - self.mu) / self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

def make_tensors(args):
    """fila,seq_length,scaler,name=args
    dia=fila.name.split('-')
    fila=fila.values
    #pos,action,atr_atrma=int(fila[2]),int(fila[0]),float(fila[3])
    pos,action,tp_largo,tp_corto=int(fila[2]),int(fila[0]),float(fila[3]),float(fila[4])
    data=pd.DataFrame(mt.copy_ticks_range(name, datetime(int(dia[0]),int(dia[1]),int(dia[2])),datetime(int(dia[0]),int(dia[1]),int(dia[2]))+timedelta(days=1), mt.COPY_TICKS_ALL))
    data=data['ask'].iloc[pos-seq_length-1:pos+1]
    #data=np.append(data.values,np.array([atr_atrma]))
    data=np.append(data.values,np.array([tp_largo,tp_corto],np.array([])))
    return torch.tensor(data),torch.tensor(action)"""

    fila,seq_length,scaler,name=args
    dia=fila.name.split('-')
    fila=fila.values
    action,entrada,tp_largo,tp_corto,data_5m=int(fila[0]),float(fila[1]),float(fila[3]),float(fila[4]),fila[5]
    data_5m=data_5m.replace('nan', '0')
    data_5m=ast.literal_eval(data_5m)
    data_5m=[int(x) if type(x)==str else x for x in data_5m]
    data_5m=data_5m+[entrada,tp_largo,tp_corto]
    data=np.array(data_5m)
    
    return torch.tensor(data),torch.tensor(action)

def train_test_dataloaders(data,nucleos,dir,seq_length=1000,pred_length=1,batch_size=32,train_size=0.7,shuffle=True,mode='create',name='EURUSD'):
    scaler=Standardizer()
        
    if mode=='create':
        train,test=[],[]

        for i in tqdm(range(len(data)),total=len(data)):
            if data.iloc[i]["Posición"]>seq_length:
                x,y=make_tensors((data.iloc[i],seq_length,scaler,name))
                train.append(x)
                test.append(y)

        train=np.array(train)

        #prices,atr=train[:,:-1],train[:,-1:]
        #prices = scaler.fit_transform(prices)
        #train = np.hstack([prices,atr])

        #train = scaler.fit_transform(train)

        np.save(dir[:-4]+'_train.npy',train,allow_pickle=False)
        np.save(dir[:-4]+'_test.npy',test,allow_pickle=False)

        training_data=TensorDataset(torch.from_numpy(train),torch.stack(test))

        #with open(f"{dir[:-4]}_scaler.pkl", 'wb') as f:
            #pickle.dump((scaler.mu,scaler.sd), f)

    elif mode=='load':
        train=np.load(dir[:-4]+'_train.npy')
        test=np.load(dir[:-4]+'_test.npy')

        training_data=TensorDataset(torch.tensor(train),torch.tensor(test))

        #with open(f"{dir[:-4]}_scaler.pkl", 'rb') as f:
            #pickle.load(f)

    train_size = int(train_size * len(training_data))
    test_size = len(training_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(training_data, [train_size, test_size]) #separo los datasets

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader,test_dataloader,scaler

def load_data(direc):
    # action=(0,1) para facilitar el trabajo de preprocesamiento, 0 (corto), 1 (largo)
    dataframe=pd.read_csv(direc)
    dataframe=dataframe.set_index(dataframe.columns[0])
    dataframe.index.names=['']
    return dataframe

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPOCHS=1000
    LR=1e-5
    name = "dataframe_GBPUSD_2023-01-01_2024-01-01"
    mt.initialize()

    print(f"Using {device} device\n")

    if device=='cuda':
        torch.cuda.empty_cache()
    
    dataframe=load_data(f"data/datasets/{name}.csv")

    try:
        train_dataloader,test_dataloader,scaler=train_test_dataloaders(dataframe,5,f'data/datasets/{name}.npy',mode='create',seq_length=10000,batch_size=32,shuffle=True,name="GBPUSD")
    except Exception as e:
        mt.shutdown()
        raise e
    
    print(next(iter(train_dataloader))[0].shape)
    print(next(iter(train_dataloader))[0])

    print(len(train_dataloader),len(test_dataloader))

    """model=TransformerModel(input_size=1,dec_seq_len=1,batch_first=True,dim_val=512,n_encoder_layers=4,n_heads=4,dropout_pos_enc=0.1,dropout_enc=0.1,out_predicted_features=1,max_seq_len=2000,option='no pre-enc',seq_len=1002)

    start=time.time()

    criterion=nn.MSELoss(reduction='mean')
    optimizer=model.configure_optimizers(learning_rate=LR,betas=(0.9, 0.95),weight_decay=0.1)

    train_model(model,train_dataloader,criterion,optimizer,num_epochs=NUM_EPOCHS,name="hola.pt",scheduler=False,forecast_window=1,enc_seq_len=1,grad_norm_clip=1.0)

    print('Tiempo ejecución: ',str(datetime.timedelta(seconds=time.time()-start)))

    if device=='cuda':
        torch.cuda.empty_cache()"""