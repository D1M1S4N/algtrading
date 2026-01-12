import torch
import sys
import os
import numpy as np
from lightgbm import LGBMClassifier,log_evaluation,early_stopping
import lightgbm as lgb
import pickle
from sklearn.metrics import accuracy_score
import gc
import traceback

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

os.environ['LIGHTGBM_VERBOSE'] = '-1'

sys.path.append(os.path.abspath('../Algorithm/AI'))

from AI.training import *

def train_model(X_train, y_train,X_test,y_test,name,use_gpu,all=False):
    """
    Entrena un modelo LGBM.

    Args:
        X_train (array-like): Características de entrenamiento.
        y_train (array-like): Etiquetas de entrenamiento.
        model_type (str): 'classification' o 'regression'.

    Returns:
        model: Modelo entrenado.
    """
    
    params = {
        'random_state': 42,
        'n_estimators': 200, #n_estimators=600 for 2020-2024 # n_estimators=200*(years-1)
        'max_depth': -1,
        'num_leaves': 63,
        'learning_rate': 0.1,
        'max_bin': 255,
        'metric': 'binary_error',  # Cambia 'l2' a una métrica de clasificación
        'objective': 'binary',
        'device': 'gpu' if use_gpu else 'cpu'
    }
    
    if not all:
        dtrain = lgb.Dataset(X_train, label=y_train)
        dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

        """model.fit(
            X_train, y_train,
            eval_set=[(X_test,y_test)],
            eval_metric='l2',
            callbacks=[log_evaluation(1),early_stopping(100, verbose = False)]
        )"""
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,  # Ajustar según sea necesario
            valid_sets=[dtest],
            valid_names=['eval'],  # Nombre de los datasets de validación
            callbacks=[
                log_evaluation(1),  # Callback para registrar la evaluación
                early_stopping(100,verbose=False)  # Callback para early stopping
            ]
        )
    else:
        all_dataset=np.concatenate((X_train, X_test), axis=0),np.concatenate((y_train, y_test), axis=0)
        dtrain = lgb.Dataset(all_dataset[0], label=all_dataset[1])
        """model.fit(
            np.concatenate((X_train, X_test), axis=0),np.concatenate((y_train, y_test), axis=0),
            callbacks=[log_evaluation(1)]
        )"""
        model = lgb.train(params, dtrain, num_boost_round=1000,callbacks=[log_evaluation(1)])

    if name:
        with open(f"data/modelos/{name}.pkl", 'wb') as f:
            pickle.dump(model, f)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evalúa un modelo entrenado.

    Args:
        model: Modelo entrenado.
        X_test (array-like): Características de prueba.
        y_test (array-like): Etiquetas de prueba.
        model_type (str): 'classification' o 'regression'.

    Returns:
        metric: Métrica de desempeño (accuracy o RMSE).
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def dataloader_to_numpy(dataloader):
    """
    Convierte un DataLoader de PyTorch a arrays de NumPy.

    Args:
        dataloader (DataLoader): DataLoader con datos.

    Returns:
        X (np.ndarray): Características extraídas.
        y (np.ndarray): Etiquetas extraídas.
    """
    X_list, y_list = [], []
    for batch in dataloader:
        X, y = batch
        X_list.append(X.numpy())  # Convertir tensores a NumPy
        y_list.append(y.numpy())
    X = np.vstack(X_list)  # Combina todas las filas
    y = np.hstack(y_list)  # Combina todas las etiquetas
    return X, y

class MarketMasterAI:
    def __init__(self):
        #name1='dataframe_EURUSD_2020-01-01_2024-01-01'
        name1='dataframe_EURUSD_2023-01-01_2024-01-01'
        name2='dataframe_USDSEK_2023-01-01_2024-01-01'
        name3='dataframe_GBPUSD_2023-01-01_2024-01-01'
        name4='dataframe_USDJPY_2023-01-01_2024-01-01'
        names={"EURUSD":name1,"USDSEK":name2,"GBPUSD":name3,"USDJPY":name4}
        self.models={}
        self.scalers={}

        for dict_name,name in names.items():
            with open(f"data/modelos/{name}.pkl", 'rb') as f:
                self.models[dict_name]=pickle.load(f)

            #with open(f"data/datasets/{name}_scaler.pkl", 'rb') as f:
                #params = pickle.load(f)
                scaler=Standardizer()
                #scaler.mu=params[0]
                #scaler.sd=params[1]
                self.scalers[dict_name]=scaler

        self.input_size = self.models["EURUSD"].num_feature()

    def run_confirmation(self,args,action,action2,name2,lotaje,peso):
        """
        Ejecuta el modelo.

        Args:
            X (np.array): datos necesarios para la prediccion.

        Returns:
            action=(0,1) para facilitar el trabajo de preprocesamiento, 0 (corto), 1 (largo)
        """
        pos_dia=args["pos_dia"]
        name=args['name']

        fecha = args['actual_time']

        data_5m = args[args["temporality"]][name]
        data_5m = data_5m.drop(columns=["cv_5m", "cv_ma_5m","impulso","all_candle_patterns"])
        data_5m = data_5m[data_5m["time"]<=fecha].iloc[-1]
        data = args["all_data"][name]
        X=data
        data = data[data["time"]<=fecha].iloc[-1]

        atr_ma=data_5m["atr_ma"]
        atr=data_5m["atr"]
        bid=data['bid']
        ask=data['ask']
        spread = ask-bid

        tp_multiplier=3 # cambiar esto

        try:
            #X=X['ask'].iloc[pos_dia-10000-1:pos_dia+1].to_numpy()
            tp_largo=bid+(atr*(atr/atr_ma)*tp_multiplier)+spread
            tp_corto=ask-(atr*(atr/atr_ma)*tp_multiplier)-spread

            #X = np.append(X,np.array([tp_largo,tp_corto]))

            X=list(data_5m)[1:]
            X=X+[ask,tp_largo,tp_corto]
            X = np.array(X)[None, :]

            #X=self.scalers[name].transform(X)
            #X=np.append(X,np.array([atr_atrma]))

            pred=self.models[name].predict(X)[0]
            
            #pred = (pred > 0.5)
            if pred<0.3333:
                pred=-1
            elif 0.3333<=pred<=0.6666:
                pred=0
            else:
                pred=1
                
            if pred==-1 and action==-1: #corto y corto
                lotaje=lotaje*peso
            elif pred==1 and action==1: #largo y largo
                lotaje=lotaje*peso

        except Exception as e:
            print(e) 
            traceback.print_exc()               
            pass

        return lotaje
        #return 0

    def run(self,args,indicador_valor):
        """
        Ejecuta el modelo.

        Args:
            X (np.array): datos necesarios para la prediccion.

        Returns:
            action=(0,1) para facilitar el trabajo de preprocesamiento, 0 (corto), 1 (largo)
        """
        pos_dia=args["pos_dia"]
        name=indicador_valor
        action=0

        fecha = args['actual_time']

        data_5m = args[args["temporality"]][name]
        data_5m = data_5m.drop(columns=["cv_5m", "cv_ma_5m","impulso","all_candle_patterns"])
        data_5m = data_5m[data_5m["time"]<=fecha].iloc[-1]
        data = args["all_data"][name]
        X=data
        data = data[data["time"]<=fecha].iloc[-1]

        atr_ma=data_5m["atr_ma"]
        atr=data_5m["atr"]
        bid=data['bid']
        ask=data['ask']
        spread = ask-bid

        tp_multiplier=3 # cambiar esto

        try:
            #X=X['ask'].iloc[pos_dia-10000-1:pos_dia+1].to_numpy()
            tp_largo=bid+(atr*(atr/atr_ma)*tp_multiplier)+spread
            tp_corto=ask-(atr*(atr/atr_ma)*tp_multiplier)-spread

            #X = np.append(X,np.array([tp_largo,tp_corto]))

            X=list(data_5m)[1:]
            X=X+[ask,tp_largo,tp_corto]
            X = np.array(X)[None, :]

            #X=self.scalers[name].transform(X)
            #X=np.append(X,np.array([atr_atrma]))

            pred=self.models[name].predict(X)[0]
            
            """pred = (pred > 0.5)

            if pred==0:
                action=-1 #corto

            elif pred==1:
                action=1 #largo"""

            
            if pred < 0.33:
                action = -1
            elif pred < 0.66:
                action = 0
            else:
                action = 1

        except Exception as e:
            print(e) 
            traceback.print_exc()               
            pass

        return action

# Ejemplo de uso con datos ficticios
if __name__ == "__main__":
    # Crear datos ficticios para clasificación
    np.random.seed(42)
    device=True if torch.cuda.is_available() else False
    if device:
        torch.cuda.empty_cache()
        gc.collect()

    name='dataframe_USDJPY_2023-01-01_2024-01-01'

    dataframe=load_data(f"data/datasets/{name}.csv")
    train_dataloader,test_dataloader,scaler=train_test_dataloaders(dataframe,5,f"data/datasets/{name}.npy",mode='load')

    #x_train,y_train=next(iter(train_dataloader))
    #print(x_train)
    #print(y_train)
    #print(x_train.shape)
    #print(y_train.shape)

    X_train, y_train = dataloader_to_numpy(train_dataloader)
    X_test, y_test = dataloader_to_numpy(test_dataloader)

    # Entrenar el modelo
    model = train_model(X_train, y_train,X_test,y_test,name=name,use_gpu=device,all=True)

    with open(f"data/modelos/{name}.pkl", 'rb') as f:
        model = pickle.load(f)

    # Evaluar el modelo
    preds=model.predict(X_test)
    #accuracy = evaluate_model(model, X_test, y_test)
    y_pred_binary = (preds > 0.5)  # Umbral de 0.5 para clasificación binaria
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Accuracy: {accuracy}")
