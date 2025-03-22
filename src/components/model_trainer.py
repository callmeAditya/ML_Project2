## confusion matrix, and model trainer
import os
import sys
import numpy as np
import pandas as pd


from catboost import CatBoostClassifier
from sklearn.ensemble import(
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)

from sklearn.neighbors import KNeighborsClassifier
from dataclasses import dataclass
# from sklearn.linear import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.exception import MyException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    
    def initiate_modelTrainer(self, trainarr, testarr):
        try:
            logging.info("model training initiated")
            X_train,y_train,X_test,y_test=(
                trainarr[:,:-1],
                trainarr[:,-1],
                testarr[:,:-1],
                testarr[:,-1]
            )
            
            # Adjust target variable to match expected classes
            # label_encoder = OneHotEncoder()
            # y_train = label_encoder.fit_transform(y_train)
            # y_test = label_encoder.transform(y_test)
            
            models ={
                "Linear_Regression": LinearRegression(),
                "Decision_Tree": DecisionTreeClassifier(),
                "Random_Forest": RandomForestClassifier(),
                "CatBoostClassifier": CatBoostClassifier(verbose=False),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                # "KNeighborsClassifier": KNeighborsClassifier()                
            }
            
            params={
                "Decision_Tree": {
                    'criterion':['log_loss', 'entropy', 'gini'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random_Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingClassifier":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear_Regression":{},
                "CatBoostClassifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostClassifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # y_train = np.array(y_train, dtype=int)
            # y_test = np.array(y_test, dtype=int)
            
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)
        
            print(model_report)
            
            
            logging.info(f"model report: {model_report}")
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.5:
                raise MyException("No best model found")
            
         
            logging.info(f"best model found: {best_model_name} with score: {best_model_score}")
            
            save_object(
                path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info("model training completed")
            logging.info("saving model")
            
            
            pred = best_model.predict(X_test)
            r2score = r2_score(y_test, pred)
            logging.info(f"model score: {r2score}")
            return r2score
        
        except Exception as e:
            logging.error("Model training failed")
            raise MyException(e, sys)