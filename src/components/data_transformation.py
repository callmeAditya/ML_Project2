import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.utils import save_object

@dataclass ## decorators to use class variables directly without constructor
class DataTransformationConfig:
    preprocess_obj_path: str = os.path.join("artifacts", "preprocess.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            num_features = ['reading_score', 'writing_score']
            categorical_features =['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
       'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), ### taking mode
                    ('encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("categorical columns encoding done")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, num_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )
            
            
            return preprocessor
            
        except Exception as e:
            logging.error("Data Transformation is failed")
            raise MyException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            preprocessor = self.get_data_transformer_object()
            
            target_column_name = 'math_score'
            
            
            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_test_df = test_df[target_column_name]
            
            logging.info("preprocessing object on training df and testing df")
            
            
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocess_obj_path), exist_ok=True)
            pd.to_pickle(preprocessor, self.data_transformation_config.preprocess_obj_path)
            logging.info("read train and test data")
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr =  preprocessor.transform(input_feature_test_df)
            
            train_arr = np.concatenate((input_feature_train_arr, target_train_df.values.reshape(-1,1)), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, target_test_df.values.reshape(-1,1)), axis=1)
            
            save_object(
                self.data_transformation_config.preprocess_obj_path,
                preprocessor
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_path
            )
        
        except Exception as e:
            logging.error("Data Transformation is failed")
            raise MyException(e, sys)
        
if __name__ == "__main__":
    
    data_inge = DataIngestion()
    train_path, test_path = data_inge.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path, test_path)