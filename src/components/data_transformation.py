# 3rd step
# After data ingestion we may have to transform the data so to transform the data se will use this file


import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.utils import save_object



@dataclass
class DataTranformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) :
        self.data_transformation_config= DataTranformationConfig()

    def get_data_transformer_obj(self): # this is for data transformation
        
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                ('impute', SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )


            cat_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
                ('Scaler',StandardScaler())
                ]
            )        

            logging.info(f'Numerical Columns : {numerical_columns}')
            logging.info(f'Categorical Columns : {categorical_columns}')


            preprocessor = ColumnTransformer(
                transformers=[
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    # Starting data transformation technique
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data completed')

            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformer_obj()

            target_col_name = 'math_socre'
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # np.c_ this concatenate like np.concat function but its axis is by default '1'

            logging.info(f"Saved preprocessing object.")

            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            # this is for saving pickle


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
            

            
        except:
            pass


