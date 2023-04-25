# 4th step

# After tranforming data I will do train my data in this file


import os
import sys
from dataclasses import dataclass

# from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import ( save_object,
                       evaluate_models
)


@dataclass
class ModelTrainerConfig:
    tranined_model_file_path = os.path.join('artifacts','model.pkl') # for saving in pickle file


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() # whenever this class would be called ModelTrainerConfig should be initialized 

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('spliting training and test array')
            """ spliting test and train from both independent and dependent features"""
            X_train,y_train,X_test,y_test = ( 

                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            """ All Algorithms for training model"""
        
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Gradient Boosting' : GradientBoostingRegressor(),
                'Linear Regression' : LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'XGBRegressor' : XGBRegressor(),
                'AdaBoost Regressor' : AdaBoostRegressor()
            }

            """ this evaluate models function is in utils and this is for training model with all algorithms , and this will give report of highest accuracy of model"""
            model_report= evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            

            """to get best model from dictionary"""
            best_model_score = max(sorted(model_report.values()))

            
            if best_model_score < 0.6:
                raise CustomException('No best model found')


            """to get best model name from dictionary"""
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            logging.info(f'{best_model} Best found model on both training and testing dataset')

            save_object(
                file_path=self.model_trainer_config.tranined_model_file_path,
                obj=best_model
            )


            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_true=y_test,y_pred=predicted)

            return(
                r2_square,
                best_model
            )


        except Exception as e:
            raise CustomException(e,sys)