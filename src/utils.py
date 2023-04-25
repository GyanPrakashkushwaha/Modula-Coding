# here I will write codes that will be basically used in entire project


import os 
import sys
from src.exception import CustomException
import dill

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV




def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    



def evaluate_models(X_train,y_train,X_test,y_test,models,param=None) -> dict:

    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            # train_model_score = r2_score(y_pred=y_train_pred,y_true=y_train)

            test_model_score = r2_score(y_true=y_test,y_pred=y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)

        


def load_object(file_path):
    try:
        with open(file_path,mode='rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)