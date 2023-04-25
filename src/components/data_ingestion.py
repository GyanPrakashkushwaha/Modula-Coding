# 2nd step

""" Reading data from a database is called data ingestion and 
  this data ingestion is part of a modul
    this will have all the codes that are related to reading data"""

import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer




@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # accessing DataIngestionConfig Class

    def initiate_data_ingestion(self):
        
        logging.info('Entered the data ingestion method Or component')

        try:
            df = pd.read_csv('notebook\data\stud.csv') # reading data
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # os.path.dirname() method is used to get the directory name from the specified path.
            # making artifacts folder folder 

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            # saving data in csv file in artifacts folder in data.csv

            logging.info('train test split initiated')

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            # dividing df in train and test sets

            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header= True)
            # saving train dataset in artifacts folder in train.csv file

            test_set.to_csv(self.ingestion_config.test_data_path,index= False,header = True)
            # saving test dataset in artifacts folder in test.csv file

            logging.info('Ingestion of the data is completed')
            
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        

        except Exception as e:
            logging.info(e,sys)
            raise CustomException(e,sys)



if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modelTrainer = ModelTrainer()

    print(f'my best model and r2_score is - {modelTrainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr)}') 
    logging.info(f'my best model and r2_score is - {modelTrainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr)} ')

