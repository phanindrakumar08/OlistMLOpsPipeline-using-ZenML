import logging

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            data.drop(['order_approved_at',
                        'order_delivered_carrier_date',
                        'order_delivered_customer_date',
                        'order_estimated_delivery_date',
                        'order_purchase_timestamp'], axis = 1, inplace=True)
            data['review_comment_message'].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])

            cols_to_drop = ['customer_zip_code_prefix', 'order_item_id']
            data.drop(cols_to_drop, axis=1, inplace=True)
            return data 
        except Exception as e:
            logging.error('Error in Preprocessing data' .format(e))
            raise e
        

class DataDivideStrategy(DataStrategy):
    """
    Strategy to divide the data
    """
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Method to divide data
        """
        try:
            X = data.drop('review_score', axis=1)
            y = data['review_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error('Error in Dividing data' .format(e))
            raise e
        

class DataCleaning:
    """
    Class for cleaning data which preprocess the data and divides the data into train and test
    """
    def __init__(self, data:pd.DataFrame, strategy: DataStrategy):
        self.data = data 
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data" .format(e))
            raise e