import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args: 
            X_train: Trains data
            Y_train: Trains the labels
        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Trains data
            y_train: Trains the labels
        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error while training model".format(e))
            raise e