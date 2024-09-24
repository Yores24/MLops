import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
class Model(ABC):
    
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self,X_train,y_train):
        """
        Train the model
        Args:
            X_train: Training Data
            y_train: Training labels
        Returns:
            None
        """
        pass

class LinearRegression(Model):

    def train(self, X_train, y_train):
        """
        Train the model
        Args:
            X_train: Training Data
            y_train: Training labels
        Returns:
            None
        """
        try:
            mod=LinearRegression()
            mod.fit(X_train,y_train)
            logging.info("Model Training Completed")
            return mod
        except Exception as e:
            logging.error("An error occured : {}".format(e))
            raise e
        
# You can add other models also here with the train functionality