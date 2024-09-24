import logging
from abc import ABC,abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score


class Evaluation(ABC):

    """
    """

    @abstractmethod

    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        """
        pass

class MSE(Evaluation):
    """
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse=mean_squared_error(y_true,y_pred)
            logging.info("Mean squared error : {}".format(mse))
            return mse
        except Exception as e:
            logging.error("The error in calc Mse is {}".format(e))
            raise e
        
class R2(Evaluation):

    """
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        try:
            logging.info("calculating R2")
            r2=r2_score(y_true,y_pred)
            logging.info("The R2 score is {}".format(r2))
            return r2
        except Exception as e:
            logging.error("The error in Calc R2 is {}".format(e))
            raise e
        
class RMSE(Evaluation):
    """
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Rmse")

            rmse=mean_squared_error(y_true,y_pred,squared=False)
            logging.info("The Rmse is {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("The error in rmse is {}".format(e))
            raise e
            