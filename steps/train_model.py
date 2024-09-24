#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegression
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

# In[ ]:


@step

def train_model(X_train:pd.DataFrame,X_test:pd.DataFrame,y_train:pd.Series,y_test:pd.Series,config:ModelNameConfig)->RegressorMixin:

    """
    Train the model on ingested data

    Args:
        X_train:pd.DataFrame,X_test:pd.DataFrame,y_train:pd.Series,y_test:pd.Series

    """
    try:
        model=None

        if config.model_name=="LinearRegression":
            model=LinearRegression().train(X_train,y_train)
            return model
        else:
            raise ValueError("Model {} not suppported".format(config.model_name))
    except Exception as e:
        logging.error("Error in the training model {}".format(e))
        raise e
