#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing import Annotated
from src.data_clean import DataClean,Datadivide,DataPreProcessing
# In[ ]:


@step

def clean_data(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],
]:
    """
    Clean the data and divide it into train and test
    Args:
        df:Raw_data
    Returns:
        X_train:Training data
        X_test:Testing data
        y_train:Training Lable
        y_test:Teasting Label
    """
    try:
        proc_strategy=DataPreProcessing()
        dataclean=DataClean(df,proc_strategy)
        proc_data=dataclean.handle_data()

        divide_strat=Datadivide()
        datadivide=DataClean(proc_data,divide_strat)
        X_train,X_test,y_train,y_test=datadivide.handle_data()
        logging.info("Data Cleaning dividing complete")
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error("Error in cleaning and divide {}".format(e))
        raise e
    
    
    

