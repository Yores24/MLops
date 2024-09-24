#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[6]:


class DataStrategy(ABC):
    """
    Abstract class degining strategy for handling data
    """

    @abstractmethod
    def handle_data(self,data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass


# In[5]:


from pandas import DataFrame


class DataPreProcessing(DataStrategy):

    """
    Strategy to preprocess data
    """

    def handle_data(self, data: DataFrame) -> DataFrame:
        
        try:
            data=data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1
            )
            data['product_weight_g'].fillna(data['product_weight_g'].median(),inplace=True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(),inplace=True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(),inplace=True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(),inplace=True)
            data['review_comment_message'].fillna("No review",inplace=True)
            data=data.select_dtypes(include=[np.number])
            data=data.drop(
                [
                    "customer_zip_code_prefix",
                    "order_item_id",
                ]
            ,axis=1)
            return data
        except Exception as e:

            logging.error("error in the preprocessing of data: {}".format(e))
            raise e
            


# In[8]:


from pandas.core.api import DataFrame as DataFrame, Series as Series


class Datadivide(DataStrategy):

    """
Divide data in train and test
"""

    def handle_data(self, data: DataFrame) -> DataFrame | Series:
        try:
            X=data.drop(['review_score'],axis=1)
            y=data['review_score']
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logging.error("Error in dividing: {}".format(e))
            raise e
        


# In[9]:


class DataClean:

    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data=data
        self.strategy=strategy

    def handle_data(self)->Union[pd.DataFrame,pd.Series]:

        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data {}".format(e))
            raise e
        


# In[ ]:




