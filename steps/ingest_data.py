#!/usr/bin/env python
# coding: utf-8

# In[2]:


import logging
import pandas as pd
from zenml import step


# In[3]:


class IngestData:
    """
    Ingesting the data from the data_path
    """
    def __init__(self,data_path:str) -> None:
        self.data_path=data_path
    
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path,index_col=0, parse_dates=True)
    


# In[4]:


@step
def ingest_data(data_path:str)->pd.DataFrame:
    """
    Ingesting data 
    """
    try:
        data=IngestData(data_path)
        df=data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data:e)")
        raise e
    


# In[ ]:




