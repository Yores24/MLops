#!/usr/bin/env python
# coding: utf-8

# In[11]:


import sys
sys.path.append('../steps')


# In[14]:


from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate import evaluate


# In[17]:

"""
Enabling cache will lead to using the data from the previous runs if no
changes are observed
"""
# @pipeline
@pipeline(enable_cache=True)


def training_pipeline(data_path:str):
    df=ingest_data(data_path)
    clean_data(df)
    train_model(df)
    evaluate(df)


# In[ ]:




