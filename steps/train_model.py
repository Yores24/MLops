#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import pandas as pd
from zenml import step


# In[ ]:


@step

def train_model(df:pd.DataFrame)->None:
    """
    Train the model on ingested data

    Args:
        df:The ingested dataframe
    """
    pass
