#!/usr/bin/env python
# coding: utf-8

# In[2]:


import logging
from zenml import step
import pandas as pd


# In[ ]:


@step

def evaluate(df:pd.DataFrame)->None:
    pass

