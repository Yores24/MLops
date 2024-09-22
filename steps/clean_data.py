#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import pandas as pd
from zenml import step


# In[ ]:


@step

def clean_data(df:pd.DataFrame)->None:
    pass
    

