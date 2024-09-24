#!/usr/bin/env python
# coding: utf-8

# In[2]:


import logging
from zenml import step
import pandas as pd
from src.evaluation import MSE,R2,RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

# In[ ]:


@step

def evaluate(model:RegressorMixin
             ,X_test:pd.DataFrame,
             y_test:pd.DataFrame)->Tuple[
                 Annotated[float,"mse"],
                 Annotated[float,"rmse"],
                 Annotated[float,"r2"]
             ]:
    prediction=model.predict(X_test)
    mse_class=MSE()
    mse=mse_class.calculate_score(y_test,prediction)

    r2_class=R2()
    r2=r2_class.calculate_score(y_test,prediction)

    rmse_class=RMSE()
    rmse=rmse_class.calculate_score(y_test,prediction)
    return mse,rmse,r2

