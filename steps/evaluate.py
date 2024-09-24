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
from zenml.client import Client
import mlflow

# In[ ]:
experiment_tracker=Client().active_stack.experiment_tracker
# log, monitor, and compare different experiments or model runs

@step(experiment_tracker=experiment_tracker.name)

def evaluate(model:RegressorMixin
             ,X_test:pd.DataFrame,
             y_test:pd.Series)->Tuple[
                 Annotated[float,"mse"],
                 Annotated[float,"rmse"],
                 Annotated[float,"r2"]
             ]:
    try:
        prediction=model.predict(X_test)
        mse_class=MSE()

        mse=mse_class.calculate_score(y_test,prediction)
        mlflow.log_metric("mse",mse)

        r2_class=R2()
        r2=r2_class.calculate_score(y_test,prediction)
        mlflow.log_metric("r2",r2)

        rmse_class=RMSE()
        rmse=rmse_class.calculate_score(y_test,prediction)
        mlflow.log_metric("rmse",rmse)

        return mse,rmse,r2
    except Exception as e:
        logging.error("Calc error {}".format(e))
        raise e

# %%
