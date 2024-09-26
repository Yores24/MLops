import json

# from .utils import get_data_for_test
import os

import numpy as np
import pandas as pd
# from materializer.custom_materializer import cs_materializer
from steps.clean_data import clean_data
from steps.evaluate import evaluate
from steps.ingest_data import ingest_data
from steps.train_model import train_model
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

# from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])
import pandas as pd

class DeploymentTrigger(BaseParameters):
    """Deployment Trigger Config"""
    min_accuracy:float=0.5

@step

def deployment_trigger(accuracy:float,config:DeploymentTrigger,):
    return accuracy>config.min_accuracy


@pipeline(enable_cache=True,settings={"docker":docker_settings})

def continuous_deployment_pipeline(
    data_path:str,
    min_accuracy:float=0.5,
    workers:int=1,
    timeout:int=DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df=ingest_data(data_path)
    X_train,X_test,y_train,y_test=clean_data(df)
    model=train_model(X_train,X_test,y_train,y_test)
    mse,rmse,r2=evaluate(model,X_test,y_test)
    deployment_dec=deployment_trigger(r2)

    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_dec,
        workers=workers,
        timeout=timeout,
    )