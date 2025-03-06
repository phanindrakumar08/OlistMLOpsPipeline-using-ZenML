import logging
 
from sklearn.base import RegressorMixin
import pandas as pd
from zenml import step
from src.evaluate import MSE, RMSE, R2
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin, 
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[
                       Annotated[float, "r2_score"], 
                       Annotated[float, "rmse"]
                   ]:
    
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)
        print(f"MSE: {mse}")

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2", r2)
        print(f"R2 score: {r2}")

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        print(f"RMSE: {rmse}")


        return r2, rmse
    
    except Exception as e:
        logging.error('Error while calculating scores'.format(e))
        raise e