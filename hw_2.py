from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

import mlflow
import pandas as pd

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository


models_dict = {
    "random_forest": RandomForestRegressor(),
    "linear_regression": LinearRegression(),
    "ada_boost": AdaBoostRegressor()
}
experiment_name = "Volzhenin_Nikolai"
TEST_SIZE=0.2

diabetes_data = load_diabetes(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(diabetes_data['data'], diabetes_data['target'])
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=TEST_SIZE)


try:
    mlflow.create_experiment(experiment_name)
    experiment_id = mlflow.set_experiment(experiment_name).experiment_id
except:
    experiment_id = mlflow.set_experiment(experiment_name).experiment_id

with mlflow.start_run(run_name=f"{experiment_name}_main", experiment_id=experiment_id, description="Main parent run") as parent_run:
    for model_key, model in models_dict.items():
        with mlflow.start_run(run_name=model_key, experiment_id=experiment_id, nested=True) as child_run:
            model.fit(pd.DataFrame(X_train), y_train)
            predictions = model.predict(X_val)
        
            evaluation_data = X_val.copy()
            evaluation_data["actual_target"] = y_val
        
            model_signature = infer_signature(diabetes_data['data'], predictions)
            model_metadata = mlflow.sklearn.log_model(model, model_key, signature=model_signature)
            
            mlflow.evaluate(
                model=model_metadata.model_uri,
                data=evaluation_data,
                targets="actual_target",
                model_type="regressor",
                evaluators=["default"]
            )
