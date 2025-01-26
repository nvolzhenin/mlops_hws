import json
import pickle
import io
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Literal
import os

from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator

from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import mlflow
from mlflow.models import infer_signature

import pandas as pd
import numpy as np

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


BUCKET_NAME = Variable.get("S3_BUCKET")

TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"

DEFAULT_DAG_ARGS = {
    "owner": "Volzhenin_Nikolai",
    "retry": 3,
    "retry_delay": timedelta(minutes=1),
}

model_types = ["random_forest", "linear_regression", "decision_tree"]

model_map = {
    "random_forest": RandomForestRegressor(),
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(),
}

experiment_name = DEFAULT_DAG_ARGS['owner']

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)


def create_dag(dag_id: str, model_key: Literal["random_forest", "linear_regression", "decision_tree"]):

    def init(model_key: Literal["random_forest", "linear_regression", "decision_tree"]) -> Dict[str, Any]:
        start_time = datetime.now().strftime("%Y_%m_%d_%H")
        logger.info("Запуск обучения модели")

        configure_mlflow()

        try:
            mlflow.create_experiment(experiment_name)
            experiment_id = mlflow.set_experiment(experiment_name).experiment_id
        except:
            experiment_id = mlflow.set_experiment(experiment_name).experiment_id
        with mlflow.start_run(run_name=f"{experiment_name}_main", 
                              experiment_id=experiment_id, 
                              description="Main parent run",
        ) as parent_run:
            run_id = parent_run.info.run_id
        
        return {
            "init_timestamp_start": start_time,
            "run_id": run_id,
            "exp_id": experiment_id,
            "model_name": model_key
        }

    def download_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids="init")
        metrics['get_data_start_time'] = datetime.now().strftime("%Y_%m_%d_%H")
        logger.info("Начало загрузки данных")

        housing_data = fetch_california_housing(as_frame=True)
        data_frame = pd.concat([housing_data["data"], pd.DataFrame(housing_data["target"])], axis=1)

        s3_hook = S3Hook("s3_connection")
        file_buffer = io.BytesIO()
        data_frame.to_pickle(file_buffer)
        file_buffer.seek(0)

        s3_hook.load_file_obj(
            file_obj=file_buffer,
            key=f"Volzhenin_Nikolai_prj/{model_key}/datasets/california_housing.pkl",
            bucket_name=BUCKET_NAME,
            replace=True,
        )

        metrics['get_data_end_time'] = datetime.now().strftime("%Y_%m_%d_%H")
        metrics['data_shape'] = data_frame.shape
        logger.info("Загрузка данных завершена")

        return metrics

    def prepare_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids="download_data")
        metrics['prepare_data_start_time'] = datetime.now().strftime("%Y_%m_%d_%H")
        logger.info("Начало предобработки данных")

        s3_hook = S3Hook("s3_connection")
        file = s3_hook.download_file(
            key=f"Volzhenin_Nikolai_prj/{model_key}/datasets/california_housing.pkl", bucket_name=BUCKET_NAME
        )
        data = pd.read_pickle(file)

        X = data[FEATURES]
        y = data[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        scaler = StandardScaler()
        X_train_fitted = scaler.fit_transform(X_train)
        X_test_fitted = scaler.transform(X_test)

        s3_hook = S3Hook("s3_connection")
        for dataset_name, dataset in zip(
            ["X_train", "X_test", "y_train", "y_test"], [X_train_fitted, X_test_fitted, y_train, y_test]
        ):
            file_buffer = io.BytesIO()
            pickle.dump(dataset, file_buffer)
            file_buffer.seek(0)
            s3_hook.load_file_obj(
                file_obj=file_buffer,
                key=f"Volzhenin_Nikolai_prj/{model_key}/datasets/{dataset_name}.pkl",
                bucket_name=BUCKET_NAME,
                replace=True,
            )

        metrics['prepare_data_end_time'] = datetime.now().strftime("%Y_%m_%d_%H")
        metrics['features'] = FEATURES
        logger.info("Предобработка данных завершена")

        return metrics

    def train_model(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        parent_run_params = ti.xcom_pull(task_ids="init")
        run_id = parent_run_params['run_id']
        exp_id = parent_run_params['exp_id']
        logger.info("Начало обучения модели")
        configure_mlflow()

        with mlflow.start_run(run_name=model_key, experiment_id=exp_id,
                        nested=True, parent_run_id = run_id) as child_run:
            metrics = ti.xcom_pull(task_ids="prepare_data")
            metrics['train_model_start_time'] = datetime.now().strftime("%Y_%m_%d_%H")
            s3_hook = S3Hook("s3_connection")
            data = {}
            for name in ["X_train", "X_test", "y_train", "y_test"]:
                file = s3_hook.download_file(
                    key=f"Volzhenin_Nikolai_prj/{model_key}/datasets/{name}.pkl", bucket_name=BUCKET_NAME
                )
                data[name] = pd.read_pickle(file)

            model = model_map[model_key]
            model.fit(data["X_train"], data["y_train"])
            prediction = model.predict(data["X_test"])

            signature = infer_signature(data["X_test"], prediction)
            model_info = mlflow.sklearn.log_model(model, model_key, signature=signature)
            mlflow.evaluate(
                model=model_info.model_uri,
                data=data["X_test"].copy(),
                targets=np.array(data["y_test"]),
                model_type="regressor",
                evaluators=["default"],
            )

            metrics['train_model_end_time'] = datetime.now().strftime("%Y_%m_%d_%H")
            logger.info("Обучение завершено")

            return metrics

    def save_results(**kwargs) -> None:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids="train_model")

        s3_hook = S3Hook("s3_connection")
        file_buffer = io.BytesIO()
        file_buffer.write(json.dumps(metrics, ensure_ascii=False, indent=4).encode('utf-8'))
        file_buffer.seek(0)

        s3_hook.load_file_obj(
            file_obj=file_buffer,
            key=f"Volzhenin_Nikolai_prj/{model_key}/results/metrics.json",
            bucket_name=BUCKET_NAME,
            replace=True,
        )


    dag = DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",
        start_date=days_ago(2),
        catchup=False,
        tags=["mlops"],
        default_args=DEFAULT_DAG_ARGS
    )

    with dag:
        task_init = PythonOperator(task_id="init", 
                                   python_callable=init, 
                                   dag=dag, 
                                   op_kwargs={"model_key": model_key}, 
        )

        task_download_data = PythonOperator(task_id="download_data", 
                                            python_callable=download_data,
                                            dag=dag, 
                                            op_kwargs={"model_key": model_key},
        )

        task_prepare_data = PythonOperator(task_id="prepare_data", 
                                              python_callable=prepare_data, 
                                              dag=dag, 
                                              op_kwargs={"model_key": model_key},
        )

        task_train_model = PythonOperator(task_id="train_model", 
                                          python_callable=train_model, 
                                          dag=dag, 
                                          op_kwargs={"model_key": model_key}, 
        )
        
        task_save_results = PythonOperator(task_id="save_results", 
                                           python_callable=save_results, 
                                           dag=dag, 
                                           op_kwargs={"model_key": model_key},
        )

        task_init >> task_download_data >> task_prepare_data >> task_train_model >> task_save_results


for model_key in model_map.keys():
    create_dag(f"Volzhenin_Nikolai_prj_{model_key}", model_key)
