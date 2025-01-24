import json
import pickle
import io
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Literal

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

import pandas as pd

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


BUCKET_NAME = Variable.get("S3_BUCKET")

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

TARGET = 'MedHouseVal'


def create_dag(dag_id: str, model_key: Literal["random_forest", "linear_regression", "decision_tree"]):


    def init(**kwargs) -> Dict[str, Any]:
        start_time = datetime.now().strftime("%A, %D, %H:%M")
        logger.info("Запуск обучения модели")
        return {"start_time": start_time, "model_name": model_key}

    def download_data(**kwargs) -> Dict[str, Any]:
        start_time = datetime.now().strftime("%A, %D, %H:%M")
        logger.info(f"Начало загрузки данных")

        housing_data = fetch_california_housing(as_frame=True)
        data_frame = pd.concat([housing_data["data"], pd.DataFrame(housing_data["target"])], axis=1)

        s3_hook = S3Hook("s3_connection")
        file_buffer = io.BytesIO()
        data_frame.to_pickle(file_buffer)
        file_buffer.seek(0)

        s3_hook.load_file_obj(
            file_obj=file_buffer,
            key=f"Volzhenin_Nikolai/{model_key}/datasets/california_housing.pkl",
            bucket_name=BUCKET_NAME,
            replace=True,
        )

        end_time = datetime.now().strftime("%A, %D, %H:%M")
        logger.info(f"Загрузка данных завершена")

        return {"data_start_time": start_time, "data_end_time": end_time, "data_shape": data_frame.shape}

    def prepare_data(**kwargs) -> Dict[str, Any]:
        start_time = datetime.now().strftime("%A, %D, %H:%M")
        logger.info(f"Начало предобработки данных")

        s3_hook = S3Hook("s3_connection")
        file = s3_hook.download_file(
            key=f"Volzhenin_Nikolai/{model_key}/datasets/california_housing.pkl", bucket_name=BUCKET_NAME
        )
        data = pd.read_pickle(file)



        X = data[FEATURES]
        y = data[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        
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
                key=f"Volzhenin_Nikolai/{model_key}/datasets/{dataset_name}.pkl",
                bucket_name=BUCKET_NAME,
                replace=True,
            )

        end_time = datetime.now().strftime("%A, %D, %H:%M")
        logger.info(f"Предобработка данных завершена")

        return {"preprocess_start_time": start_time, "preprocess_end_time": end_time, "features": FEATURES, "target": TARGET}

    def train_model(**kwargs) -> Dict[str, Any]:
        start_time = datetime.now().strftime("%A, %D, %H:%M")
        logger.info(f"Начало обучения модели")

        s3_hook = S3Hook("s3_connection")
        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(
                key=f"Volzhenin_Nikolai/{model_key}/datasets/{name}.pkl", bucket_name=BUCKET_NAME
            )
            data[name] = pd.read_pickle(file)

        model = model_map[model_key]
        model.fit(data["X_train"], data["y_train"])
        predictions = model.predict(data["X_test"])

        metrics = {
            "r2_score": r2_score(data["y_test"], predictions),
            "mae": median_absolute_error(data["y_test"], predictions),
        }

        end_time = datetime.now().strftime("%A, %D, %H:%M")
        logger.info("Обучение завершено")

        return {"train_start_time": start_time, "train_end_time": end_time, "metrics": metrics}

    def save_results(**kwargs) -> None:
        # Извлекаем контекст данных из предыдущих задач
        ti = kwargs['ti']
        init_data = ti.xcom_pull(task_ids='initialize_task')
        download_data_result = ti.xcom_pull(task_ids='download_data')
        preprocess_data_result = ti.xcom_pull(task_ids='preprocess_data')
        train_results = ti.xcom_pull(task_ids='train_and_evaluate_model')

        all_results = [init_data, download_data_result, preprocess_data_result, train_results]

        # Сохраняем все результаты в JSON на S3
        s3_hook = S3Hook("s3_connection")
        file_buffer = io.BytesIO()
        file_buffer.write(json.dumps(all_results, ensure_ascii=False, indent=4).encode('utf-8'))
        file_buffer.seek(0)

        s3_hook.load_file_obj(
            file_obj=file_buffer,
            key=f"Volzhenin_Nikolai/{model_key}/results/metrics.json",
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
        task_init = PythonOperator(task_id="init", python_callable=init, dag=dag, templates_dict={"model_key": model_key}, provide_context=True)
        task_download_data = PythonOperator(task_id="download_data", python_callable=download_data, dag=dag, templates_dict={"model_key": model_key}, provide_context=True)
        task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag, templates_dict={"model_key": model_key}, provide_context=True)
        task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag, templates_dict={"model_key": model_key}, provide_context=True)
        task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag, templates_dict={"model_key": model_key}, provide_context=True)

        task_init >> task_download_data >> task_prepare_data >> task_train_model >> task_save_results


for model_key in model_map.keys():
    create_dag(f"Volzhenin_Nikolai_{model_key}", model_key)
