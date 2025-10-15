from datetime import datetime
import io
import boto3
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from botocore.client import Config
import os

BUCKET_NAME = "airflow-data"
OBJECT_KEY = "Titanic-Dataset.csv"
TABLE_NAME = "titanic_data"
POSTGRES_CONN_ID = "my_postgres_connection"

def load_csv_to_postgres():
    # --- 1️⃣ Connect to MinIO ---
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(s3={"addressing_style": "path"})
    )

    # --- 2️⃣ Read the CSV ---
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=OBJECT_KEY)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    # --- 3️⃣ Load to PostgreSQL ---
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    engine = pg_hook.get_sqlalchemy_engine()

    # Create the table if it doesn’t exist, replace or append
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
    print(f"✅ Loaded {len(df)} rows into '{TABLE_NAME}' table.")

with DAG(
    dag_id="load_titanic_from_minio",
    start_date=datetime(2024, 1, 1),
    schedule="@once",
    catchup=False,
    default_args={"owner": "data_eng"},
    description="Load Titanic dataset from MinIO to Postgres"
) as dag:
    
    load_task = PythonOperator(
        task_id="load_csv_to_postgres",
        python_callable=load_csv_to_postgres
    )
