from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

BUCKET = "airflow-minio-smoke"
KEY = "ping.txt"
CONTENT = b"hello from airflow -> minio\n"

def write_obj(**_):
    # If you created AIRFLOW_CONN_MINIO_S3, use S3Hook("minio_s3"); otherwise plain S3Hook() uses env vars
    hook = S3Hook(aws_conn_id="minio_s3") if False else S3Hook()
    try:
        if not hook.check_for_bucket(BUCKET):
            hook.create_bucket(bucket_name=BUCKET)
    except Exception:
        pass  # bucket may already exist
    hook.load_bytes(CONTENT, key=KEY, bucket_name=BUCKET, replace=True)

def read_obj(**_):
    hook = S3Hook()
    data = hook.read_key(key=KEY, bucket_name=BUCKET)
    print("READ FROM MINIO:\n", data)

with DAG(
    dag_id="minio_smoke_test",
    start_date=datetime(2024,1,1),
    schedule="@once",
    catchup=False,
) as dag:
    put = PythonOperator(task_id="put", python_callable=write_obj)
    get = PythonOperator(task_id="get", python_callable=read_obj)
    put >> get