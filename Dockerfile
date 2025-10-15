FROM astrocrpublic.azurecr.io/runtime:3.1-2

RUN pip install apache-airflow-providers-amazon  apache-airflow-providers-postgres boto3 minio
