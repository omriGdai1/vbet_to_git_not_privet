from celery import Celery
import psycopg2
import pandas as pd
import boto3
import pickle
from datetime import datetime
from utils import get_secret

# broker = 'pyamqp://guest@localhost//'  # local host
# # # broker = 'pyamqp://guest@host.docker.internal//'  # Update the broker URL
# celery_app = Celery('tasks', broker=broker)

broker = "amqps://GDAI_broker:matilda141414@b-dc80107f-7e65-4fee-918a-95517d71b7e8.mq.eu-north-1.amazonaws.com:5671"
celery_app = Celery('tasks', broker=broker)


@celery_app.task
def add(x, y):
    print(x + y)
    return x + y


@celery_app.task
def duplicate(x, y):
    print(x * y)
    return x * y


@celery_app.task
def execute_query_and_save_to_s3(query):
    secret = get_secret()
    secret_username = secret["username"]
    secret_password = secret["password"]
    # PostgreSQL database connection details
    db_host = "demo-db.ctk9oi0waozt.eu-north-1.rds.amazonaws.com"
    db_port = 5432
    db_user = secret_username
    db_password = secret_password
    db_name = "postgres"  # Replace with your actual database name

    # AWS S3 bucket details
    s3_bucket = 'omritestjune'

    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )

    # Execute the query and retrieve the DataFrame
    df = pd.read_sql(query, conn)

    # Generate a timestamp-based filename for the pickled DataFrame
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'dataframe_{timestamp}.pkl'

    # Pickle the DataFrame
    pickled_data = pickle.dumps(df)

    # Upload the pickled DataFrame to the AWS S3 bucket
    s3_client = boto3.client('s3')  # , aws_access_key_id="AKIA443OBNCFNSPKO3HH",
    # aws_secret_access_key="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")

    s3_client.put_object(Body=pickled_data, Bucket=s3_bucket, Key=filename)

    # Close the database connection
    conn.close()

    print(f"DataFrame pickled and uploaded to '{s3_bucket}/{filename}' successfully.")


@celery_app.task
def Vbet_pre_process():
    pass


