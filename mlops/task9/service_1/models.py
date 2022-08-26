import os
import mlflow

os.environ["MLFLOW_S3_ENDPOINT_URL"] = 'http://46.101.217.205:19001'
os.environ["MLFLOW_TRACKING_URI"] = 'http://46.101.217.205:5900'
os.environ["AWS_ACCESS_KEY_ID"] = 'IAM_ACCESS_KEY'
os.environ["AWS_SECRET_ACCESS_KEY"] = 'IAM_SECRET_KEY'


MODEL_FLOAT_PATH = "models:/iris_sklearn/production"
MODEL_STRING_PATH = "models:/iris_pyfunc/production"


model_float = mlflow.sklearn.load_model(MODEL_FLOAT_PATH)
model_string = mlflow.pyfunc.load_model(MODEL_STRING_PATH)
