import dagshub
import mlflow

mlflow.set_tracking_uri('https://dagshub.com/pxxthik/MlOps-mini-project.mlflow')
dagshub.init(repo_owner='pxxthik', repo_name='MlOps-mini-project', mlflow=True)

with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)
