import mlflow,mlflow.sklearn
def set_experiment(name):
    return mlflow.set_experiment(name)

def start_run(run_name=None):
    return mlflow.start_run(run_name=run_name)

def log_params_and_metrics(params=None, metrics=None):
    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)

def log_model(model, artifact_path="model"):
    mlflow.sklearn.log_model(model,name=artifact_path,serialization_format="cloudpickle")

def set_tags(tags):
    mlflow.set_tags(tags)

def log_artifact(file_path):
    mlflow.log_artifact(file_path)