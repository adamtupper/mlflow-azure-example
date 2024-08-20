# Track Experiments on Azure with MLflow

A minimal example of using MLflow to track experiments on Microsoft Azure.

## Setup Development Environment

To run this example you need to have setup the Azure Machine Learning CLI. Instructions on how to do this can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public)

Furthermore, you need to have installed the Python dependencies listed in `requirements.txt`. It is reccomended that you do this in an isolated virtual environment (using virtualenv, Anaconda, or Poetry, etc.).

## Running Locally

Instead of using an Azure Machine Learning workspace as an MLflow server, you can run this example locally by spinning up a local MLFlow server:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

and setting the the `mlflow_tracking_uri` as follows:

```bash
python src/ml/train.py  \
    dataset_dir=/absolute/path/to/dataset/directory \
    mlflow_tracking_uri=http://127.0.0.1:8080 \
    mlflow_experiment=retina_mnist \
    fast_dev_run=True
```

Note that enabling `fast_dev_run` mode will run only a single training and validation batch. After verifying that everything's working, you can re-run the above without setting the `fast_dev_run` flag to start a full training run. At this point, you'll be able to track the progress via the MLServer by pointing your web browser to `http://127.0.0.1:8080`.

For more details see [Starting the MLflow Tracking Server](https://mlflow.org/docs/latest/getting-started/logging-first-model/step1-tracking-server.html#starting-the-mlflow-tracking-server).

## Running with an Azure Backend

Instead of running an MLFlow server locally, you can use an Azure ML workspace as the MLFlow backend by instead setting the `mlflow_tracking_uri` as follows:

```bash
python src/ml/train.py  \
    dataset_dir=/absolute/path/to/dataset/directory \
    mlflow_tracking_uri=$(az ml workspace show --query mlflow_tracking_uri) \
    mlflow_experiment=retina_mnist \
    fast_dev_run=True
```

Note that enabling `fast_dev_run` mode will run only a single training and validation batch. After verifying that everything's working, you can re-run the above without setting the `fast_dev_run` flag to start a full training run. At this point, you'll be able to track the progress in your Azure ML workspace.

For more details on Azure's MLFlow compatibility see [MLflow and Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2).

# Evaluation

## Local

To evaluate the model, you can run the following command with the appropriate run ID:

```bash
python src/ml/test.py  \
    dataset_dir=/absolute/path/to/dataset/directory \
    mlflow_tracking_uri=http://127.0.0.1:8080 \
    mlflow_experiment=retina_mnist \
    mlflow_run_id=...
```

If you want to register the model in the MLFlow model registry, you can add the model name as follows:

```bash
python src/ml/test.py  \
    dataset_dir=/absolute/path/to/dataset/directory \
    mlflow_tracking_uri=http://127.0.0.1:8080 \
    mlflow_experiment=retina_mnist \
    mlflow_run_id=... \
    model_name=DiabeticRetinopathySeverityClassifier
```

## Azure

To evaluate the model, you can run the following command with the appropriate run ID:

```bash
python src/ml/test.py  \
    dataset_dir=/absolute/path/to/dataset/directory \
    mlflow_tracking_uri=$(az ml workspace show --query mlflow_tracking_uri) \
    mlflow_experiment=retina_mnist \
    mlflow_run_id=...
```

If you want to register the model in the MLFlow model registry, you can add the model name as follows:

```bash
python src/ml/test.py  \
    dataset_dir=/absolute/path/to/dataset/directory \
    mlflow_tracking_uri=$(az ml workspace show --query mlflow_tracking_uri) \
    mlflow_experiment=retina_mnist \
    mlflow_run_id=... \
    model_name=DiabeticRetinopathySeverityClassifier
```



