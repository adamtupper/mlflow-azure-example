# Track Experiments on Azure with MLflow

A minimal example of using MLflow to track experiments on Microsoft Azure.

## Setup Development Environment

To run this example you need to have setup the Azure Machine Learning CLI. Instructions on how to do this can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public)

Furthermore, you need to have installed the Python dependencies listed in `requirements.txt`. It is reccomended that you do this in an isolated virtual environment (using virtualenv, Anaconda, or Poetry, etc.).

## Run

To quickly test that everything is set up correctly, you can run the training script with the `fast_dev_run` flag set. This runs only a single training and validation batch. Note this will disable logging, so you won't see anything in your Azure ML workspace yet.

```python
python src/ml/train.py  \
    mlflow_tracking_uri=$(az ml workspace show --query mlflow_tracking_uri) \
    fast_dev_run=True
```

After verfiying that everything's working, you can re-run the above without setting the `fast_dev_run` flag to start a full training run. At this point, you'll be able to track the progress in your Azure ML workspace.

