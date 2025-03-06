





zenml integration install mlflow -y

zenml stack descsribe
  
zenml disconnect
zenml up
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES\n
 

zenml experiment-tracker register mlflow_tracker --flavor=mlflow

zenml model-deployer register mlflow --flavor=mlflow

zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set

python run_pipeline.py


mlflow ui --backend-store-uri "file:/Users/phanindrakumar/Library/Application Support/zenml/local_stores/1d1823df-5c53-4ec0-abe9-67b09502844c/mlruns"















