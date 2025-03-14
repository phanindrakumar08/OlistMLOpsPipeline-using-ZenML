# OlistMLOpsPipeline using ZenML 

![image](https://github.com/user-attachments/assets/c96ce926-8868-48d3-9cde-75288aa65022)


This----------- repository showcases a practical Machine Learning Operations (MLOps) pipeline built around the Brazilian Olist e-commerce dataset. The project focuses on demonstrating robust MLOps practices using ZenML, MLflow, and clean, modular code.

## Project Overview

This project utilizes the publicly available Olist Store dataset from Kaggle(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data), which contains information on 100k orders from 2016 to 2018 across multiple marketplaces in Brazil. I have focused on building a streamlined MLOps pipeline to ingest, clean, train, and evaluate a Linear Regression model, with an emphasis on automation and reproducibility.

**Key Features:**

* **ZenML Pipelines:** Orchestrated data ingestion, cleaning, model training, and evaluation steps using ZenML.
* **MLflow Integration:** Experiment tracking and metric logging with MLflow.
* **Modular Design:** Implementation of abstract class methods for data cleaning, model training, and evaluation, promoting code reusability and maintainability.
* **Data Preprocessing:** Numerical data extraction and cleaning based on Exploratory Data Analysis (EDA) findings.
* **Model Training:** Training a Linear Regression model for predictive insights.
* **Model Evaluation:** Comprehensive evaluation using metrics like R2 score, MSE, and RMSE.
* **ZenML Dashboard:** Detailed insights into pipeline execution and artifact tracking

## Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone [your_repository_url]
    cd OlistMLOpsPipeline
    ```
2.  **Set up a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS and Linux
    venv\Scripts\activate  # On Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the ZenML Pipeline:**
    ```bash
    python run_pipeline.py
    ```
5.  **View Results:**
    * Access the ZenML dashboard at `http://127.0.0.1:8237/` to view pipeline runs and artifacts.
    * View MLflow metrics in the MLflow UI.

## Future Enhancements

* **Streamlit Application:** Develop an interactive Streamlit app for users to input data and receive predictions.
* **Deployment Pipeline:** Implement a deployment pipeline to serve the trained model in a production environment.
* **Advanced Model Evaluation:** Incorporate more sophisticated evaluation techniques and visualizations.
* **CI/CD Integration:** Set up continuous integration and continuous deployment pipelines for automated testing and deployment.

## Dataset Information

The dataset is a real-world commercial dataset provided by Olist Store, anonymized for public use. It includes order information, customer reviews, product attributes, and geolocation data.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to suggest improvements or report bugs.
