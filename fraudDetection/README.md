# 🧠 Machine Learning Project: Fraud Detection
Welcome to the **Automated Training and Prediction Pipeline** repository! This project is designed to streamline the process of data validation, training, and prediction using machine learning models. It includes comprehensive modules for data ingestion, preprocessing, model training, and predictions, along with robust logging and monitoring functionalities.

---

## 📁 Project Structure

Here's a breakdown of the key directories and files in this repository:

- **Data Transformation:**
  - `DataTransform_Training/` - Scripts for transforming training data.
  - `DataTransformation_Prediction/` - Scripts for transforming prediction data.

- **Data Validation & Insertion:**
  - `DataTypeValidation_Insertion_Training/` - Validates data types and inserts training data.
  - `DataTypeValidation_Insertion_Prediction/` - Validates data types and inserts prediction data.
  - `Prediction_Raw_Data_Validation/` - Validates raw data for prediction.
  - `Training_Raw_data_validation/` - Validates raw data for training.

- **Exploratory Data Analysis:**
  - `EDA/` - Contains notebooks and scripts for performing exploratory data analysis on the dataset.

- **Data Ingestion & Preprocessing:**
  - `data_ingestion/` - Scripts to ingest data from various sources.
  - `data_preprocessing/` - Preprocessing scripts for cleaning and preparing data for model training.
  - `preprocessing_data/` - Additional preprocessing routines.

- **Model Management:**
  - `best_model_finder/` - Scripts to identify and save the best performing model.
  - `models/` - Directory where trained models are stored.

- **File Operations:**
  - `file_operations/` - Utilities to handle file operations like saving and loading models.

- **Training and Prediction:**
  - `trainingModel.py` - Main script to train models.
  - `predictFromModel.py` - Main script to perform predictions using trained models.
  - `training_Validation_Insertion.py` - Validates and inserts data before training.
  - `prediction_Validation_Insertion.py` - Validates and inserts data before prediction.
  - `Prediction_Batch_files/` - Batch files for prediction.
  - `Training_Batch_Files/` - Batch files for training.

- **Database Management:**
  - `Prediction_Database/` - Manages database operations for prediction.
  - `Training_FileFromDB/` - Handles training files retrieved from the database.
  - `Prediction_FileFromDB/` - Handles prediction files retrieved from the database.

- **Logging:**
  - `Training_Logs/` - Logs for training processes.
  - `Prediction_Logs/` - Logs for prediction processes.
  - `application_logging/` - General application logs.

- **Miscellaneous:**
  - `flask_monitoringdashboard.db` - Database file for monitoring the Flask application.
  - `Procfile` - For deploying the application on platforms like Heroku.
  - `requirements.txt` - Lists all the dependencies required to run the project.
  - `schema_prediction.json` - JSON schema for validating prediction data.
  - `schema_training.json` - JSON schema for validating training data.
  - `test.py` - Test scripts for various components of the project.

---

## 🚀 Getting Started

### 1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository

### 2. **Install Dependencies:**
Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt

