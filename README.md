# AutoJudge: Predicting Programming Problem Difficulty
## Personal info 
NAME-Akhil Kumar
ENROLLMENT-24117009  
BRANCH- B.Tech Mechanical Engineering
COLLEGE-Indian Institute of Technology, Roorkee

##Demonstration video
https://drive.google.com/file/d/1HWRIjEmJUA-TKhkHXC89HBbiNOOtH0qJ/view?usp=drivesdk
## Project Overview
AutoJudge is a machine learning-based tool that predicts the difficulty level of competitive programming problems. By analyzing problem statements and metadata from platforms like Codeforces, AutoJudge classifies problems into **Easy**, **Medium**, or **Hard**, and also predicts the **rating** of the problem.

---

## Features
- Predicts problem difficulty using **NLP** and **tabular features**
- Provides both **difficulty level (classification)** and **rating (regression)**
- Simple **Streamlit web interface** for easy input and instant predictions

---

## Dataset
- Original dataset downloaded from [Hugging Face](https://huggingface.co/datasets/open-r1/codeforces) consisting of **Codeforces** data
- Dataset contains three subsets: `default`, `verifiable`, and `verifiable-prompts`
- This project uses the **default** subset
- Original dataset had separate **train** and **test** splits
- For this project, both splits were **combined** and a custom **80/20** train-test split was created
- Only features available to users on Codeforces are used, such as:
  - Title
  - Description
  - Input format
  - Output format
  - Time limit
  - Memory limit
  - Tags

> ⚠️ Note: The original dataset files are large; only a combined and preprocessed version is used for model training.

---

## File Structure
- **data/**
  - `codeforces_data.csv` – combined dataset  
  - `preprocessed_data.pkl` – dataset after preprocessing generated from `preprocessing_codeforces.ipynb`
- **model_training/**
  - Classification and regression model training scripts
  - TF-IDF vectorizer and feature engineering steps
- **preprocessing/**
  - Basic preprocessing notebooks and scripts
- **web_ui/**
  - `auto_judge_app.py` – Streamlit application  
  - `preprocessing.py` – preprocessing for user input  
  - `.pkl` files – saved preprocessing artifacts
- **best-models/**
  - Trained classification and regression models with best evaluation metrics

---

## Models Used (Classification)
- **XGBoost** – Accuracy: 67.15%  
- **Logistic Regression** – Accuracy: 67.55%  
- **Random Forest** – Accuracy: 64.69%  
- **Support Vector Machine (SVM)** – Accuracy: 64.12%  

---

## Models Used (Regression)
- **XGBoost Regressor** – MAE: ~355.26 | R² Score: 0.5819  
- **Random Forest Regressor** – MAE: ~385.78 | R² Score: 0.5227 

---

## Installation
```bash
# Clone the repository
git clone https://github.com/QyAkhil/AutoJudge
cd AutoJudge

# Create a virtual environment
python -m venv venv
venv\Scripts\Activate
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the app
```bash
streamlit run web_ui/auto_judge_app.py
