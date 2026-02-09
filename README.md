# Breast Cancer Prediction with Hyperparameter Tuning

A machine learning project that predicts whether a breast tumor is **malignant** or **benign** using diagnostic features.  
The project focuses on **model comparison and hyperparameter tuning** to improve classification performance on structured medical data.

---

## Tech Stack
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## Dataset
- Breast Cancer Wisconsin (Diagnostic) dataset  
- Source: Public UCI Machine Learning Repository  
- File used: `Breast Cancer Wisconsin sample.csv`  
- Target classes:
  - Malignant  
  - Benign  

---

## Approach
- Loaded and explored structured medical diagnostic data
- Performed data preprocessing and feature scaling
- Trained multiple classification models
- Applied hyperparameter tuning to identify the best-performing model
- Evaluated models using:
  - Accuracy
  - Classification Report
  - Confusion Matrix

---

## Results
- Best model selected based on validation performance
- Classification report shows strong precision and recall
- Confusion matrix confirms reliable class separation

Key result artifacts:
- `Best_Model_Score.png`
- `Classification_Report.png`
- `Confusion_Matrix.png`

---

## Files in This Repository
- `Breast Cancer Wisconsin (Diagnostic) - Hyper Parameter tuning.ipynb`  
  Main notebook containing data preprocessing, model training, and tuning

- `Breast Cancer Wisconsin sample.csv`  
  Dataset used for training and evaluation

- `Best_Model_Score.png`  
  Comparison of model performance

- `Classification_Report.png`  
  Precision, recall, and F1-score visualization

- `Confusion_Matrix.png`  
  Model prediction performance across classes

- `requirements.txt`  
  Python dependencies required to run the notebook

---

## How to Run
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Open the notebook:
4. Breast Cancer Wisconsin (Diagnostic) - Hyper Parameter tuning.ipynb
5. Run all cells sequentially

## Future Improvements

1. Add feature importance and interpretability (SHAP / LIME)
2. Explore ensemble methods for further performance gains
3. Deploy the model as a simple prediction application