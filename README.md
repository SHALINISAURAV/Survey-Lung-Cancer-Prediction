# 🩺 Lung Cancer Prediction using Machine Learning

## 📖 Project Overview
This project applies **basic to advanced Machine Learning models** to predict the likelihood of lung cancer based on patient survey data.  
It demonstrates the **end-to-end ML workflow**: from **data preprocessing** → **training** → **evaluation** → **saving models for deployment**.  

**Dataset:** `survey lung cancer.csv` (309 rows, 16 columns)

---

## 🎯 Goals
- Explore and preprocess the dataset  
- Train multiple classification models  
- Evaluate performance using accuracy & classification metrics  
- Save trained models (`.pkl`) for reuse  
- Compare results visually and conclude the best model  

---

## 🗂️ Dataset Information
- **Rows:** 309  
- **Columns:** 16  
- **Target Variable:** `LUNG_CANCER (YES/NO)`  
- **Features:**  
  - `GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL CONSUMING, COUGHING, SHORTNESS OF BREATH, SWALLOWING DIFFICULTY, CHEST PAIN`

---

## ⚙️ Workflow
1. **Data Loading & Exploration**  
   - Checked datatypes, unique values, and missing values  

2. **Data Preprocessing**  
   - Encoded categorical variables (`Gender`, `Lung Cancer`)  
   - Converted binary features from `1/2` → `0/1`  
   - Applied feature scaling for models like KNN & SVM  

3. **Model Training & Saving**  
   Implemented and trained using **Scikit-learn Pipelines**:  
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  
   - Support Vector Machine (SVM)  
   - K-Nearest Neighbors (KNN)  
   - Naive Bayes  

   ➡️ Each trained pipeline was **saved as `.pkl` file** using `joblib`.  

4. **Evaluation**  
   - Accuracy Score  
   - Confusion Matrix  
   - Classification Report  

5. **Comparison**  
   - Tabular results of model accuracies  
   - Visualization with bar chart  

6. **Future Use**  
   - Any saved model can be loaded (`joblib.load`) and directly used for prediction on new patient data.  

---

## 📊 Results (Accuracy)
| Model               | Accuracy |
|----------------------|----------|
| Logistic Regression | **91.18%** |
| Decision Tree       | **79.41%** |
| Random Forest       | **91.18%** |
| SVM                 | **85.29%** |
| KNN                 | **85.29%** |
| Naive Bayes         | **88.24%** |

✅ **Logistic Regression & Random Forest** performed the best overall.  

---

## 🛠️ Tech Stack
- Python 🐍  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Joblib (for saving/loading models)  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/shalinisaurav/lung-cancer-prediction.git
   cd lung-cancer-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Load and predict with a saved model:
   ```python
   import joblib
   model = joblib.load("Random_Forest.pkl")
   prediction = model.predict([[1, 62, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0]])
   print(prediction)  # 0 = No Cancer, 1 = Cancer
   ```

---

## 👩‍💻 Author✨
**Shalini Saurav**  
📧 Contact: [shalinisourav07@gmail.com]  

---

