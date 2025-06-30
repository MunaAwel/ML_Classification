# Breast Cancer Classification

## Overview

This project aims to classify breast tumors as **malignant** or **benign** using machine learning techniques applied to the **Breast Cancer Wisconsin dataset**. The objective is to support early and accurate diagnosis by building and evaluating predictive models based on features extracted from digitized images of breast mass biopsies.

The workflow includes **data preprocessing**, **exploratory data analysis**, **feature selection**, **model training**, and **evaluation**. The project concludes with a model selection based on both technical performance and organizational needs, emphasizing clinical interpretability.

---

## Tools & Technologies

### Programming Language & Libraries

- **Python** – end-to-end model pipeline  
- **scikit-learn** – classification models, feature selection, cross-validation  
- **Pandas, NumPy** – data manipulation  
- **Matplotlib, Seaborn** – visualization  
- **SHAP** – model explainability and feature importance  

### Models Implemented

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- XGBoost  
- Decision Tree  

---

## Workflow Summary

1. **Data Loading**  
   - Loaded Breast Cancer dataset from `sklearn.datasets`.

2. **Preprocessing**  
   - Checked for missing values  
   - Standardized features  
   - Performed feature selection using RFE and feature importance  

3. **Exploratory Data Analysis (EDA)**  
   - Visualized distributions and feature relationships  
   - Correlation matrix and class balance review  

4. **Modeling & Evaluation**  
   - Trained and tuned multiple models  
   - Evaluated using Accuracy, F1/F2 Score, AUROC  
   - Cross-validated using Stratified K-Fold  

5. **Model Explainability**  
   - Applied SHAP for global and local interpretability  
   - Visualized individual predictions for clinicians  

---

## Results

- Best-performing model: **Logistic Regression**  
- Accuracy: **0.988**  
- F1/F2 Score: **0.990**  
- AUROC: **0.998**  
- SHAP analysis highlighted key features influencing predictions such as **mean concavity, radius, and perimeter**

---

## Model Selection & Conclusion

Based on both technical performance and organizational requirements, **Logistic Regression** was chosen as the final model for deployment.

From a technical standpoint, Logistic Regression achieved the **highest performance across all evaluation metrics**, including:
- Accuracy: 0.988  
- F1/F2 Score: 0.990  
- AUROC: 0.998  

From an organizational perspective, Logistic Regression offers:
- **High interpretability**, making it ideal for healthcare settings  
- **Straightforward coefficients** that clearly show feature influence  
- Seamless integration with **SHAP** for visual and individualized explanations  
- **Ease of deployment** in real-world clinical environments  

While other models such as **XGBoost** and **SVM** showed competitive performance, they operate as "black-box" models, which could hinder clinical trust. **Random Forest** provided a balance between performance and interpretability but still lacked the simplicity of Logistic Regression. **Decision Tree**, although interpretable, showed the lowest accuracy and was not suitable for deployment.

### Limitations & Ethical Considerations

Every model comes with trade-offs. Before deployment, the following steps are critical:
- **External validation** on an independent dataset to confirm generalizability  
- **Bias checks** across different populations  
- Ensuring alignment with **ethical standards and clinical guidelines**  

---

## Output

- Confusion matrix plots  
- ROC and PR curves  
- SHAP summary and force plots  
- Model comparison report  

