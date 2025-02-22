# AI-Driven Credit Risk Assessment for Banking: Predicting Bankruptcy with Machine Learning

## Overview
Bankruptcy prediction is a critical tool for financial institutions, investors, and risk managers. This project leverages machine learning models to assess corporate financial health, reducing risk exposure and enhancing portfolio decision-making. Using the **Polish Companies Bankruptcy Dataset (2000–2013)**, this model evaluates key financial indicators to classify firms as financially stable or at risk of bankruptcy. This project achieved **96.28% accuracy using Calibrated XGBoost**, demonstrating a scalable approach to modern credit risk evaluation and financial health assessment.

---

## Why It Matters
- **Enhance portfolio resilience**: Early identification of financial distress helps mitigate losses.  
- **Strengthen Lending & Investment Models**: Improved credit risk assessments enhance loan approvals and investment strategies.  
- **Optimize Risk Management**: A data-driven approach to corporate health assessment helps businesses proactively navigate financial challenges.  

---

## Key Features
**Machine Learning Models**:
This project evaluates multiple models to balance accuracy, precision, and recall for bankruptcy prediction:

- **Baseline Model: Logistic Regression**:
  - A straightforward, explainable starting point leveraging essential financial metrics.
  - **Accuracy**: 94.91%
  - **Precision (Minority Class)**: 50.00%
  - **Recall (Minority Class)**: 0.009%
  - **F1-Score (Minority Class)**: 0.018
    
- **Advanced Model: Random Forest**:
  - Achieves **95.14% accuracy**, uncovering complex relationships in financial data.
  - **Precision (Minority Class)**: 95.24%
  - **Recall (Minority Class)**: 4.54%
  - **F1-Score (Minority Class)**: 8.66%

- **Advanced Model: XGBoost Model**:
  - Achieves **95.84% accuracy**, brief sentence.
  - **Precision (Minority Class)**: 90.00%
  - **Recall (Minority Class)**: 20.41%
  - **F1-Score (Minority Class)**: 33.27%
    
- **Advanced Model: Calibrated XGBoost (Best Model)**:
  - Achieves **96.01% accuracy**, brief sentence.
  - **Precision (Minority Class)**: 80.65%
  - **Recall (Minority Class)**: 28.34%
  - **F1-Score (Minority Class)**: 41.95%
    
**Why Calibrated XGBoost?**  
By applying **Platt Scaling**, the calibrated XGBoost model improves probability estimates, making it more effective for financial decision-making where precision and recall must be balanced.
 
  
---

## Future Roadmap
- **Short-Term**:
  - **Enhance SMOTE Implementation**: Experiment with variations such as Borderline-SMOTE and SMOTE-Tomek links to improve minority class recall further.
  - **Feature Engineering**: Include derived ratios like Altman Z-scores and other financial indices for better bankruptcy prediction.
- **Medium-Term**:
  - **Test Advanced Models**: Evaluate Gradient Boosting Machines (e.g., XGBoost, LightGBM) and Neural Networks to explore non-linear relationships.
  - **Hyperparameter Tuning**: Use automated tools like Optuna or GridSearchCV for optimal model configurations.
- **Long-Term**:
  - **Real-Time Risk Monitoring**: Integrate real-time financial data streams for dynamic risk assessment.
  - **Explainability Tools**: Leverage SHAP and LIME to interpret model decisions for stakeholders.

---

## Data Source
- **Dataset**: Polish Companies Bankruptcy Dataset (UCI Repository)
- **Scope**: 2000–2013 financial data for 5 years, representing solvency and insolvency cases.
- **Variables**: Includes profitability, liquidity, leverage, and efficiency ratios.

---

## Technology Stack
- **Languages**: Python
- **Libraries**: scikit-learn, pandas, NumPy, Matplotlib, Seaborn, imbalanced-learn
- **Techniques**: Data preprocessing, SMOTE for class balancing, Random Forest, Logistic Regression
- **Workflow**: End-to-end pipeline covering preprocessing, model training, evaluation, and visualization.

---

## Key Results
### Logistic Regression:
- **Accuracy**: 94.91%
- **Precision (Minority Class)**: 50.00%
- **Recall (Minority Class)**: 0.007%
- **F1-Score (Minority Class)**: 0.017

### Random Forest:
- **Accuracy**: 95.14%
- **Precision (Minority Class)**: 95.28%
- **Recall (Minority Class)**: 4.35%
- **F1-Score (Minority Class)**: 8.66%

- **Top Features**: Key financial metrics include profitability ratios and liquidity measures.

---

## Real-World Applications
- **For Banks**: Develop automated underwriting systems that evaluate corporate creditworthiness.
- **For Investors**: Use bankruptcy risk predictions to make informed portfolio allocation decisions.
- **For Enterprises**: Benchmark internal financial health against industry trends.

---

## Visuals
1. **Class Distribution Before and After SMOTE**:
   - Visualizes the impact of balancing the dataset for training.
2. **Feature Importance**:
   - Displays the top 10 financial metrics driving the Random Forest model.
3. **Confusion Matrices**:
   - Highlights improved predictions for the minority class with Random Forest.

---

## How to Run the Code
1. Clone this repository.
2. Install dependencies listed in `requirements.txt`.
3. Run `bankruptcy_prediction.py` in your Python environment.
4. Review generated visualizations and performance metrics in the `output` folder.

---

## Next Steps
This project addresses the challenge of predicting rare events like bankruptcy, especially with imbalanced datasets. While the results show promise, future enhancements will focus on:
- **Boosting Recall for the Minority Class**: Explore cost-sensitive learning and further tuning of SMOTE variations.
- **Dynamic Data Integration**: Incorporate real-time financial data for continuous risk monitoring.
- **Explainability**: Apply interpretability techniques like SHAP and LIME to build trust and provide actionable insights for stakeholders.

Feel free to reach out or contribute to this project for further improvements!
