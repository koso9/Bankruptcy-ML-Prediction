# AI-Driven Credit Risk Assessment for Banking: Predicting Bankruptcy with Machine Learning

## Overview
Predicting a company's risk of bankruptcy is a cornerstone of financial strategy. This project leverages machine learning to proactively assess credit and investment risks, offering data-driven solutions to safeguard financial portfolios. By analyzing the Polish Companies Bankruptcy Dataset (2000–2013), this project achieved **89.54% accuracy**, showcasing a scalable pipeline for modern credit risk evaluation.

---

## Why It Matters
- **Enhance portfolio resilience**: Identify early warning signs of financial distress to prevent losses.
- **Build smarter lending models**: Enable banks to make more informed credit decisions.
- **Empower investment strategies**: Provide tools to assess corporate health for investors.

---

## Features
- **Baseline Model: Logistic Regression**:
  - A straightforward, explainable starting point leveraging essential financial metrics.
  - **Accuracy**: 94.91%
  - **Precision (Minority Class)**: 50.00%
  - **Recall (Minority Class)**: 0.007%
  - **F1-Score (Minority Class)**: 0.017
- **Advanced Model: Random Forest**:
  - Achieves **95.14% accuracy**, uncovering complex relationships in financial data.
  - **Precision (Minority Class)**: 95.28%
  - **Recall (Minority Class)**: 4.35%
  - **F1-Score (Minority Class)**: 8.66%

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
