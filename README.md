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
By applying Isotonic Regression, the calibrated XGBoost model improves probability estimates, making it more effective for financial decision-making where precision and recall must be balanced.
________________________________________
## **Feature Importance: What Drives Bankruptcy Risk?**
This model identifies the most influential financial ratios in predicting bankruptcy using XGBoost’s gain-based ranking.
What does "gain" mean?
- **Gain measures how much a feature improves the model’s accuracy when used in decision-making.**
- **A higher gain score means the feature is more critical in predicting bankruptcy risk.**

## **Top 10 Features Influencing Bankruptcy Predictions**
1️⃣ **Return on Assets = Net Profit / Total Assets**  
2️⃣ **Liabilities Ratio = Total Liabilities / Total Assets**  
3️⃣ **Cash Ratio**  
4️⃣ **Net Working Capital Ratio = Working Capital / Total Assets**  
5️⃣ **Basic Earning Power = EBIT / Total Assets**  
6️⃣ **Debt Repayment Ratio**  
7️⃣ **Asset Turnover Ratio = Sales / Total Assets**  
8️⃣ **Equity Ratio = Equity / Total Assets**  
9️⃣ **Short-Term Profitability Ratio = Gross Profit / Short-Term Liabilities**  
🔟 **Retained Earnings Ratio = Retained Earnings / Total Assets**  

________________________________________
## **Data Source**

- **Dataset: Polish Companies Bankruptcy Dataset (UCI Repository) https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data. Please go to the UCI Repository link and download the zip file before running the code**
  
-	**Scope: 2000–2013 financial data covering 5 years of company records**
- **Variables: Profitability, liquidity, leverage, and efficiency ratios**
________________________________________

## **Business Use Case & Deployment**
This model has broad applications across banking, investment, and corporate finance, offering predictive insights to mitigate financial risk.
### **Lending & Credit Risk Management**
- **Lenders can integrate this model into loan underwriting systems to assign bankruptcy risk scores to borrowers before extending credit.**
- **The model can support automated risk-based pricing, adjusting loan terms or requiring additional collateral based on bankruptcy probability**
- **Regulatory compliance teams can use probability scores for stress testing and risk mitigation**
 
### **Investment & Portfolio Risk Analysis**
- **Hedge funds and asset managers can identify financially unstable companies before making investment decisions**
- **Bankruptcy probability can inform short-selling strategies or risk-adjusted portfolio allocation**
- **Private equity firms can use predictions to assess acquisition targets and distressed assets**

### **Corporate Finance & Risk Monitoring**
- **CFOs and financial planning teams can use the model to benchmark their company's financial health vs. industry peers**
- **Companies can implement the model in early warning systems to predict financial distress and adjust capital structure**

### **Model Deployment & Integration**
- **The model can be deployed via API for real-time bankruptcy risk assessment**
- **Financial institutions can integrate predictions into internal risk dashboards for enhanced decision-making**
- **A real-time monitoring system can be built to automate early intervention strategies**
  
________________________________________
## **Technology Stack**
- **Python – scikit-learn, XGBoost, NumPy, Pandas**
- **Data Processing – Feature engineering, class balancing (SMOTE), and scaling**
- **Model Training – Random Forest, XGBoost, Logistic Regression**
- **Evaluation Metrics – Confusion matrix, precision-recall, and F1-score**
  
________________________________________

## Next Steps & Future Roadmap
While this model demonstrates strong predictive performance, particularly with Calibrated XGBoost, there are opportunities to refine its accuracy and applicability further. Enhancing recall for the minority class, incorporating real-time financial data, and improving model interpretability will be key areas of focus moving forward.

### **Further Optimization of Calibrated XGBoost**:
- **Fine-tune hyperparameters for better predictive performance**
- **Explore additional financial ratios to enhance accuracy**
- **Test alternative probability calibration methods**
  
### **Feature Engineering & Refinement**:
- **Identify key financial metrics with the highest predictive power**
- **Remove low-impact features to improve efficiency**
- **Enhance SMOTE Implementation – Experiment with SMOTE-Tomek and Borderline-SMOTE to improve recall**
- **Investigate non-linear transformations and interaction effects**

### **Business Application & Deployment**:
- **Evaluate use cases in lending, credit risk, and investment analysis**
- **Deploy the model for real-time bankruptcy predictions**
- **Develop a monitoring framework to track and recalibrate performance**

### **Model Interpretability**:
- **Implement SHAP & LIME to explain key drivers of bankruptcy risk**
- **Provide visualization tools for stakeholders**
  
### **Alternative Modeling Techniques**:
- **Test Ensemble Stacking with Random Forest, XGBoost, and Logistic Regression**
- **Use Bayesian Optimization for hyperparameter tuning**
- **Explore LSTM models for time-series bankruptcy prediction**

### **Expanding Dataset Scope**:
- **Incorporate macroeconomic indicators (e.g., interest rates, GDP) for added predictive power**
- **Validate performance on global financial statements**

### **Deployment & Integration**:
- **Develop an API or dashboard for real-time bankruptcy assessments**
- **Automate model retraining with new financial data**
- **Ensure compliance with Basel III credit risk standards**

________________________________________

Feel free to reach out or contribute to this project for further improvements!
