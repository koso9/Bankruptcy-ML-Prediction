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
 
---
## **Top 10 Features Influencing Bankruptcy Predictions**
1️⃣ **Net Profit / Total Assets = Return on Assets**  
2️⃣ **Total Liabilities / Total Assets = Liabilities Ratio**  
3️⃣ **Cash Ratio**  
4️⃣ **Working Capital / Total Assets**  
5️⃣ **EBIT / Total Assets**  
6️⃣ **Debt Repayment Ratio**  
7️⃣ **Sales / Total Assets**  
8️⃣ **Equity / Total Assets**  
9️⃣ **Gross Profit / Short-Term Liabilities**  
🔟 **Retained Earnings / Total Assets**  

*(Feature importance is derived from XGBoost’s gain-based ranking.)*

## **Data Source**

- Dataset: Polish Companies Bankruptcy Dataset (UCI Repository) https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data
  
-	Scope: 2000–2013 financial data covering 5 years of company records
- Variables: Profitability, liquidity, leverage, and efficiency ratios
________________________________________

## **Real-World Applications**
- **For Banks & Lenders – Develop AI-powered underwriting models for credit risk evaluation**
- **For Investors & Hedge Funds – Identify at-risk firms and refine investment strategies**
- **For Enterprises & CFOs – Benchmark corporate financial health**
________________________________________
## **Technology Stack**
- **Python – scikit-learn, XGBoost, NumPy, Pandas**
- **Data Processing – Feature engineering, class balancing (SMOTE), and scaling**
- **Model Training – Random Forest, XGBoost, Logistic Regression**
- **Evaluation Metrics – Confusion matrix, precision-recall, and F1-score**

---
## Next Steps & Future Roadmap
While this model demonstrates strong predictive performance, particularly with Calibrated XGBoost, there are opportunities to refine its accuracy and applicability further. Enhancing recall for the minority class, incorporating real-time financial data, and improving model interpretability will be key areas of focus moving forward.

- **Boosting Recall for the Minority Class**: Explore cost-sensitive learning and further tuning of SMOTE variations.
- **Dynamic Data Integration**: Incorporate real-time financial data for continuous risk monitoring.
- **Explainability**: Apply interpretability techniques like SHAP and LIME to build trust and provide actionable insights for stakeholders.
---


## How to Run the Code
1. Clone this repository.
2. Install dependencies listed in `requirements.txt`.
3. Run `bankruptcy_prediction.py` in your Python environment.
4. Review generated visualizations and performance metrics in the `output` folder.

---
Feel free to reach out or contribute to this project for further improvements!
