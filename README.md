Machine Learning for Credit Risk Assessment in Banking

Overview
Predicting a company's risk of bankruptcy is a critical aspect of financial decision-making. This project demonstrates how machine learning can assist institutions in proactively managing credit and investment risks. Using a dataset of Polish company financials (2000–2013), I developed a machine learning pipeline that achieves 89.54% accuracy and lays the groundwork for further enhancements.

Why It Matters
Credit risk is a key element of financial strategy. By identifying early indicators of financial distress, businesses can:

Safeguard their portfolios against potential losses.
Build more reliable lending models.
Make data-driven investment decisions with greater confidence.
This model provides a foundation for integrating real-time financial monitoring, paving the way for smarter, more responsive risk management.

Features
Baseline Model: Logistic Regression
An explainable starting point for predicting bankruptcy, leveraging core financial metrics.

Advanced Model: Random Forest
Achieving 89.54% accuracy, the Random Forest model uncovers complex relationships in the data, offering significant improvements over linear methods.

Future Roadmap:
Short-Term Goals: Improve feature selection to capture more nuanced financial interactions.
Medium-Term Goals: Test advanced AI models, such as Gradient Boosting and Neural Networks.
Long-Term Goals: Incorporate real-time data sources for dynamic risk assessment.

Data
Source: Polish Companies Bankruptcy Dataset (UCI Machine Learning Repository)
Time Period: 2000–2013
Class Distribution: Balanced across solvent and insolvent companies.

Technology Stack
Programming Language: Python
Libraries: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
Machine Learning Models: Logistic Regression, Random Forest
Workflow: Data preprocessing → Feature selection → Model training → Evaluation

Key Results
Accuracy: 89.54% (Random Forest)

Insights:
Financial metrics such as profitability and liquidity ratios strongly correlate with bankruptcy risk.
Ensemble models like Random Forest outperform simpler, linear models in identifying complex patterns.
Real-World Applications
For Lenders: Strengthen underwriting processes by assessing creditworthiness more accurately.
For Investors: Evaluate corporate health to mitigate risk in investment portfolios.
For Enterprises: Use as a benchmark for internal financial planning and risk management.
