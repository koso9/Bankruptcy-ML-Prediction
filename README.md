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
  - A straightforward, explainable starting point, leveraging essential financial metrics to build a foundational bankruptcy prediction model.
- **Advanced Model: Random Forest**:
  - Achieves **89.54% accuracy** by uncovering nonlinear relationships in financial data, demonstrating the power of ensemble methods over linear techniques.

---

## Future Roadmap
- **Short-Term**: Optimize feature selection using techniques like Recursive Feature Elimination (RFE) to capture subtle financial interactions.
- **Medium-Term**: Integrate advanced models like Gradient Boosting Machines (e.g., XGBoost) and Neural Networks for improved performance.
- **Long-Term**: Incorporate real-time financial data streams for dynamic, live risk assessment.

---

## Data Source
- **Dataset**: Polish Companies Bankruptcy Dataset (UCI Repository)
- **Scope**: 2000–2013 financial data for 5 years, representing solvency and insolvency cases.
- **Variables**: Includes profitability, liquidity, leverage, and efficiency ratios.

---

## Technology Stack
- **Languages**: Python
- **Libraries**: scikit-learn, pandas, NumPy, Matplotlib, Seaborn
- **Techniques**: Data preprocessing, SMOTE for class balancing, ensemble methods
- **Workflow**: End-to-end pipeline covering preprocessing, model training, evaluation, and visualization.

---

## Key Results
- Achieved **89.54% accuracy** with Random Forest, a significant improvement over baseline Logistic Regression (71%).
- **F1-Score for the minority class** increased from 0.02 (Logistic Regression) to 0.09 (Random Forest), addressing class imbalance issues.

---

## Real-World Applications
- **For Banks**: Develop automated underwriting systems that evaluate corporate creditworthiness.
- **For Investors**: Use bankruptcy risk predictions to make informed portfolio allocation decisions.
- **For Enterprises**: Benchmark internal financial health against industry trends.

---

## Visuals
1. **Class Distribution Before and After SMOTE**:
   ![Class Distribution](images/class_distribution.png)

2. **Feature Importance**:
   ![Feature Importance](images/feature_importance.png)

3. **Performance Metrics Comparison**:
   | Metric           | Logistic Regression | Random Forest |
   |------------------|---------------------|---------------|
   | Accuracy         | 71%                 | 89.54%        |
   | F1-Score (Class 1)| 0.02                | 0.09          |

---

## How to Run the Code
1. Clone this repository.
2. Install the dependencies listed in `requirements.txt`.
3. Run `bankruptcy_prediction.py` in your Python environment.
4. View the generated visualizations and performance metrics in the output folder.

Machine Learning for Credit Risk Assessment in Banking

## Next Steps
For detailed results and insights, explore:
- The full model implementation in [bankruptcy_prediction.py](bankruptcy_prediction.py)
- Visualizations stored in the `output` folder.
- Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data).

---

Feel free to reach out or contribute to this project for further improvements!
