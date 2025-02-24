import zipfile
import os
import pandas as pd
from scipy.io import arff  # For handling .arff files

# Define file paths
zip_file_path = input("Enter the path to your bankruptcy dataset ZIP file: ")
extraction_path = input("Enter the path where you want to extract the files: ")

# Create the extraction directory (if it doesn't exist)
os.makedirs(extraction_path, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)
    print(f"Files extracted to: {extraction_path}")

# Function to load .arff files into Pandas DataFrames
def load_arff_to_dataframe(arff_file_path):
    """
    Converts a .arff file into a Pandas DataFrame with corrected feature names.
    
    Args:
        arff_file_path (str): Path to the .arff file.
        
    Returns:
        DataFrame: Loaded data as a Pandas DataFrame.
    """
    data, meta = arff.loadarff(arff_file_path)
    df = pd.DataFrame(data)  

    # Correct column names using the known feature names
    feature_names = {
        f"Attr{i+1}": f"X{i+1}" for i in range(len(df.columns))  # Map AttrXX to X1, X2, etc.
    }
    df.rename(columns=feature_names, inplace=True)

    return df

# Load all .arff files into a dictionary of DataFrames
dataframes = {}

# Loop through each year's .arff file (1year.arff to 5year.arff)
for i in range(1, 6):
    arff_file_path = os.path.join(extraction_path, f'{i}year.arff')
    dataframes[f'{i}year'] = load_arff_to_dataframe(arff_file_path)

# Access the first year's DataFrame
df_1year = dataframes['1year']
print(df_1year.head())  # Print the first few rows

# Verify column names
print("Updated Column Names:", df_1year.columns)


# Combine all yearly DataFrames into a single DataFrame
combined_df = pd.concat([dataframes['1year'], dataframes['2year'], dataframes['3year'],
                         dataframes['4year'], dataframes['5year']], ignore_index=True)

# Print the shape of the combined DataFrame
print(f"Combined dataset shape: {combined_df.shape}")

# Display the first few rows of the combined dataset for inspection
print(combined_df.head())

# Check for missing values in the combined dataset
missing_values = combined_df.isnull().sum()  # Count missing values per column
print("Columns with missing values:")
print(missing_values[missing_values > 0])

# Calculate the percentage of data loss if rows with missing values are dropped
total_rows = combined_df.shape[0]
rows_with_nan = combined_df.dropna().shape[0]
percentage_lost = ((total_rows - rows_with_nan) / total_rows) * 100
print(f"Percentage of data lost by dropping rows with missing values: {percentage_lost:.2f}%")


from sklearn.model_selection import train_test_split

# Step 8: Separate Features (X) and Target Variable (y)
# Assuming the target variable column is named 'class'
# Replace 'class' with the actual column name if it differs in your dataset
X = combined_df.drop('class', axis=1)  # All columns except the target
y = combined_df['class']  # The target column

# Step 9: Split the Dataset into Training and Testing Sets
# Use an 80/20 split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets for verification
print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

# Convert byte strings (e.g., b'0') to integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Verify the transformation
print(f"Unique values in y_train: {y_train.unique()}")
print(f"Unique values in y_test: {y_test.unique()}")


from sklearn.impute import SimpleImputer

# Identify Columns with Missing Values
# Calculate the number of missing values in each column of the training set
missing_values = X_train.isnull().sum()
print("Columns with missing values:")
print(missing_values[missing_values > 0])

# Strategy for Missing Values
# Threshold: Drop columns where more than 5% of the rows have missing values
threshold = 0.05 * X_train.shape[0]
cols_to_drop = [col for col in X_train.columns if X_train[col].isnull().sum() > threshold]
cols_to_impute = [col for col in X_train.columns if X_train[col].isnull().sum() <= threshold]

print(f"Columns to drop: {cols_to_drop}")
print(f"Columns to impute: {cols_to_impute}")

# Drop Columns with Excessive Missing Values
# Drop these columns from both training and testing sets
X_train_dropped = X_train.drop(columns=cols_to_drop)
X_test_dropped = X_test.drop(columns=cols_to_drop)

# Impute Remaining Missing Values
# Using the median to fill in missing values
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_dropped)
X_test_imputed = imputer.transform(X_test_dropped)

# Imputed arrays back into DataFrames
X_train_final = pd.DataFrame(X_train_imputed, columns=X_train_dropped.columns, index=X_train_dropped.index)
X_test_final = pd.DataFrame(X_test_imputed, columns=X_test_dropped.columns, index=X_test_dropped.index)

# Verify the Results
print(f"Final Training Set Shape: {X_train_final.shape}")
print(f"Final Test Set Shape: {X_test_final.shape}")
print("\nMissing Values in Final Training Set (should be 0):")
print(X_train_final.isnull().sum().sum())
print("\nMissing Values in Final Test Set (should be 0):")
print(X_test_final.isnull().sum().sum())


from sklearn.preprocessing import StandardScaler

# Initialize the Scaler
scaler = StandardScaler()

# Fit the Scaler on the Training Data and Transform Both Training and Testing Sets
# StandardScaler standardizes the data to have a mean of 0 and a standard deviation of 1
X_train_scaled = scaler.fit_transform(X_train_final)  # Fit to training data and scale it
X_test_scaled = scaler.transform(X_test_final)       # Scale the testing data using the same scaler

# Convert the Scaled Data Back into DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_final.columns, index=X_train_final.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_final.columns, index=X_test_final.index)

# Verify Scaling Results
print("Scaled Training Set (first few rows):")
print(X_train_scaled.head())
print("\nScaled Test Set (first few rows):")
print(X_test_scaled.head())


from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Verify the class distribution
print("Class distribution before SMOTE:")
print(y_train.value_counts())

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())


import seaborn as sns
import matplotlib.pyplot as plt

# Before SMOTE
plt.figure(figsize=(10, 6))
sns.countplot(x=y_train, palette="viridis")
plt.title("Class Distribution Before SMOTE", fontsize=14)
plt.show()

# After SMOTE
plt.figure(figsize=(10, 6))
sns.countplot(x=y_train_smote, palette="viridis")
plt.title("Class Distribution After SMOTE", fontsize=14)
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize and Train the Logistic Regression Model
logreg = LogisticRegression(max_iter=1000, random_state=42)  # Increase max_iter to ensure convergence
logreg.fit(X_train_scaled, y_train)

#  Evaluate the Model on the Test Data
y_pred_logreg = logreg.predict(X_test_scaled)

# Print Evaluation Metrics
print("Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))


from sklearn.ensemble import RandomForestClassifier

# Initialize and Train the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate the Model on the Test Data
y_pred_rf = rf_model.predict(X_test_scaled)

# Step 24: Print Evaluation Metrics
print("Random Forest Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


import matplotlib.pyplot as plt
import numpy as np

# Extract feature importance values from the trained model
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort features by importance (descending)

# Ensure feature names dictionary is available
feature_names_dict = {
    "X1": "Net Profit / Total Assets",
    "X2": "Total Liabilities / Total Assets",
    "X3": "Working Capital / Total Assets",
    "X4": "Current Assets / Short-Term Liabilities",
    "X5": "Cash Ratio",
    "X6": "Retained Earnings / Total Assets",
    "X7": "EBIT / Total Assets",
    "X8": "Book Value of Equity / Total Liabilities",
    "X9": "Sales / Total Assets",
    "X10": "Equity / Total Assets",
    "X11": "Gross Profit + Extraordinary Items / Total Assets",
    "X12": "Gross Profit / Short-Term Liabilities",
    "X13": "Gross Profit + Depreciation / Sales",
    "X14": "Gross Profit + Interest / Total Assets",
    "X15": "Debt Repayment Ratio",
    "X16": "Gross Profit + Depreciation / Total Liabilities",
    "X17": "Total Assets / Total Liabilities",
    "X18": "Gross Profit / Total Assets",
    "X19": "Gross Profit / Sales",
    "X20": "Inventory Turnover",
    "X21": "Sales (n) / Sales (n-1)",
    "X22": "Profit on Operating Activities / Total Assets",
    "X23": "Net Profit / Sales",
    "X24": "Gross Profit (3 years) / Total Assets",
    "X25": "Equity - Share Capital / Total Assets",
    "X26": "Net Profit + Depreciation / Total Liabilities",
    "X27": "Profit on Operating Activities / Financial Expenses",
    "X28": "Working Capital / Fixed Assets",
    "X29": "Logarithm of Total Assets",
    "X30": "Total Liabilities - Cash / Sales",
    "X31": "Gross Profit + Interest / Sales",
    "X32": "Current Liabilities Turnover",
    "X33": "Operating Expenses / Short-Term Liabilities",
    "X34": "Profit on Sales / Total Assets",
    "X35": "Total Sales / Total Assets",
    "X36": "Current Assets - Inventories / Long-Term Liabilities",
    "X37": "Constant Capital / Total Assets",
    "X38": "Profit on Sales / Sales",
    "X39": "Receivables Turnover Ratio",
    "X40": "Liquidity Ratio",
    "X41": "Debt Coverage Ratio",
    "X42": "Rotation Receivables + Inventory Turnover",
    "X43": "Receivables * 365 / Sales",
    "X44": "Net Profit / Inventory",
    "X45": "EBITDA / Total Assets",
    "X46": "EBITDA / Sales",
    "X47": "Current Assets / Total Liabilities",
    "X48": "Short-Term Liabilities / Total Assets",
    "X49": "Short-Term Liabilities * 365 / Cost of Products Sold",
    "X50": "Equity / Fixed Assets",
    "X51": "Constant Capital / Fixed Assets",
    "X52": "Working Capital",
    "X53": "Sales - Cost of Products Sold / Sales",
    "X54": "Short-Term Liabilities / Sales",
    "X55": "Long-Term Liabilities / Equity",
    "X56": "Sales / Inventory",
    "X57": "Sales / Receivables",
    "X58": "Sales / Short-Term Liabilities",
    "X59": "Sales / Fixed Assets"
}

# Extract column names from X_train
feature_names = list(X_train.columns)

# Convert generic attribute names to readable names
mapped_feature_names = [feature_names_dict.get(feature, feature) for feature in feature_names]

# Select top 10 features by importance
top_10_features = [mapped_feature_names[i] for i in indices[:10]]
top_10_importances = importances[indices[:10]]

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(top_10_features, top_10_importances, color='skyblue')
plt.gca().invert_yaxis()
plt.title("Top 10 Features by Importance")
plt.xlabel("Feature Importance")
plt.show()


comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Logistic Regression": [
        accuracy_score(y_test, y_pred_logreg),
        classification_report(y_test, y_pred_logreg, output_dict=True)['1']['precision'],
        classification_report(y_test, y_pred_logreg, output_dict=True)['1']['recall'],
        classification_report(y_test, y_pred_logreg, output_dict=True)['1']['f1-score']
    ],
    "Random Forest": [
        accuracy_score(y_test, y_pred_rf),
        classification_report(y_test, y_pred_rf, output_dict=True)['1']['precision'],
        classification_report(y_test, y_pred_rf, output_dict=True)['1']['recall'],
        classification_report(y_test, y_pred_rf, output_dict=True)['1']['f1-score']
    ]
})
print(comparison)


# Confusion Matrices
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


%who


!pip install xgboost

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



#  Gradient Boosting (XGBoost)

xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluation
print("\nXGBoost Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))


#  Support Vector Machine (SVM)

svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluation
print("\nSVM Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))


#  K-Nearest Neighbors (KNN)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_knn = knn_model.predict(X_test_scaled)

# Evaluation
print("\nKNN Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))


# Probability Calibration (For XGBoost)

calibrated_xgb = CalibratedClassifierCV(xgb_model, method='isotonic', cv=5)
calibrated_xgb.fit(X_train_scaled, y_train)

# Predict probabilities
y_pred_xgb_calibrated = calibrated_xgb.predict_proba(X_test_scaled)[:, 1]

# Convert to binary (threshold at 0.5)
y_pred_xgb_binary = (y_pred_xgb_calibrated >= 0.5).astype(int)

# Evaluation
print("\nCalibrated XGBoost Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb_binary) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb_binary))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb_binary))


# Model Comparison Table

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "XGBoost": [
        accuracy_score(y_test, y_pred_xgb),
        classification_report(y_test, y_pred_xgb, output_dict=True)['1']['precision'],
        classification_report(y_test, y_pred_xgb, output_dict=True)['1']['recall'],
        classification_report(y_test, y_pred_xgb, output_dict=True)['1']['f1-score']
    ],
    "Calibrated XGBoost": [
        accuracy_score(y_test, y_pred_xgb_binary),
        classification_report(y_test, y_pred_xgb_binary, output_dict=True)['1']['precision'],
        classification_report(y_test, y_pred_xgb_binary, output_dict=True)['1']['recall'],
        classification_report(y_test, y_pred_xgb_binary, output_dict=True)['1']['f1-score']
    ],
    "SVM": [
        accuracy_score(y_test, y_pred_svm),
        classification_report(y_test, y_pred_svm, output_dict=True)['1']['precision'],
        classification_report(y_test, y_pred_svm, output_dict=True)['1']['recall'],
        classification_report(y_test, y_pred_svm, output_dict=True)['1']['f1-score']
    ],
    "KNN": [
        accuracy_score(y_test, y_pred_knn),
        classification_report(y_test, y_pred_knn, output_dict=True)['1']['precision'],
        classification_report(y_test, y_pred_knn, output_dict=True)['1']['recall'],
        classification_report(y_test, y_pred_knn, output_dict=True)['1']['f1-score']
    ]
})

print("\nModel Comparison:")
print(comparison)
