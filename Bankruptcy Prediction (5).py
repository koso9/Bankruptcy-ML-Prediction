#!/usr/bin/env python
# coding: utf-8

# In[13]:


import zipfile
import os
import pandas as pd
from scipy.io import arff  # For handling .arff files

# Step 1: Define file paths
# Path to the zip file containing .arff files
zip_file_path = r"C:\Users\aesic\Downloads\polish+companies+bankruptcy+data (1).zip"

# Directory where .arff files will be extracted
extraction_path = r"C:\Users\aesic\Downloads\polish_data"

# Step 2: Create the extraction directory (if it doesn't exist)
os.makedirs(extraction_path, exist_ok=True)

# Step 3: Unzip the file
# Extract all .arff files from the zip file into the specified directory
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)
    print(f"Files extracted to: {extraction_path}")

# Step 4: Function to load .arff files into Pandas DataFrames
def load_arff_to_dataframe(arff_file_path):
    """
    Converts a .arff file into a Pandas DataFrame.
    
    Args:
        arff_file_path (str): Path to the .arff file.
        
    Returns:
        DataFrame: Loaded data as a Pandas DataFrame.
    """
    data, meta = arff.loadarff(arff_file_path)
    df = pd.DataFrame(data)  # Convert to Pandas DataFrame
    return df

# Step 5: Load all .arff files into a dictionary of DataFrames
dataframes = {}  # Dictionary to hold DataFrames for each year's data

# Loop through each year's .arff file (1year.arff to 5year.arff)
for i in range(1, 6):
    arff_file_path = os.path.join(extraction_path, f'{i}year.arff')
    dataframes[f'{i}year'] = load_arff_to_dataframe(arff_file_path)

# Access the DataFrame for the first year (as an example)
df_1year = dataframes['1year']
print(df_1year.head())  # Print the first few rows of the 1st year's data


# In[14]:


# Step 6: Combine all yearly DataFrames into a single DataFrame
# Assumes all DataFrames have the same columns
combined_df = pd.concat([dataframes['1year'], dataframes['2year'], dataframes['3year'],
                         dataframes['4year'], dataframes['5year']], ignore_index=True)

# Print the shape of the combined DataFrame
print(f"Combined dataset shape: {combined_df.shape}")

# Display the first few rows of the combined dataset for inspection
print(combined_df.head())

# Step 7: Check for missing values in the combined dataset
missing_values = combined_df.isnull().sum()  # Count missing values per column
print("Columns with missing values:")
print(missing_values[missing_values > 0])

# Calculate the percentage of data loss if rows with missing values are dropped
total_rows = combined_df.shape[0]
rows_with_nan = combined_df.dropna().shape[0]
percentage_lost = ((total_rows - rows_with_nan) / total_rows) * 100
print(f"Percentage of data lost by dropping rows with missing values: {percentage_lost:.2f}%")


# In[39]:


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


# In[40]:


from sklearn.impute import SimpleImputer

# Step 10: Identify Columns with Missing Values
# Calculate the number of missing values in each column of the training set
missing_values = X_train.isnull().sum()
print("Columns with missing values:")
print(missing_values[missing_values > 0])

# Step 11: Decide on a Strategy
# Threshold: Drop columns where more than 5% of the rows have missing values
threshold = 0.05 * X_train.shape[0]
cols_to_drop = [col for col in X_train.columns if X_train[col].isnull().sum() > threshold]
cols_to_impute = [col for col in X_train.columns if X_train[col].isnull().sum() <= threshold]

print(f"Columns to drop: {cols_to_drop}")
print(f"Columns to impute: {cols_to_impute}")

# Step 12: Drop Columns with Excessive Missing Values
# Drop these columns from both training and testing sets
X_train_dropped = X_train.drop(columns=cols_to_drop)
X_test_dropped = X_test.drop(columns=cols_to_drop)

# Step 13: Impute Remaining Missing Values
# Use the median strategy to fill in missing values
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_dropped)
X_test_imputed = imputer.transform(X_test_dropped)

# Convert the imputed arrays back into DataFrames
X_train_final = pd.DataFrame(X_train_imputed, columns=X_train_dropped.columns, index=X_train_dropped.index)
X_test_final = pd.DataFrame(X_test_imputed, columns=X_test_dropped.columns, index=X_test_dropped.index)

# Step 14: Verify the Results
print(f"Final Training Set Shape: {X_train_final.shape}")
print(f"Final Test Set Shape: {X_test_final.shape}")
print("\nMissing Values in Final Training Set (should be 0):")
print(X_train_final.isnull().sum().sum())
print("\nMissing Values in Final Test Set (should be 0):")
print(X_test_final.isnull().sum().sum())


# In[41]:


from sklearn.preprocessing import StandardScaler

# Step 15: Initialize the Scaler
scaler = StandardScaler()

# Step 16: Fit the Scaler on the Training Data and Transform Both Training and Testing Sets
# StandardScaler standardizes the data to have a mean of 0 and a standard deviation of 1
X_train_scaled = scaler.fit_transform(X_train_final)  # Fit to training data and scale it
X_test_scaled = scaler.transform(X_test_final)       # Scale the testing data using the same scaler

# Step 17: Convert the Scaled Data Back into DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_final.columns, index=X_train_final.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_final.columns, index=X_test_final.index)

# Step 18: Verify Scaling Results
print("Scaled Training Set (first few rows):")
print(X_train_scaled.head())
print("\nScaled Test Set (first few rows):")
print(X_test_scaled.head())


# In[42]:


from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Verify the class distribution
print("Class distribution before SMOTE:")
print(y_train.value_counts())

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# In[45]:


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


# In[46]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 19: Initialize and Train the Logistic Regression Model
logreg = LogisticRegression(max_iter=1000, random_state=42)  # Increase max_iter to ensure convergence
logreg.fit(X_train_scaled, y_train)

# Step 20: Evaluate the Model on the Test Data
y_pred_logreg = logreg.predict(X_test_scaled)

# Step 21: Print Evaluation Metrics
print("Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))


# In[47]:


from sklearn.ensemble import RandomForestClassifier

# Step 22: Initialize and Train the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Step 23: Evaluate the Model on the Test Data
y_pred_rf = rf_model.predict(X_test_scaled)

# Step 24: Print Evaluation Metrics
print("Random Forest Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# In[49]:


importances = rf_model.feature_importances_
indices = importances.argsort()[::-1]
selected_features = X_train.columns[indices[:10]]

plt.figure(figsize=(10, 6))
plt.barh(selected_features, importances[indices[:10]], color='skyblue')
plt.gca().invert_yaxis()
plt.title("Top 10 Features by Importance")
plt.xlabel("Feature Importance")
plt.show()


# In[50]:


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


# In[52]:


# Step 9: Confusion Matrices
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

