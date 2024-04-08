import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import pandas as pd

# Read the datas
df = pd.read_excel('grid_orig.xlsx')

# Update stability column based on power1, power2, and power3 values
df['stability'] = df.apply(lambda row: 'unstable' if row['power1'] == 0 and row['power2'] == 0 and row['power3'] == 0 else row['stability'], axis=1)

# Write the updated dataset back to CSV
df.to_excel('grid_orig.xlsx', index=False)


# Handle missing values
df.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Filtering out erroneous values
df = df[df['date'] != "date"]

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')

# Extract month and hour from the datetime column
df['Month'] = df['date'].dt.month
df['Hour'] = df['date'].dt.hour

##Drop the original datetime column and the 'power generated' column
df.drop(columns=['date'], inplace=True)

# # Assuming 'stability' is the output label column, and other columns are input features
X = df.drop(columns=['stability'])
y = df['stability']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Predicting on the testing set
y_pred = lda.predict(X_test)


# Coefficient vectors for each class
coef_vectors = lda.coef_

# Intercept terms for each class
intercepts = lda.intercept_

# Print the LDA equation for each class
for i, (coef_vector, intercept) in enumerate(zip(coef_vectors, intercepts)):
    print(f"Class {i + 1} LDA equation:")
    equation = " + ".join([f"{coef:.2f} * {feature}" for coef, feature in zip(coef_vector, X.columns)])
    print(f"delta_{i + 1}(x) = {equation} + {intercept:.2f}\n")


df.drop(columns=['stability'], inplace=True)
a=df.cov()
print(a)
# Evaluating the model
accuracy = lda.score(X_test, y_test)
print("Accuracy:", accuracy)

# Compute classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Compute AUC-ROC
auc_roc = roc_auc_score(y_test, lda.predict_proba(X_test)[:, 1])
print("AUC-ROC:", auc_roc)

plt.figure(figsize=(10, 8))
sns.heatmap(df.cov(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Covariance Matrix')
plt.show()

# Plotting the LDA equation coefficients
plt.figure(figsize=(10, 6))
for i, (coef_vector, intercept) in enumerate(zip(lda.coef_, lda.intercept_)):
    plt.bar(X.columns, coef_vector, alpha=0.5, label=f'Class {i+1}')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('LDA Coefficient Values')
plt.legend()
plt.show()


