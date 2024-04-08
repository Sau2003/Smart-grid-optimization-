from sklearn.ensemble import RandomForestClassifier

# Assuming X_train and y_train are your training data

# Initialize Random Forest Classifier
rf = RandomForestClassifier()
X_train,y_train,X_test,y_test
# Fit the classifier to your training data
rf.fit(X_train, y_train)

# Get feature importances
feature_importances = rf.feature_importances_

# Sort features by their importance scores
sorted_indices = feature_importances.argsort()[::-1]

# Print the most prominent features
print("Most Prominent Features:")
for index in sorted_indices:
    print(f"{X.columns[index]}: {feature_importances[index]}")
