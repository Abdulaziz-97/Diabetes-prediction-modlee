import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# data set
file_path = r"C:\Users\Azooo\Downloads\diabetes\diabetes.csv"
data = pd.read_csv(file_path)

#Target column is the column we want to predicte for the test data
target_column = 'Outcome'

# Handle missing values/bcz the Logistic reggrsion can't work with missing data
imputer = SimpleImputer(strategy='mean')
X = data.drop(target_column, axis=1)
X = imputer.fit_transform(X)
y = data[target_column]

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# T/T split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model 
model = LogisticRegression(max_iter=100)  
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Implement cross-validation/ i tried to implemnt cross-validation to imorove the accuracy but it didn't make any diffrance
cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')

print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {np.mean(cv_scores)}')
