import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler 
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from Data_cleaning import lines
from sklearn.model_selection import GridSearchCV
import matplotlib as plt
import time

start_time = time.time()
# Load Data
df = pd.read_csv('Customer-Churn-Data-cleaned.csv')

# Extract the feature and the targets
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# One Hot encode the values
X = pd.get_dummies(X, drop_first=True)

# Split the train and test data
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)

# Standardize the values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find the best Hyperparameter
params = {'C' : [0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio': [0, 1], 'solver': ['liblinear']}
grid = GridSearchCV(estimator=LogisticRegression(max_iter=500, class_weight='balanced'), param_grid=params, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print(F'The best hyperparameter to use: {grid.best_params_}')
lines()

# Train the model
model = grid.best_estimator_
model.fit(X_train_scaled, y_train)

# Get the best bias and weights
w_best = model.coef_
b_best = model.intercept_
print(f'The best weight after gradient descent is {w_best}\n The best bias after gradient descent is {b_best}')
lines()

#Predict the output
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

#Check for the Model's accuracy score
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"Model's accuracy score: {accuracy:.2%}")
print(f"Model's ROC-AUC score: {roc_auc:.4f}")
lines()

end_time = time.time()
training_time = end_time - start_time
print(f'Time for training the model is {training_time}')
print("Classification Report:")
print(classification_report(y_test, y_pred))
lines()