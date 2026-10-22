from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

print('Training Starts...')
start_time = time.time()
# Load Data
df = pd.read_csv('Customer-Churn-Data-cleaned.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# One hot encode values
X = pd.get_dummies(X, drop_first=True)

# Split test and train values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42)


# Find the best hyperparameters that could be used
params = {'min_samples_split' : [2,10, 30, 50, 100, 200, 300, 700] , 'max_depth': [2, 4, 8, 16, 32, 64, None], 'n_estimators': [10, 50, 100, 500]}
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),param_distributions=params, n_iter=20, cv=5, scoring="roc_auc", n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Train the model
model = random_search.best_estimator_

# Model Prediction
y_pred = model.predict(X_test)

print(f"Test accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

end_time = time.time()
training_time = end_time - start_time
print('Training ends...')
print(f"Training Time {training_time:.1f}s")