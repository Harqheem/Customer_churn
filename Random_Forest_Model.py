from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import time

start_time = time.time()
# Load Data
df = pd.read_csv('Customer-Churn-Data-cleaned.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# One hot encode values
X = pd.get_dummies(X, drop_first=True)

# Split test and train values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42)

# Values standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Find the best hyperparameters that could be used

#  Best Minimun samples split
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]  
sample_split_accuracy_dict = {}
for min_sample_split in min_samples_split_list:
    model = RandomForestClassifier(min_samples_split=min_sample_split, random_state=42)

    model.fit(X_train_scaled, y_train)
    
    y_predict = model.predict(X_test_scaled)

    accuracy_score_test = accuracy_score(y_test, y_predict)
    sample_split_accuracy_dict[min_sample_split] = accuracy_score_test

best_sample_split = max(sample_split_accuracy_dict, key=sample_split_accuracy_dict.get)
best_sample_split_accuracy = sample_split_accuracy_dict[best_sample_split]


# Best Max Depth
max_depth_list = [2, 4, 8, 16, 32, 64, None]
max_depth_accuracy_dict = {}
for depth in max_depth_list:
    model = RandomForestClassifier(max_depth=depth, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_predict = model.predict(X_test_scaled)

    accuracy_score_test = accuracy_score(y_test, y_predict)
    max_depth_accuracy_dict[depth] = accuracy_score_test

best_max_depth = max(max_depth_accuracy_dict, key=max_depth_accuracy_dict.get)
best_max_depth_accuracy = max_depth_accuracy_dict[best_max_depth]


# Best N Estimator
n_estimator_list = [10, 50, 100, 500]
n_estimator_accuracy_dict = {}
for n_estimator in n_estimator_list:
    model = RandomForestClassifier(n_estimators=n_estimator, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_predict = model.predict(X_test_scaled)

    accuracy_score_test = accuracy_score(y_test, y_predict)
    n_estimator_accuracy_dict[n_estimator] = accuracy_score_test

best_n_estimator = max(n_estimator_accuracy_dict, key=n_estimator_accuracy_dict.get)
best_n_estimator_accuracy = n_estimator_accuracy_dict[best_n_estimator]

print(f"Best Sample Split: {best_sample_split} with accuracy score of {best_sample_split_accuracy:.2%}")
print(f"Best Max Depth: {best_max_depth} with accuracy score of {best_max_depth_accuracy:.2%} ")
print(f"Best N_Estimator: {best_n_estimator} with accuracy score of {best_n_estimator_accuracy:.2%}")


# Model building with best hyperparameters
model = RandomForestClassifier(min_samples_split=best_sample_split, max_depth= best_max_depth, n_estimators=best_n_estimator, random_state=42)
model.fit(X_train_scaled, y_train)

y_predict = model.predict(X_train_scaled)
