from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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

# Model Training
model = RandomForestClassifier(min_samples_split=2, random_state=42)
model.fit(X_train_scaled, y_train)

# Check for Accuracy
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy score is {accuracy:.2%}')