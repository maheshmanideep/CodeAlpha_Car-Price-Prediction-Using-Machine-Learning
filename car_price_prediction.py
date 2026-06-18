import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("car data.csv")

# Display first 5 rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Encode categorical columns
le = LabelEncoder()

categorical_cols = ['Car_Name', 'Fuel_Type',
                    'Selling_type', 'Transmission']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model Training
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Actual vs Predicted Graph
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print(importance)

plt.figure(figsize=(10,5))
plt.bar(importance['Feature'],
        importance['Importance'])
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()