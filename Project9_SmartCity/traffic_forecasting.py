import pandas as pd

# Load dataset
train = pd.read_csv("train_aWnotuB.csv")

# Convert DateTime column into proper date/time format
train["DateTime"] = pd.to_datetime(train["DateTime"])

# Sort data by date and time
train = train.sort_values("DateTime")

print("Data prepared successfully!")
print(train.head())

print("\nData types:")
print(train.dtypes)

print("\nMissing values:")
print(train.isnull().sum())
# Feature Engineering
train["Hour"] = train["DateTime"].dt.hour
train["Day"] = train["DateTime"].dt.day
train["Month"] = train["DateTime"].dt.month
train["Year"] = train["DateTime"].dt.year
train["DayOfWeek"] = train["DateTime"].dt.dayofweek

# Weekend indicator
train["IsWeekend"] = (train["DayOfWeek"] >= 5).astype(int)

print("\nFeatures created successfully!")
print(train.head())
# Prepare data for Machine Learning

# Features used for prediction
features = [
    "Junction",
    "Hour",
    "Day",
    "Month",
    "Year",
    "DayOfWeek",
    "IsWeekend"
]

X = train[features]
y = train["Vehicles"]

print("\nMachine Learning data prepared!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Prepare features and target
X = train[features]
y = train["Vehicles"]

print("\nMachine Learning data prepared!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nData split successfully!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

print("Model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Show sample predictions
print("\nSample predictions:")
for actual, predicted in zip(y_test.iloc[:10], y_pred[:10]):
    print("Actual:", actual, "Predicted:", round(predicted, 2))
    import joblib

# Save the trained model
joblib.dump(model, "traffic_model.pkl")

print("\nModel saved successfully as traffic_model.pkl!")
# Load test dataset
test = pd.read_csv("datasets_8494_11879_test_BdBKkAj.csv")

# Convert DateTime column
test["DateTime"] = pd.to_datetime(test["DateTime"])

# Create the same features as training data
test["Hour"] = test["DateTime"].dt.hour
test["Day"] = test["DateTime"].dt.day
test["Month"] = test["DateTime"].dt.month
test["Year"] = test["DateTime"].dt.year
test["DayOfWeek"] = test["DateTime"].dt.dayofweek
test["IsWeekend"] = (test["DayOfWeek"] >= 5).astype(int)

# Select features
X_final = test[features]

# Predict vehicle count
predictions = model.predict(X_final)

# Add predictions to test dataset
test["Predicted_Vehicles"] = predictions

# Display results
print("\nTRAFFIC PREDICTIONS")
print(test[["DateTime", "Junction", "Predicted_Vehicles"]].head(10))
# Save predictions to a CSV file

output = test[["DateTime", "Junction", "Predicted_Vehicles"]]

output.to_csv("traffic_predictions.csv", index=False)

print("\nPredictions saved successfully as traffic_predictions.csv!")
print(output.head())
import matplotlib.pyplot as plt

# Create Actual vs Predicted graph
plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

plt.xlabel("Actual Vehicle Count")
plt.ylabel("Predicted Vehicle Count")
plt.title("Actual vs Predicted Vehicle Count")

# Reference line for perfect predictions
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.show()
import matplotlib.pyplot as plt

# Get feature importance
importance = model.feature_importances_

# Create graph
plt.figure(figsize=(8, 5))
plt.bar(features, importance)

plt.title("Feature Importance for Traffic Prediction")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# ==========================================
# CUSTOM TRAFFIC PREDICTION
# ==========================================

print("\n--- SMART CITY TRAFFIC PREDICTION ---")

# Take input from user
junction = int(input("Enter Junction number: "))
date_input = input("Enter Date (YYYY-MM-DD): ")
time_input = input("Enter Time (HH:MM): ")

# Combine date and time
input_datetime = pd.to_datetime(date_input + " " + time_input)

# Create features
hour = input_datetime.hour
day = input_datetime.day
month = input_datetime.month
year = input_datetime.year
day_of_week = input_datetime.dayofweek
is_weekend = int(day_of_week >= 5)

# Create input data
custom_data = pd.DataFrame([{
    "Junction": junction,
    "Hour": hour,
    "Day": day,
    "Month": month,
    "Year": year,
    "DayOfWeek": day_of_week,
    "IsWeekend": is_weekend
}])

# Predict traffic
prediction = model.predict(custom_data)

print("\n🚦 TRAFFIC PREDICTION RESULT")
print("--------------------------------")
print("Date & Time:", input_datetime)
print("Junction:", junction)
print("Predicted Number of Vehicles:", round(prediction[0]))
