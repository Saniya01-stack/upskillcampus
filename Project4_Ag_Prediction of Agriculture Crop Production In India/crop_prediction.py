import pandas as pd

files = [
    "datafile.csv",
    "datafile (1).csv",
    "datafile (2).csv",
    "datafile (3).csv",
    "produce.csv"
]

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", file)
    print("=" * 60)

    data = pd.read_csv(file)

    print("Shape:", data.shape)
    print("Columns:")
    print(list(data.columns))

    print("\nMissing values:")
    print(data.isnull().sum())
    # Check duplicate rows
print("\nDuplicate rows:", data.duplicated().sum())

# Remove duplicate rows
data = data.drop_duplicates()

print("Shape after removing duplicates:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())
print("\nData types:")
print(data.dtypes)

print("\nSummary statistics:")
print(data.describe())
print("\nColumns in dataset:")
print(data.columns.tolist())
print("\nData types:")
print(data.dtypes)
print("\nUnique values in Particulars:")
print(data["Particulars"].unique())

print("\nUnique values in Frequency:")
print(data["Frequency"].unique())

print("\nUnique values in Unit:")
print(data["Unit"].unique())
# Remove extra spaces from text columns
data["Particulars"] = data["Particulars"].str.strip()
data["Frequency"] = data["Frequency"].str.strip()
data["Unit"] = data["Unit"].str.strip()

print("\nText data cleaned successfully.")
# Convert year columns to numeric values
year_columns = [col for col in data.columns if "3-" in str(col)]

print("\nYear columns found:")
print(year_columns)

for col in year_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

print("\nData types after conversion:")
print(data[year_columns].dtypes)

print("\nFinal dataset shape:")
print(data.shape)

# Save the cleaned dataset
data.to_csv("cleaned_production.csv", index=False)

print("\nCleaned dataset saved successfully!")
# Display final cleaned dataset
print("\nFinal cleaned dataset:")
print(data.head())

print("\nFinal columns:")
print(data.columns.tolist())
# Prepare data for machine learning

# Convert the yearly columns into rows
year_columns = [col for col in data.columns if "3-" in str(col)]

long_data = data.melt(
    id_vars=["Particulars", "Frequency", "Unit"],
    value_vars=year_columns,
    var_name="Year",
    value_name="Production"
)

print("\nData prepared for machine learning:")
print(long_data.head())

print("\nShape of ML dataset:")
print(long_data.shape)

print("\nMissing values in Production:")
print(long_data["Production"].isna().sum())
# Remove rows where production value is missing
ml_data = long_data.dropna(subset=['Production']).copy()

print("Shape after removing missing values:", ml_data.shape)

# Convert Year to numeric
ml_data['Year'] = ml_data['Year'].str.extract(r'(\d{4})').astype(int)

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

ml_data['Particulars_encoded'] = le.fit_transform(ml_data['Particulars'])

# Features and target
X = ml_data[['Particulars_encoded', 'Year']]
y = ml_data['Production']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model trained successfully!")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
# Make a sample prediction

sample = X_test.iloc[[0]]
prediction = model.predict(sample)

print("\nSample Prediction:")
print("Predicted Production:", prediction[0])
# User input prediction
print("\n--- Crop Production Prediction ---")

print("\nAvailable Particulars:")
for i, item in enumerate(le.classes_):
    print(i, ":", item)

choice = int(input("\nEnter the number of the Particulars: "))

selected_particular = le.classes_[choice]

year_input = int(input("Enter the year: "))

particular_encoded = le.transform([selected_particular])[0]

user_input = pd.DataFrame(
    [[particular_encoded, year_input]],
    columns=['Particulars_encoded', 'Year']
)

prediction = model.predict(user_input)

print("\nPredicted Production:", prediction[0])



import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.title("Actual vs Predicted Crop Production")
plt.show()

import joblib
joblib.dump(model, "crop_production_model.pkl")
print("Model saved successfully!")
