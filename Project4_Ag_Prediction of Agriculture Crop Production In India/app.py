import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import glob

# -----------------------------
# Load trained model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "crop_production_model.pkl")

model = joblib.load(model_path)
# -----------------------------
# Find cleaned CSV file
# -----------------------------
csv_files = glob.glob(os.path.join(BASE_DIR, "cleaned_production*.csv"))

if not csv_files:
    messagebox.showerror(
        "Error",
        "Cleaned production CSV file was not found."
    )
    raise FileNotFoundError("Cleaned production CSV not found.")

data = pd.read_csv(csv_files[0])

# -----------------------------
# Encode Particulars
# -----------------------------
le = LabelEncoder()
le.fit(data["Particulars"])

# -----------------------------
# Prediction function
# -----------------------------
def predict_production():
    try:
        particular = particular_box.get()
        year = int(year_entry.get())

        if particular == "":
            messagebox.showwarning(
                "Input Required",
                "Please select a Particular."
            )
            return

        particular_encoded = le.transform([particular])[0]

        user_input = pd.DataFrame(
            [[particular_encoded, year]],
            columns=["Particulars_encoded", "Year"]
        )

        prediction = model.predict(user_input)

        result_label.config(
            text=f"Predicted Production:\n{prediction[0]:.2f}"
        )

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Please enter a valid year."
        )

    except Exception as e:
        messagebox.showerror(
            "Error",
            str(e)
        )


# -----------------------------
# Create GUI
# -----------------------------
root = tk.Tk()
root.title("Agriculture Crop Production Prediction")
root.geometry("600x450")

title_label = tk.Label(
    root,
    text="Agriculture Crop Production Prediction",
    font=("Arial", 20, "bold")
)
title_label.pack(pady=25)

# Particular
particular_label = tk.Label(
    root,
    text="Select Particular:",
    font=("Arial", 13)
)
particular_label.pack(pady=8)

particular_box = ttk.Combobox(
    root,
    values=list(le.classes_),
    width=60,
    state="readonly"
)
particular_box.pack(pady=5)

# Year
year_label = tk.Label(
    root,
    text="Enter Year:",
    font=("Arial", 13)
)
year_label.pack(pady=8)

year_entry = tk.Entry(
    root,
    font=("Arial", 13),
    width=20
)
year_entry.pack(pady=5)

# Predict button
predict_button = tk.Button(
    root,
    text="PREDICT PRODUCTION",
    font=("Arial", 13, "bold"),
    command=predict_production
)
predict_button.pack(pady=25)

# Result
result_label = tk.Label(
    root,
    text="Predicted Production:\n---",
    font=("Arial", 16, "bold")
)
result_label.pack(pady=15)

# Start application
root.mainloop()