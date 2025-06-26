import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
import numpy as np
import sys
import os

# For PyInstaller compatibility
def resource_path(relative_path):
    """Get absolute path to resource (for PyInstaller .exe or normal run)."""
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load model and scaler
model = joblib.load(resource_path('Tables/optimal_best_model.pkl'))
scaler = joblib.load(resource_path('Tables/scaler.pkl'))

# Load feature names, ranges, and detect target from Excel
def load_features_and_ranges(data_file):
    data = pd.read_excel(data_file)
    columns = data.columns.tolist()
    target_col = columns[-1]  # Last column as target
    feature_names = columns[:-1]
    feature_ranges = {
        feature: (data[feature].min(), data[feature].max()) for feature in feature_names
    }
    return feature_names, feature_ranges, target_col

feature_names, feature_ranges, target_col = load_features_and_ranges(resource_path('new_data.xlsx'))

# GUI Application
class PredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Prediction GUI")

        self.entries = {}
        for idx, feature in enumerate(feature_names):
            # Feature label
            tk.Label(master, text=feature).grid(row=idx, column=0, padx=10, pady=5, sticky='w')

            # Entry box
            entry = tk.Entry(master)
            entry.grid(row=idx, column=1, padx=10, pady=5)
            self.entries[feature] = entry

            # Min/Max comment (grey)
            min_val, max_val = feature_ranges[feature]
            comment_text = f"(Enter between {min_val:.2f} - {max_val:.2f})"
            tk.Label(master, text=comment_text, fg='grey').grid(row=idx, column=2, padx=10, pady=5, sticky='w')

        # Predict button
        self.predict_button = tk.Button(master, text="Predict !!", command=self.predict)
        self.predict_button.grid(row=len(feature_names), column=0, columnspan=3, pady=10)

        # Prediction result
        self.result_label = tk.Label(master, text="", font=('Palatino Linotype', 12, 'bold'))
        self.result_label.grid(row=len(feature_names)+1, column=0, columnspan=3, pady=10)

    def predict(self):
        try:
            input_values = []
            for feature in feature_names:
                value = float(self.entries[feature].get())
                min_val, max_val = feature_ranges[feature]
                if not (min_val <= value <= max_val):
                    messagebox.showwarning(
                        "Input Out of Range",
                        f"{feature} must be between {min_val:.2f} and {max_val:.2f}."
                    )
                    return
                input_values.append(value)

            input_array = np.array(input_values).reshape(1, -1)

            # Scale and predict
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)

            self.result_label.config(text=f"Predicted {target_col}: {prediction[0]:.4f}")

        except ValueError:
            messagebox.showerror("Invalid input", "Please enter numeric values only.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
