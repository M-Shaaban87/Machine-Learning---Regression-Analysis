# Program for Using Machine Learning in Training and Predicting any Set of Data in Excel File
# Created by: Dr. Mohammed Shaaban, PhD in Structural Engineering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import seaborn as sns

def load_data(file_path):
    return pd.read_excel(file_path)

if not os.path.exists('Tables'):
        os.makedirs('Tables')

if not os.path.exists('Figures'):
        os.makedirs('Figures')

def check_and_fill_na(data):
    nan_columns = data.columns[data.isna().any()].tolist()
    print("Columns with NaN values:", nan_columns)
    for column in data.columns:
        data[column].fillna(data[column].mean(), inplace=True)
    nan_columns = data.columns[data.isna().any()].tolist()
    if not nan_columns:
        print("All NaN values have been filled.")
    return data

def perform_statistical_analysis(data):
    data_description = data.describe()
    correlation_matrix = data.corr()
    return data_description, correlation_matrix

def visualize_correlation_matrix(correlation_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, square=True)
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join('Figures', 'Correlation Matrix.png'), bbox_inches="tight")
    plt.show()


def save_statistical_analysis(data_description):
    if not os.path.exists('Tables'):
        os.makedirs('Tables')
    data_description.to_excel(os.path.join('Tables', "statistical_analysis.xlsx"))


def visualize_boxplot(data):
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data)
    plt.title("Box Plot")
    plt.savefig(os.path.join('Figures', 'Box Plot.png'), bbox_inches="tight")
    plt.show()


def visualize_pairplot(data):
    sns.pairplot(data=data)
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
    plt.savefig(os.path.join('Figures', "Pair_Plot.png"), bbox_inches="tight")
    plt.show()


def preprocess_data(data, normalization=False):
    X = data.drop("H UAV", axis=1)
    y = data["H UAV"]

    if normalization:
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42), scaler


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor()),
        ("Random Forest", RandomForestRegressor()),
        ("Ridge", Ridge()),
        ("Lasso", Lasso()),
        ("SVR", SVR()),
        ("KNeighborsRegressor", KNeighborsRegressor()),
        ("XGBRegressor", XGBRegressor())
    ]

    results = []
    for model_name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results.append((model_name, r2, mse, mae))
    return models, pd.DataFrame(results, columns=["Model Name", "R-squared (R2)", "Mean Squared Error (MSE)",
                                                  "Mean Absolute Error (MAE)"])


def save_results_to_excel(results_df, file_path):
    results_df.to_excel(file_path, index=False)


def visualize_results(results_df):
    model_names = results_df["Model Name"]
    r2_values = results_df["R-squared (R2)"]
    mse_values = results_df["Mean Squared Error (MSE)"]
    mae_values = results_df["Mean Absolute Error (MAE)"]

    plt.bar(model_names, r2_values)
    plt.xlabel("Model")
    plt.ylabel("R-squared")
    plt.title("R-squared Values for Different Models")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join('Figures', "R-Squared.png"), bbox_inches="tight")
    plt.show()

    plt.bar(model_names, mse_values)
    plt.xlabel("Model")
    plt.ylabel("MSE")
    plt.title("MSE Values for Different Models")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join('Figures', "MSE.png"), bbox_inches="tight")
    plt.show()

    plt.bar(model_names, mae_values)
    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.title("MAE Values for Different Models")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join('Figures', "MAE.png"), bbox_inches="tight")
    plt.show()


def plot_true_vs_predicted(models, X_test, y_test):
    for model_name, model in models:
        y_pred = model.predict(X_test)

        plt.scatter(y_test, y_pred)
        plt.xlabel("True Response")
        plt.ylabel("Predicted Response")
        plt.title(f"{model_name} - True vs. Predicted")
        max_value = max(max(y_test), max(y_pred))
        min_value = min(min(y_test), min(y_pred))
        plt.xlim(min_value - 10, max_value + 10)
        plt.ylim(min_value - 10, max_value + 10)
        plt.plot([min_value - 10, max_value + 10], [min_value - 10, max_value + 10], color='red', linestyle='--')
        plt.savefig(os.path.join('Figures', f"True vs Predicted_{model_name}.png"), bbox_inches="tight")
        plt.show()

        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals)
        plt.xlabel("Predicted Response")
        plt.ylabel("Residuals")
        plt.title(f"{model_name} - Residuals Plot")
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig(os.path.join('Figures', f"Residuals_{model_name}.png"), bbox_inches="tight")
        plt.show()

        results_df = pd.DataFrame({'True Response': y_test, 'Predicted Response': y_pred})
        results_df.sort_values('True Response', inplace=True)
        plt.scatter(results_df.index, results_df['True Response'], label='True Response')
        plt.scatter(results_df.index, results_df['Predicted Response'], label='Predicted Response')
        plt.xlabel("Record Number")
        plt.ylabel("Response Value")
        plt.title(f"{model_name} - True vs. Predicted")
        plt.legend()
        plt.savefig(os.path.join('Figures', f"Record Number_{model_name}.png"), bbox_inches="tight")
        plt.show()


def visualize_radar_chart(results_df, save_path=None):
    labels = results_df["Model Name"].tolist()
    r2_values = results_df["R-squared (R2)"].tolist()

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    r2_values += r2_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.fill(angles, r2_values, color='blue', alpha=0.25)
    ax.plot(angles, r2_values, color='blue', linewidth=2)
    ax.set_yticklabels([])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    for angle, r2_value, label in zip(angles[:-1], r2_values[:-1], labels):
        ax.text(angle, r2_value, f'{r2_value:.2f}', ha='center', va='center')

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()



def predict_new_data(models, scaler, file_path):
    new_data = pd.read_excel(file_path)
    new_data_scaled = scaler.transform(new_data)
    predictions_df = pd.DataFrame(columns=["Model Name", "Predicted"])

    for model_name, model in models:
        predictions = model.predict(new_data_scaled)
        model_predictions_df = pd.DataFrame({
            "Model Name": [model_name] * len(new_data),
            "Predicted": predictions
        })
        predictions_df = pd.concat([predictions_df, model_predictions_df], ignore_index=True)
    return predictions_df


def main():
    # Load data
    file_path = "data.xlsx"  # Replace with your data file path
    data = load_data(file_path)

    # Check and fill NaN values
    data = check_and_fill_na(data)

    # Perform statistical analysis
    data_description, correlation_matrix = perform_statistical_analysis(data)

    # Visualize correlation matrix
    visualize_correlation_matrix(correlation_matrix)

    # Visualize box plot
    visualize_boxplot(data)

    # Save statistical analysis to Excel
    save_statistical_analysis(data_description)

    # Visualize pairplot
    visualize_pairplot(data)

    # Preprocess data with normalization
    (X_train, X_test, y_train, y_test), scaler = preprocess_data(data, normalization=True)

    # Train and evaluate models
    models, results_df = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    # Save results to Excel
    save_results_to_excel(results_df, os.path.join('Tables', "model_results.xlsx"))

    # Visualize results
    visualize_results(results_df)

    # Plot true vs predicted values
    plot_true_vs_predicted(models, X_test, y_test)

    # Visualize radar chart
    visualize_radar_chart(results_df, save_path=os.path.join('Figures', 'Radar Chart.png'))

    # Predict new data (if available)
    new_data_file_path = "test.xlsx"  # Replace with your new data file path
    predictions_df = predict_new_data(models, scaler, new_data_file_path)
    predictions_df.to_excel(os.path.join('Tables', "new_data_predictions.xlsx"), index=False)


if __name__ == "__main__":
    main()

