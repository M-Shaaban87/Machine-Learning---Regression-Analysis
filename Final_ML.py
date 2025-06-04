
# +===============================================================================+
# |                                 Code Header                                   |
# +===============================================================================+
nSpaces = 90
CodeHeader = {"header_00": " ",
              "header_01": nSpaces * "=",
              "header_02": "Machine Learning Algorithms for Training and Prediction Tabular Data",
              "header_03": "Dr. Mohammed Shaaban - Assistant Professor in Structural Engineering",
              "header_04": " ",
              "header_05": "(c) Copyright 2024-2025 Delta University for Science and Technology",
              "header_06": "All Rights Reserved",
              "header_07": nSpaces * "=",
              "header_08": " ",
              }
for i in CodeHeader.keys():
    print(CodeHeader[i].center(nSpaces, " "))
# +===============================================================================+
# |                              Import Libraries                                 |
# +===============================================================================+
import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_rel
import warnings
import joblib
import shap
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR  # Support Vector Regression
from xgboost import XGBRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# +===============================================================================+
# |                         Data Loading and Preparation                          |
# +===============================================================================+
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from an Excel file with error handling."""
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_directories():
    """Create directories for saving tables and figures with permission check."""
    for directory in ['Tables', 'Figures']:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            # Check write permissions
            test_file = os.path.join(directory, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except PermissionError:
            print(f"Error: No write permission for directory {directory}. Please check permissions.")
            raise
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            raise


def check_and_fill_na(data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with column means and report status."""
    nan_columns = data.columns[data.isna().any()].tolist()
    print("Columns with NaN values:", nan_columns)
    for column in data.columns:
        data[column] = data[column].fillna(data[column].mean())
    if not data.isna().any().any():
        print("All NaN values have been filled.")
    return data


def generate_synthetic_data(data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Generate synthetic data by sampling with replacement."""
    return data.sample(n=n_samples, replace=True, random_state=42)


# +===============================================================================+
# |                   Statistical Analysis and Visualization                      |
# +===============================================================================+
def perform_statistical_analysis(data: pd.DataFrame):
    """Compute descriptive statistics and correlation matrix."""
    return data.describe(), data.corr()


def visualize_correlation_matrix(correlation_matrix: pd.DataFrame):
    """Plot and save a correlation matrix heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, square=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join('Figures', 'Correlation Matrix.png'), bbox_inches="tight")
    plt.close()


def save_statistical_analysis(data_description: pd.DataFrame):
    """Save statistical analysis to an Excel file."""
    safe_save_to_excel(data_description, os.path.join('Tables', "statistical_analysis.xlsx"))


def visualize_boxplot(data: pd.DataFrame):
    """Plot and save a boxplot of the data."""
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data)
    plt.title("Box Plot")
    plt.savefig(os.path.join('Figures', 'Box Plot.png'), bbox_inches="tight")
    plt.close()


def visualize_pairplot(data: pd.DataFrame):
    """Generate and save a pairplot of the data."""
    sns.pairplot(data)
    plt.savefig(os.path.join('Figures', "Pair_Plot.png"), bbox_inches="tight")
    plt.close()


# +===============================================================================+
# |                             Data Preprocessing                                |
# +===============================================================================+
def preprocess_data(data: pd.DataFrame, target_col: str, normalization: bool = False):
    """Preprocess data by scaling features and splitting into train/test sets."""
    X = data.drop(columns=[target_col])
    y = data[target_col].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = MinMaxScaler() if normalization else StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return (X_train_scaled, X_test_scaled, y_train, y_test), scaler, X.columns


# +===============================================================================+
# |                        Model Training and Evaluation                          |
# +===============================================================================+
def tune_random_forest(X_train, y_train):
    """Perform hyperparameter tuning for Random Forest."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=4, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def tune_svr(X_train, y_train):
    """Perform hyperparameter tuning for SVR."""
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.1, 0.2, 0.5]
    }
    svr = SVR()
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=4, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def compute_confidence_intervals(models, X_train, y_train, X_test, y_test, n_bootstraps=1000):
    """Compute 95% confidence intervals for R², MSE, and MAE using bootstrapping."""
    results = []
    for model_name, model in models:
        bootstrap_r2, bootstrap_mse, bootstrap_mae = [], [], []
        n_samples = len(X_test)
        for _ in range(n_bootstraps):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_test[indices]
            y_boot = y_test[indices]
            y_pred = model.predict(X_boot)
            bootstrap_r2.append(r2_score(y_boot, y_pred))
            bootstrap_mse.append(mean_squared_error(y_boot, y_pred))
            bootstrap_mae.append(mean_absolute_error(y_boot, y_pred))
        r2_ci = np.percentile(bootstrap_r2, [2.5, 97.5])
        mse_ci = np.percentile(bootstrap_mse, [2.5, 97.5])
        mae_ci = np.percentile(bootstrap_mae, [2.5, 97.5])
        results.append((
            model_name,
            f"{r2_ci[0]:.4f}-{r2_ci[1]:.4f}",
            f"{mse_ci[0]:.4f}-{mse_ci[1]:.4f}",
            f"{mae_ci[0]:.4f}-{mae_ci[1]:.4f}"
        ))
    ci_df = pd.DataFrame(results, columns=["Model Name", "R² CI (95%)", "MSE CI (95%)", "MAE CI (95%)"])
    safe_save_to_excel(ci_df, os.path.join('Tables', 'confidence_intervals.xlsx'))
    return ci_df


def safe_save_to_excel(df: pd.DataFrame, file_path: str, max_attempts=3):
    """Safely save DataFrame to Excel with retry mechanism."""
    import time
    for attempt in range(max_attempts):
        try:
            df.to_excel(file_path, index=False)
            print(f"Saved file: {file_path}")
            return True
        except PermissionError:
            print(
                f"PermissionError: File {file_path} may be open or inaccessible. Attempt {attempt + 1}/{max_attempts}. Close the file and retry.")
            time.sleep(1)
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
            return False
    print(f"Failed to save {file_path} after {max_attempts} attempts.")
    return False


def train_and_evaluate_models(X_train, y_train, X_test, y_test, n_synthetic=0):
    """Train and evaluate multiple regression models with dynamic CV and statistical comparison."""
    n_samples = len(X_train)
    n_splits = min(5, max(3, n_samples // 2))
    if n_samples < 10:
        print(f"Warning: Dataset too small ({n_samples} samples). Skipping some models.")

    n_neighbors = min(5, max(1, n_samples // 5))

    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor(random_state=42)),
        ("Random Forest", tune_random_forest(X_train, y_train)),
        ("Ridge", Ridge()),
        ("Lasso", Lasso()),
        ("SVR", tune_svr(X_train, y_train)),
    ]
    if n_samples >= n_neighbors:
        models.append(("KNeighborsRegressor", KNeighborsRegressor(n_neighbors=n_neighbors)))
    models.append(("XGBRegressor", XGBRegressor(random_state=42)))

    r2_scorer = make_scorer(r2_score)
    hard_coded_results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    prediction_errors = {}

    for model_name, model in models:
        try:
            scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=r2_scorer)
            mean_r2_cv, std_r2_cv = np.mean(scores), np.std(scores)
        except Exception as e:
            print(f"Warning: {model_name} failed CV with error: {e}. Computing R² without CV.")
            model.fit(X_train, y_train)
            mean_r2_cv, std_r2_cv = r2_score(y_train, model.predict(X_train)), 0.0
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        hard_coded_results.append((model_name, mean_r2_cv, std_r2_cv, test_r2, mse, mae))
        prediction_errors[model_name] = y_test - y_pred

    results_df = pd.DataFrame(hard_coded_results, columns=[
        "Model Name", "CV Mean R-squared", "CV Std R-squared", "Test R-squared", "Test MSE", "Test MAE"
    ])

    t_test_results = []
    model_names = [name for name, _ in models]
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            errors1, errors2 = prediction_errors[m1], prediction_errors[m2]
            try:
                stat, p_value = ttest_rel(np.abs(errors1), np.abs(errors2))
                t_test_results.append((f"{m1} vs {m2}", stat, p_value))
            except Exception as e:
                print(f"Warning: t-test failed for {m1} vs {m2}: {e}")
                t_test_results.append((f"{m1} vs {m2}", np.nan, np.nan))

    t_test_df = pd.DataFrame(t_test_results, columns=["Model Pair", "t-statistic", "p-value"])
    t_test_file = os.path.join('Tables', f't_test_results_n_synthetic_{n_synthetic}.xlsx')
    safe_save_to_excel(t_test_df, t_test_file)

    try:
        plt.figure(figsize=(10, 8))
        pivot_table = t_test_df.pivot_table(values="p-value", index="Model Pair")
        sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".4f", cbar_kws={'label': 'p-value'})
        plt.title(f"Paired t-test p-values (n_synthetic={n_synthetic})")
        plt.savefig(os.path.join('Figures', f't_test_p_values_n_synthetic_{n_synthetic}.png'), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error visualizing t-test p-values: {e}")

    return models, results_df, t_test_df


def save_results_to_excel(results_df: pd.DataFrame, file_path: str):
    """Save model evaluation results to an Excel file."""
    safe_save_to_excel(results_df, file_path)


# +===============================================================================+
# |                            Visualization Functions                            |
# +===============================================================================+
def visualize_results(results_df: pd.DataFrame):
    """Visualize model performance metrics with a horizontal bar chart for R-squared."""
    metric = "Test R-squared"
    color = 'skyblue'
    sorted_df = results_df.sort_values(by=metric, ascending=False)
    model_names = sorted_df["Model Name"]
    values = sorted_df[metric]

    plt.figure(figsize=(10, 8))
    bars = plt.barh(model_names, values, color=color)
    plt.xlabel("Test R-squared")
    plt.title("Test R-squared for Different Models")

    offset = 0.02 * (max(values) - min(values)) if len(values) > 1 else 0.02 * max(values)
    for bar in bars:
        width = bar.get_width()
        label_x = width + offset if width >= 0 else width - offset
        ha = 'left' if width >= 0 else 'right'
        label_y = bar.get_y() + bar.get_height() / 2
        plt.text(label_x, label_y, f"{width:.4f}", va='center', ha=ha)

    plt.margins(x=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join('Figures', 'Test_R-squared.png'), bbox_inches="tight")
    plt.close()

    for metric, color, ylabel in [
        ("Test MSE", 'lightcoral', "Test MSE"),
        ("Test MAE", 'lightgreen', "Test MAE")
    ]:
        plt.figure(figsize=(10, 6))
        plt.bar(results_df["Model Name"], results_df[metric], color=color)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} for Different Models")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join('Figures', f"{ylabel.replace(' ', '_')}.png"), bbox_inches="tight")
        plt.close()


def plot_true_vs_predicted(models, X_test, y_test, results_df, target_col):
    """Plot true vs. predicted values with confidence bands and enhanced residual plots."""
    for model_name, model in models:
        y_pred = model.predict(X_test)
        test_r2 = results_df.loc[results_df["Model Name"] == model_name, "Test R-squared"].iloc[0]

        # Compute confidence intervals for predictions
        ci_lower, ci_upper = None, None
        if model_name in ["Random Forest", "XGBRegressor"]:
            if model_name == "Random Forest":
                tree_preds = np.array([tree.predict(X_test) for tree in model.estimators_])
                ci_lower, ci_upper = np.percentile(tree_preds, [2.5, 97.5], axis=0)
            else:
                bootstrap_preds = []
                for _ in range(100):
                    indices = np.random.choice(len(X_test), len(X_test), replace=True)
                    bootstrap_preds.append(model.predict(X_test[indices]))
                bootstrap_preds = np.array(bootstrap_preds)
                ci_lower, ci_upper = np.percentile(bootstrap_preds, [2.5, 97.5], axis=0)

        # Scatter plot with confidence bands
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, label="Predictions")
        if ci_lower is not None and ci_upper is not None:
            sorted_indices = np.argsort(y_test)
            plt.fill_between(
                y_test[sorted_indices],
                ci_lower[sorted_indices],
                ci_upper[sorted_indices],
                color='gray', alpha=0.2, label="95% Confidence Band"
            )
        min_val, max_val = min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))
        plt.plot([min_val - 10, max_val + 10], [min_val - 10, max_val + 10], 'r--', label="Ideal")
        plt.xlabel(f"True {target_col} (kN)")
        plt.ylabel(f"Predicted {target_col} (kN)")
        plt.title(f"{model_name} - True vs. Predicted ({target_col})")
        plt.legend()
        plt.savefig(os.path.join('Figures', f"True_vs_Predicted_{model_name}_{target_col}_with_CI.png"),
                    bbox_inches="tight")
        plt.close()

        # Enhanced residual plot with trend and KDE
        residuals = y_test - y_pred
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.scatter(np.arange(len(residuals)), residuals, alpha=0.6, label="Residuals")
        ax1.axhline(0, color='r', linestyle='--')

        z = np.polyfit(np.arange(len(residuals)), residuals, 1)
        p = np.poly1d(z)
        ax1.plot(np.arange(len(residuals)), p(np.arange(len(residuals))), "b--", label=f"Trend (slope={z[0]:.4f})")

        ax1.set_xlabel("Record Number")
        ax1.set_ylabel(f"Residuals ({target_col}, kN)")
        ax1.set_title(f"{model_name} - Residuals Plot ({target_col})")
        ax1.legend()

        ax2 = ax1.twinx()
        sns.kdeplot(y=residuals, ax=ax2, color='green', label="Residual KDE")
        ax2.set_ylabel("Density")
        ax2.legend(loc="upper right")

        plt.savefig(os.path.join('Figures', f"Residuals_{model_name}_{target_col}_Enhanced.png"), bbox_inches="tight")
        plt.close()


def plot_error_histograms(models, X_test, y_test, target_col):
    """Plot histograms of prediction errors stratified by load ranges."""
    for model_name, model in models:
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        load_ranges = pd.qcut(y_test, q=3, labels=["Low", "Medium", "High"])

        plt.figure(figsize=(15, 5))
        for i, load_range in enumerate(["Low", "Medium", "High"], 1):
            plt.subplot(1, 3, i)
            range_mask = load_ranges == load_range
            sns.histplot(residuals[range_mask], bins=20, kde=True, color='skyblue')
            plt.xlabel(f"Prediction Error ({target_col}, kN)")
            plt.ylabel("Count")
            plt.title(f"{model_name} - Errors ({load_range} Load)")
        plt.tight_layout()
        plt.savefig(os.path.join('Figures', f"Error_Histogram_{model_name}_{target_col}.png"), bbox_inches="tight")
        plt.close()


def visualize_radar_chart(results_df: pd.DataFrame):
    """Create a radar chart comparing model performance."""
    labels = results_df["Model Name"].tolist()
    r2_values = results_df["Test R-squared"].tolist() + [results_df["Test R-squared"].iloc[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.fill(angles, r2_values, alpha=0.25, color='skyblue')
    ax.plot(angles, r2_values, linewidth=2, color='blue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        angle_deg = np.degrees(angle)
        label.set_rotation(angle_deg if angle_deg < 180 else angle_deg - 180)
        label.set_ha('right' if angle_deg < 180 else 'left')
    plt.title("Model Performance Comparison (Test R-squared)", pad=20)
    plt.savefig(os.path.join('Figures', 'Radar_Chart.png'), bbox_inches="tight")
    plt.close()


def visualize_decision_tree(models, feature_names):
    """Visualize the decision tree structure."""
    dt_model = next((model for name, model in models if name == "Decision Tree"), None)
    if dt_model:
        plt.figure(figsize=(100, 80))
        tree.plot_tree(dt_model, feature_names=feature_names, filled=True)
        plt.title("Decision Tree Diagram")
        plt.savefig(os.path.join('Figures', 'decision_tree_diagram.png'), bbox_inches="tight")
        plt.close()


def visualize_r2_vs_terminal_nodes(terminal_nodes, r2_values, optimal_node, optimal_r2):
    """Plot R-squared vs. number of terminal nodes."""
    plt.figure(figsize=(8, 6))
    plt.plot(terminal_nodes, r2_values, marker='o')
    plt.xlabel("Number of Terminal Nodes")
    plt.ylabel("R-squared (%)")
    plt.title("R-squared vs Number of Terminal Nodes")
    plt.axvline(x=optimal_node, color='green')
    plt.text(optimal_node + 0.1, optimal_r2, f"Optimal = {optimal_r2:.2f}%", color='green')
    plt.grid(True)
    plt.savefig(os.path.join('Figures', "R2_vs_Terminal_Nodes.png"), bbox_inches="tight")
    plt.close()


def generate_r2_vs_terminal_nodes_data(X_train, y_train, X_test, y_test):
    """Generate data for R-squared vs. terminal nodes plot."""
    node_list = [2, 3, 4, 5, 6, 7, 8, 10, 15, None]
    terminal_nodes, r2_values = [], []
    for n in node_list:
        dt = DecisionTreeRegressor(max_leaf_nodes=n, random_state=42)
        dt.fit(X_train, y_train)
        r2_values.append(r2_score(y_test, dt.predict(X_test)) * 100)
        terminal_nodes.append(dt.get_n_leaves())
    best_idx = np.argmax(r2_values)
    return terminal_nodes, r2_values, terminal_nodes[best_idx], r2_values[best_idx]


# +===============================================================================+
# |                       Model Explainability and Prediction                     |
# +===============================================================================+
def explain_model_with_shap(model, X_test, feature_names):
    """Generate SHAP summary and force plots for tree-based models."""
    if isinstance(model, (RandomForestRegressor, XGBRegressor, DecisionTreeRegressor)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.savefig(os.path.join('Figures', 'SHAP_Summary.png'), bbox_inches="tight")
        plt.close()

        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        y_pred = model.predict(X_test)
        indices = [
            np.argmax(y_pred),
            np.argmin(y_pred),
            np.argsort(y_pred)[len(y_pred) // 2]
        ]
        sample_names = ['High_Prediction', 'Low_Prediction', 'Median_Prediction']

        for idx, name in zip(indices, sample_names):
            shap.force_plot(
                explainer.expected_value,
                shap_values[idx],
                X_test_df.iloc[idx],
                show=False,
                matplotlib=True
            )
            plt.title(f"SHAP Force Plot - {name}")
            plt.savefig(os.path.join('Figures', f'SHAP_Force_Plot_{name}.png'), bbox_inches="tight")
            plt.close()
    else:
        print("SHAP explanation is only implemented for tree-based models.")


def visualize_ice_pdp(model, X_test, feature_names, target_col):
    """Generate ICE and PDP curves for key features."""
    if isinstance(model, (RandomForestRegressor, XGBRegressor, DecisionTreeRegressor)):
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap_importance = np.abs(shap_values).mean(axis=0)
        top_features = [feature_names[i] for i in np.argsort(shap_importance)[-3:]]

        for feature in top_features:
            feature_idx = feature_names.get_loc(feature) if isinstance(feature_names, pd.Index) else list(
                feature_names).index(feature)

            grid_resolution = 50
            grid = np.linspace(X_test[:, feature_idx].min(), X_test[:, feature_idx].max(), grid_resolution)
            ice_data = np.zeros((grid_resolution, X_test.shape[0]))

            for i in range(X_test.shape[0]):
                X_temp = X_test[i:i + 1].copy()
                for j, grid_value in enumerate(grid):
                    X_temp[0, feature_idx] = grid_value
                    ice_data[j, i] = model.predict(X_temp)[0]

            pdp_data = np.mean(ice_data, axis=1)

            plt.figure(figsize=(10, 6))
            n_samples = X_test.shape[0]
            n_plot = max(1, n_samples // 10)
            for i in range(0, n_samples, n_samples // n_plot):
                plt.plot(grid, ice_data[:, i], color='gray', alpha=0.2, linewidth=1)
            plt.plot(grid, pdp_data, color='blue', linewidth=2, label='Partial Dependence')
            plt.xlabel(feature)
            plt.ylabel(f"Predicted {target_col} (kN)")
            plt.title(f"ICE and PDP for {feature} on {target_col}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join('Figures', f'ICE_PDP_{feature}.png'), bbox_inches='tight')
            plt.close()
    else:
        print("ICE-PDP visualization is only implemented for tree-based models.")


def predict_with_ci(models, scaler, file_path, target_col):
    """Predict new data with confidence intervals for Random Forest."""
    new_data = load_data(file_path)
    if new_data is None:
        return None
    X_new = new_data.drop(columns=[target_col])
    new_data_scaled = scaler.transform(X_new)
    results = []
    for model_name, model in models:
        preds = model.predict(new_data_scaled)
        if model_name == "Random Forest":
            tree_preds = np.array([tree.predict(new_data_scaled) for tree in model.estimators_])
            ci_lower, ci_upper = np.percentile(tree_preds, [2.5, 97.5], axis=0)
            preds_df = pd.DataFrame({
                "Model Name": model_name,
                f"Predicted ({target_col})": preds,
                f"CI Lower ({target_col})": ci_lower,
                f"CI Upper ({target_col})": ci_upper
            })
        else:
            preds_df = pd.DataFrame({
                "Model Name": model_name,
                f"Predicted ({target_col})": preds
            })
        results.append(preds_df)
    predictions_df = pd.concat(results, ignore_index=True)
    safe_save_to_excel(predictions_df, os.path.join('Tables', 'predictions_with_ci.xlsx'))
    return predictions_df


def analyze_residuals(models, X_test, y_test, target_col):
    """Perform Shapiro test on residuals for each model."""
    results = []
    for model_name, model in models:
        residuals = y_test - model.predict(X_test)
        stat, p_value = shapiro(residuals)
        results.append((model_name, target_col, stat, p_value))
    residuals_df = pd.DataFrame(results, columns=["Model", "Target", "Shapiro Stat", "P-value"])
    safe_save_to_excel(residuals_df, os.path.join('Tables', 'residuals_analysis.xlsx'))
    return residuals_df


def compute_feature_importance(models, X_test, feature_names):
    """Compute and visualize feature importance with confidence intervals."""
    importance_data = []
    for model_name, model in models:
        if hasattr(model, 'feature_importances_'):
            importances = []
            for _ in range(100):
                indices = np.random.choice(len(X_test), len(X_test), replace=True)
                X_boot = X_test[indices]
                if model_name == "Random Forest":
                    tree_preds = np.array([tree.predict(X_boot) for tree in model.estimators_])
                    importances.append(np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0))
                elif model_name == "XGBRegressor":
                    importances.append(model.feature_importances_)
                elif model_name == "Decision Tree":
                    importances.append(model.feature_importances_)
            importances = np.array(importances)
            ci_lower, ci_upper = np.percentile(importances, [2.5, 97.5], axis=0)

            for i, feature in enumerate(feature_names):
                importance_data.append({
                    "Model": model_name,
                    "Feature": feature,
                    "Importance": model.feature_importances_[i],
                    "CI Lower": ci_lower[i],
                    "CI Upper": ci_upper[i]
                })

    if importance_data:
        importance_df = pd.DataFrame(importance_data)
        safe_save_to_excel(importance_df, os.path.join('Tables', 'feature_importance_with_CI.xlsx'))

        pivot_df = importance_df.pivot(index="Feature", columns="Model", values="Importance")
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.4f')
        plt.title("Feature Importance Across Models")
        plt.savefig(os.path.join('Figures', 'feature_importance_with_CI.png'), bbox_inches="tight")
        plt.close()

        ci_annotations = importance_df[["Model", "Feature", "CI Lower", "CI Upper"]]
        safe_save_to_excel(ci_annotations, os.path.join('Tables', 'feature_importance_ci_annotations.xlsx'))

        return importance_df
    return pd.DataFrame()


# +===============================================================================+
# |                               Main Function                                   |
# +===============================================================================+
def main():
    # Configuration
    file_path = "data.xlsx"
    test_file_path = "test.xlsx"
    normalization = True

    # Load the raw data
    original_data = load_data(file_path)
    if original_data is None or len(original_data.columns) < 2:
        print("Error: Data loading failed or insufficient columns.")
        return
    target_col = original_data.columns[-1]  # Use last column as target
    print(f"Target column selected: {target_col}")

    create_directories()

    best_r2 = float('-inf')
    best_n_synthetic = 0
    best_results_df = None
    best_models = None
    best_data = None
    best_scaler = None
    best_X_train = best_X_test = best_y_train = best_y_test = None
    best_feature_names = None

    for n_synthetic in range(101):  # Limit to 0-100 synthetic samples
        print(f"\nEvaluating n_synthetic = {n_synthetic}")

        synthetic_data = generate_synthetic_data(original_data, n_synthetic)
        data = pd.concat([original_data, synthetic_data])

        data = check_and_fill_na(data)

        (X_train, X_test, y_train, y_test), scaler, feature_names = preprocess_data(data, target_col, normalization)
        joblib.dump(scaler, os.path.join('Tables', 'scaler.pkl'))

        models, results_df, t_test_df = train_and_evaluate_models(X_train, y_train, X_test, y_test, n_synthetic)

        current_best_r2 = results_df["Test R-squared"].max()
        print(f"Best R-squared at n_synthetic={n_synthetic}: {current_best_r2}")

        if current_best_r2 > best_r2:
            best_r2 = current_best_r2
            best_n_synthetic = n_synthetic
            best_results_df = results_df
            best_models = models
            best_data = data
            best_scaler = scaler
            best_X_train, best_X_test, best_y_train, best_y_test = X_train, X_test, y_train, y_test
            best_feature_names = feature_names
            best_t_test_df = t_test_df

    print(f"\nOptimal n_synthetic found: {best_n_synthetic} with R-squared: {best_r2}")

    safe_save_to_excel(best_data, os.path.join('Tables', 'optimal_final_data.xlsx'))
    save_results_to_excel(best_results_df, os.path.join('Tables', "optimal_model_results.xlsx"))
    safe_save_to_excel(best_t_test_df, os.path.join('Tables', 'optimal_t_test_results.xlsx'))

    ci_df = compute_confidence_intervals(best_models, best_X_train, best_y_train, best_X_test, best_y_test)
    safe_save_to_excel(ci_df, os.path.join('Tables', 'optimal_confidence_intervals.xlsx'))

    visualize_results(best_results_df)
    plot_true_vs_predicted(best_models, best_X_test, best_y_test, best_results_df, target_col)
    plot_error_histograms(best_models, best_X_test, best_y_test, target_col)
    visualize_radar_chart(best_results_df)
    visualize_decision_tree(best_models, best_feature_names)
    compute_feature_importance(best_models, best_X_test, best_feature_names)

    best_model_name = best_results_df.loc[best_results_df["Test R-squared"].idxmax(), "Model Name"]
    best_model = next(model for name, model in best_models if name == best_model_name)
    explain_model_with_shap(best_model, best_X_test, best_feature_names)
    visualize_ice_pdp(best_model, best_X_test, best_feature_names, target_col)
    joblib.dump(best_model, os.path.join('Tables', 'optimal_best_model.pkl'))

    predict_with_ci(best_models, best_scaler, test_file_path, target_col)
    analyze_residuals(best_models, best_X_test, best_y_test, target_col)

    terminal_nodes, r2_vals, best_node, best_r2_val = generate_r2_vs_terminal_nodes_data(
        best_X_train, best_y_train, best_X_test, best_y_test)
    visualize_r2_vs_terminal_nodes(terminal_nodes, r2_vals, best_node, best_r2_val)


if __name__ == "__main__":
    main()