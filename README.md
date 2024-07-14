This script provides a comprehensive pipeline for using machine learning to train and predict data from an Excel file. Here is an outline of its main functionalities:

1. **Loading Data**: 
    - Loads data from an Excel file using `pandas`.

2. **Handling Missing Values**:
    - Checks for and fills missing values with the mean of each column.

3. **Statistical Analysis**:
    - Computes descriptive statistics and the correlation matrix.
    - Saves statistical descriptions to an Excel file.

4. **Data Visualization**:
    - Generates and saves plots including correlation matrix, box plots, and pair plots using `seaborn` and `matplotlib`.

5. **Data Preprocessing**:
    - Splits the data into training and test sets.
    - Scales the features using either `StandardScaler` or `MinMaxScaler`.

6. **Model Training and Evaluation**:
    - Trains multiple regression models: Linear Regression, Decision Tree, Random Forest, Ridge, Lasso, SVR, KNeighbors, and XGBoost.
    - Evaluates models using R-squared, Mean Squared Error, and Mean Absolute Error.
    - Saves the evaluation results to an Excel file.

7. **Model Performance Visualization**:
    - Generates bar plots for model performance metrics.
    - Plots true vs predicted values, residual plots, and sorted true vs predicted values for each model.
    - Generates and saves a radar chart of R-squared values for comparison.

8. **Prediction on New Data**:
    - Loads new data, scales it, and makes predictions using trained models.
    - Saves the predictions to an Excel file.


### Example Usage:
To use the script:
1. Prepare your data file (`data.xlsx`) with the target variable labeled as "H UAV".
2. Adjust the file paths in the script if necessary.
3. Run the script to generate results, plots, and predictions saved in the `Tables` and `Figures` directories.
