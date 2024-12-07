# Machine Learning Project: Predictive Sales Analytics with Walmart Data
![cover](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/cover_final.png)

# Table of Contents

- [Introduction](#introduction)  
- [Dataset Understanding](#dataset-understanding)  
- [Data Loading and Inspection](#data-loading-and-inspection)  
- [Data Preprocessing](#data-preprocessing)  
   - [Stationarity Check](#stationarity-check)  
   - [Correcting Date Formats](#correcting-date-formats)  
   - [Feature Engineering](#feature-engineering)  
   - [Handling Missing Data](#handling-missing-data)  
   - [Encoding and Scaling](#encoding-and-scaling)  
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
   - [Decomposition](#decomposition)  
   - [Holiday Impact Analysis](#holiday-impact-analysis)  
   - [Correlation Analysis](#correlation-analysis)  
- [Data Preparation for Modeling](#data-preparation-for-modeling)  
   - [Train-Test Split](#train-test-split)  
   - [Feature Selection](#feature-selection)  
- [Model Selection and Training](#model-selection-and-training)  
   - [Random Forest Regressor](#random-forest-regressor)  
   - [ARIMA Model](#arima-model)  
- [Model Evaluation](#model-evaluation)  
   - [Weighted MAE](#weighted-mae)  
   - [RMSE](#rmse)  
   - [MAPE](#mape)  
- [Visualizations and Residual Analysis](#visualizations-and-residual-analysis)  
- [Insights and Recommendations](#insights-and-recommendations)  


## Introduction

This project focuses on building a predictive model for Walmart sales forecasting using historical data to predict future sales performance. By leveraging machine learning techniques, including Random Forest and ARIMA models, the project aims to predict sales trends, seasonal fluctuations, and assesses how sales are being affected with temperature and fuel price. The insights gained from this analysis can assist in inventory planning, demand forecasting, and supply chain optimization, ultimately contributing to more efficient operations and improved sales strategies for businesses in retail.

## Dataset Understanding
Find the dataset from its source on Kaggle: 'https://www.kaggle.com/datasets/mikhail1681/walmart-sales'

**Columns**
Date: *Sales week start date*
Weekly_Sales: *Sales*
Holiday_Flag: *Mark on the presence or absence of a holiday*
Temperature: *Air temperature in the region*
Fuel_Price: *Fuel cost in the region*
CPI: *Consumer price index*
Unemployment: *Unemployment rate.*

Disclaimer: *The model performs reasonably well but has room for improvement because the prediction error is significantly relative to the data's context.*

## Data Loading and Inspection
 Importing necessary Python Libraries
 ```python
import pandas as pd
import matplotlib.pyplot as plt
```
Loding the Dataset
```python
data = pd.read_csv('/kaggle/input/walmart-sales/Walmart_Sales.csv')  
data
```
![Data](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/data_preview.png)

Inspecting the data
```python
# Inspect the columns, data types, and initial statistics
print(data.info())
print(data.describe())
```
![Inspection](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/data_info.png)
![Describe](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/data_describe.png)

Key Featues
```python
print(data[['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].head())
```
![Key Features](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/key_features.png)

## 4: Data Preprocessing
### Stationarity Check
```python
# Check for stationarity using Augmented Dickey-Fuller test
def check_stationarity(series):
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print("Series is non-stationary")
    else:
        print("Series is stationary")

check_stationarity(data['Weekly_Sales'])

# Apply transformations if non-stationary
data['Weekly_Sales_diff'] = data['Weekly_Sales'].diff().dropna()
```
![adf](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/adf.png)

### Correcting Date Formats

```python
# Correct Date Format, Handle Mixed or Unknown Formats
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
print(data[data['Date'].isna()])
```
![Correct formats](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/handling_mixed_formarts.png)

### Feature Engineering

```python
# Create new time-based features
data['Date'] = pd.to_datetime(data['Date'])
data['WeekOfYear'] = data['Date'].dt.isocalendar().week
data['Month'] = data['Date'].dt.month
data['Season'] = data['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else ('Spring' if x in [3, 4, 5] else ('Summer' if x in [6, 7, 8] else 'Fall')))

# Expand Holiday_Flag to specific holiday flags (example shown with placeholders)
data['Super_Bowl'] = data['Holiday_Flag'] & (data['Date'].isin(['YYYY-MM-DD']))  # Replace with actual Super Bowl dates
data['Labor_Day'] = data['Holiday_Flag'] & (data['Date'].isin(['YYYY-MM-DD']))  # Replace with actual Labor Day dates

# Generate rolling averages and lagged features for Weekly_Sales
data['Weekly_Sales_Rolling'] = data['Weekly_Sales'].rolling(window=4).mean()
data['Weekly_Sales_Lag1'] = data['Weekly_Sales'].shift(1)

data
```
![Feature Eng](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/feature_eng.png)

### Handling Missing Data

```python
data.dropna(inplace=True)
```
### Encoding and Scaling

Encoding Categorical features
```python
data = pd.get_dummies(data, columns=['Holiday_Flag', 'Season'], drop_first=True)

```
Scaling Continuous Features
```python
from sklearn.preprocessing import StandardScaler

# Scale continuous features
scaler = StandardScaler()
data[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']] = scaler.fit_transform(data[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']])
```

## 5: Exploratory Data Analysis (EDA)
### Decomposition

```python
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data['Weekly_Sales'], model='additive', period=52)

# Set the figure size and plot
plt.figure(figsize=(14, 8))  # Adjust width and height as needed
decomposition.plot()
plt.show()
```
![Decompostion](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/decomposition.png)

### Holiday Impact Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create a new column for holidays (Super Bowl or Labor Day)
data['Holiday'] = data['Super_Bowl'].fillna(0).astype(int) + data['Labor_Day'].fillna(0).astype(int)

# Plotting the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Holiday', y='Weekly_Sales', data=data)

# Add holiday labels for better understanding (optional)
holiday_labels = {0: 'No Holiday', 1: 'Super Bowl', 2: 'Labor Day'}
plt.xticks(ticks=[0, 1, 2], labels=[holiday_labels[i] for i in [0, 1, 2]])

plt.title("Impact of Holidays on Weekly Sales")
plt.show()
```
![Holiday analysis](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/holidays_on_sales.png)

### Correlation Analysis
```python
# Correlation analysis
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```
![Heatmap](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/correlation.png)

## 6. Data Preparation for Modeling
### Train-Test Split

```python
# setting the date as index
data.index = pd.to_datetime(data.index)

# Sort the data by index (which is the 'Date')
data = data.sort_index()

# Calculate the split index for 80/20 split
split_index = int(len(data) * 0.8)

# Split the data into train and test based on the split index
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]
```
```python
# Handling Non-Stationarity (for both train and test datasets)
train_data['Weekly_Sales_diff'] = train_data['Weekly_Sales'].diff()
test_data['Weekly_Sales_diff'] = test_data['Weekly_Sales'].diff()

# Drop NaN values that result from the difference operation (only on the diff columns)
train_data.dropna(subset=['Weekly_Sales_diff'], inplace=True)
test_data.dropna(subset=['Weekly_Sales_diff'], inplace=True)

# Feature selection based on feature importance from the initial model
features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Super_Bowl', 'Labor_Day', 'Weekly_Sales_Lag1', 'Weekly_Sales_Rolling']
target = 'Weekly_Sales'
```
### Feature Selection

Using Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots to identify the ARIMA(p, d, q) parameters.
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data['Weekly_Sales_diff'].dropna(), lags=30)
plt.show()
```
![Autocorrelation](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/autocorrelation.png)
```python
plot_pacf(data['Weekly_Sales_diff'].dropna(), lags=30)
plt.show()
```

![Faeture selection](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/partial_autocorelation.png)


Time Series
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(data['Weekly_Sales_diff'], label='Differenced Weekly Sales')
plt.title('Stationary Time Series')
plt.legend()
plt.show()
```
![Time series](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/stationary_time_series.png)

Validate and clean data
```python
print(data['Date'].unique())
```
![Validate](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/validate_data.png)

Infer Format Dynamically
```python
data['Date'] = pd.to_datetime(data['Date'], format='mixed', errors='coerce')
print(data['Date'].head())
print(data['Date'].dtype)
```
![infer](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/infer_format.png)

Removing nulls brought about by lagging and rolling action

```python
data = data.dropna()
```


## 7. Model Selection and Training
### Random Forest Regressor

Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(train_data[features], train_data[target])
```
```python
# Predict and calculate error
predictions = rf_model.predict(test_data[features])
print("MAE:", mean_absolute_error(test_data[target], predictions))
```
![Model Selection and training](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/model_selection_and_train.png)

### ARIMA Model
```python
from statsmodels.tsa.arima.model import ARIMA

# Train ARIMA model
arima_model = ARIMA(train_data['Weekly_Sales_diff'].dropna(), order=(1, 1, 1))
arima_results = arima_model.fit()
print(arima_results.summary())
```
![ARIMA](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/arima.png)

## 8. Model Evaluation 
### Weighted MAE
```python
import numpy as np

# Calculate WMAE
def weighted_mae(y_true, y_pred, weights):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

holiday_weeks = (test_data['Labor_Day'] == 1) & (test_data['Super_Bowl'] == 1)
weights = np.where(holiday_weeks, 5, 1)  # Assign higher weights to holiday weeks
wmae = weighted_mae(test_data[target], predictions, weights)
print("WMAE:", wmae)
```
![evaluation](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/wmae.png)

Additional Evaluation Metrics

### RMSE
```python
from sklearn.metrics import mean_squared_error

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data[target], predictions))
print("RMSE:", rmse)
```
![RMSE](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/RMSE.png)

### MAPE
```python
# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((test_data[target] - predictions) / test_data[target])) * 100
print(f'MAPE: {mape:.2f}%')
```
![MAPE](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/MAPE.png)

Visualising the Model
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Creating a DataFrame for actual and predicted sales
results = pd.DataFrame({'Actual': test_data[target], 'Predicted': predictions})

# Plotting the actual vs predicted sales
plt.figure(figsize=(12, 6))
sns.lineplot(data=results, palette="tab10", linewidth=2)
plt.title('Actual vs Predicted Weekly Sales')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.legend(['Actual', 'Predicted'])
plt.show()
```
![visual model](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/actual_vs_pred.png)

Residuals
```python
# Calculate residuals (actual - predicted)
residuals = results['Actual'] - results['Predicted']

# Plot the residuals
plt.figure(figsize=(12, 6))
sns.lineplot(x=test_data.index, y=residuals, color='red', linewidth=2)
plt.title('Residuals: Actual vs Predicted Weekly Sales')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.axhline(0, color='black', linestyle='--')  # Line at 0 for reference
plt.show()
```

## Visualizations

Dashboard Screenshot

![Dash](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/Screenshot%202024-11-18%20191444.png)


Download the dashboard file here: https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/docs/Walmart_Stores_Sales_Report.pbix

## Insights and Recommendations

Error Analysis: *Examine residuals for unexplained patterns.*

Advanced Features: *Add interaction terms and external variables.*

Model Refinements: *Use ensemble models or advanced hyperparameter tuning.*

External Data: *Incorporate weather, competitor data, and economic trends.*

Promotional Data: *This analysis would have even been more than this but for some reason I did not promotional campaigns or pricing data for that part of the analysis.*

