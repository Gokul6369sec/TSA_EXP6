# Ex.No: 6               HOLT WINTERS METHOD
### Date: 



### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
~~~
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

# ----------------------------------------------------------------------
# Configuration for Housing Price Dataset
# ----------------------------------------------------------------------
file_path = "housing_price_dataset.csv"
target_column = "Price"
date_column = "year" # Note: This is now the 'year' column, not a full date.
# ----------------------------------------------------------------------

# Load the dataset
data = pd.read_csv(file_path)

# 1. Create a suitable time index (Yearly)
# Convert the 'year' column to a full date (e.g., '1969' -> '1969-01-01')
data['Date'] = pd.to_datetime(data[date_column].astype(str) + '-01-01')
data.set_index('Date', inplace=True)

# Use only the Price column and ensure it's numeric
data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
data = data.dropna(subset=[target_column])

# Resample the data to annual frequency (mean of each year)
# 'AS' means Annual Start frequency
yearly_data = data[target_column].resample('AS').mean()

# Split the data into train and test sets (90% train, 10% test)
train_data = yearly_data[:int(0.9 * len(yearly_data))]
test_data = yearly_data[int(0.9 * len(yearly_data)):]

# ----------------------------------------------------------------------
# 2. Model Training and Testing (Holt-Winters without seasonality)
# ----------------------------------------------------------------------

# Holt-Winters model with additive trend and NO seasonality (seasonal=None)
fitted_model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal=None, # Removed seasonality due to annual data
    # seasonal_periods is ignored when seasonal=None
).fit()

# Forecast on the test set
test_predictions = fitted_model.forecast(len(test_data))

# Plot the results
plt.figure(figsize=(12, 8))
train_data.plot(legend=True, label='Train')
test_data.plot(legend=True, label='Test')
test_predictions.plot(legend=True, label='Predicted')
plt.title(f'Train, Test, and Predicted Housing Prices (Holt-Winters Trend Only)')
plt.savefig('housing_test_forecast_plot.png')
plt.close()

# Evaluate model performance
mae = mean_absolute_error(test_data, test_predictions)
mse = mean_squared_error(test_data, test_predictions)
print(f"Mean Absolute Error = {mae:.4f}")
print(f"Mean Squared Error = {mse:.4f}")

# ----------------------------------------------------------------------
# 3. Final Model Fit and Future Forecast
# ----------------------------------------------------------------------

# Fit the model to the entire yearly dataset
final_model = ExponentialSmoothing(
    yearly_data,
    trend='add',
    seasonal=None
).fit()

forecast_steps = 5 # Forecast 5 future years
forecast_predictions = final_model.forecast(steps=forecast_steps)

# Plot the original and forecasted data
plt.figure(figsize=(12, 8))
yearly_data.plot(legend=True, label='Original Annual Data')
forecast_predictions.plot(legend=True, label=f'Forecasted Data ({forecast_steps} years)', color='red')
plt.title(f'Original and Forecasted Housing Prices (Holt-Winters Trend Only)')
plt.savefig('housing_future_forecast_plot.png')
plt.close()
~~~

### OUTPUT:
<img width="885" height="570" alt="image" src="https://github.com/user-attachments/assets/5b35b20b-8fcd-4e29-baad-a97bc9fac7e9" />
<img width="791" height="559" alt="image" src="https://github.com/user-attachments/assets/2698b48e-7455-4cd1-8bac-1fe14f3c08c9" />





### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
