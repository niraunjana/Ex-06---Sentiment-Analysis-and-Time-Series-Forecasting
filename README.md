# Ex-06---Sentiment-Analysis-and-Time-Series-Forecasting

## A: Sentiment Analysis

## Aim :
To analyze textual data and derive sentiment scores using Python's NLTK and TextBlob libraries.

## Tools Required :
1. Python Programming Language
2. Libraries:
- NLTK (Natural Language Toolkit)
- TextBlob

## Procedure :

### Step 1: Install Required Libraries
    - Install NLTK and TextBlob using pip and download additional resources like punkt, stopwords, and vader_lexicon.

### Step 2: Prepare Sample Text
    - Choose or input a sample text for sentiment analysis.

### Step 3: Analyze Sentiment with TextBlob

### Step 4: Create a TextBlob object with the sample text.
    - Extract Polarity (sentiment strength) and Subjectivity (level of opinion).
    - Set Up NLTK VADER Analyzer

### Step 5: Import NLTK’s SentimentIntensityAnalyzer.
    - Initialize the analyzer and analyze the sentiment of the text.
    - Extract Sentiment Scores Using NLTK (VADER)
    - Obtain Positive, Neutral, Negative, and Compound scores for the text.

### Step 6: Compare Results
    - Compare the results from TextBlob and VADER to assess similarities and differences in sentiment interpretation.

## Program :

```
# Install required libraries
# Run this in the terminal if not already installed:
# pip install nltk textblob

# Import necessary libraries
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Sample text for sentiment analysis
text = "I love programming in Python. It is such a powerful and enjoyable language!"

# --- Sentiment Analysis using TextBlob ---
print("Sentiment Analysis using TextBlob:")
blob = TextBlob(text)
sentiment_blob = blob.sentiment
print("Polarity:", sentiment_blob.polarity)  # -1 (negative) to 1 (positive)
print("Subjectivity:", sentiment_blob.subjectivity)  # 0 (objective) to 1 (subjective)

# --- Sentiment Analysis using NLTK VADER ---
print("\nSentiment Analysis using NLTK (VADER):")
sia = SentimentIntensityAnalyzer()
sentiment_vader = sia.polarity_scores(text)
print("Positive:", sentiment_vader['pos'])
print("Neutral:", sentiment_vader['neu'])
print("Negative:", sentiment_vader['neg'])
print("Compound:", sentiment_vader['compound'])  # -1 (negative) to 1 (positive)

# --- Comparison of Results ---
print("\nComparison of Results:")
print(f"TextBlob Polarity: {sentiment_blob.polarity} (Positive if > 0, Negative if < 0)")
print(f"VADER Compound: {sentiment_vader['compound']} (Positive if > 0.05, Negative if < -0.05)")

```
## Outputs :

### TextBlob Output:

![image](https://github.com/user-attachments/assets/5165a057-7a70-4aaf-9baa-fe1f25e4a39a)

### NLTK VADER Output :

![image](https://github.com/user-attachments/assets/1feb557d-7db7-4ff2-8dc2-3275985a6e97)

## Result :
The sentiment analysis outputs indicate that the sample text has a strong positive sentiment, as evidenced by the polarity (TextBlob) and compound score (VADER).



## B: Time Series Forecasting with ARIMA

## Aim :
To forecast future trends in time series data using the ARIMA model and evaluate forecast accuracy.


## Tools Required :
1. Python Programming Language
2. Libraries:
- Statsmodels
- Pandas
- Matplotlib
- NumPy
- Scikit-learn

## Procedure :

### Step 1: Install Required Libraries
- Install Statsmodels, Pandas, Matplotlib, and other necessary packages using pip.

### Step 2: Prepare Time Series Data
- Generate or input time series data (e.g., sales, stock prices) and visualize it using Matplotlib.

### Step 3: Split the Data
- Divide the dataset into training (80%) and testing (20%) sets.

### Step 4: Fit ARIMA Model
- Use the ARIMA function from Statsmodels to build and train the model with specified parameters (p, d, q).

### Step 5: Forecast Future Values
- Predict future values for the test period and visualize the forecast alongside the original data.

### Step 6: Evaluate Model Performance
- Calculate and analyze evaluation metrics like MAE, MSE, and RMSE to measure forecast accuracy.

## Program :

```
# Install required libraries
# Run this in the terminal if not already installed:
# pip install pandas matplotlib statsmodels

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Step 1: Load the Dataset
# Replace 'time_series.csv' with your dataset containing a time series
# Ensure the file has a column named 'Date' and 'Value'
data = pd.read_csv('time_series.csv', parse_dates=['Date'], index_col='Date')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Plot the original time series
plt.figure(figsize=(10, 6))
plt.plot(data, label='Original Time Series', color='blue')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Step 2: Check Stationarity of the Time Series
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print("\nResults of Augmented Dickey-Fuller Test:")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] <= 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is not stationary.")

check_stationarity(data['Value'])

# Step 3: Differencing to Make the Series Stationary (if necessary)
data_diff = data.diff().dropna()
plt.figure(figsize=(10, 6))
plt.plot(data_diff, label='Differenced Time Series', color='orange')
plt.title('Differenced Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Re-check stationarity after differencing
check_stationarity(data_diff['Value'])

# Step 4: Fit the ARIMA Model
# Define the ARIMA order (p, d, q)
# p: autoregressive terms, d: differencing, q: moving average terms
p, d, q = 1, 1, 1  # Adjust based on data and analysis

# Fit the ARIMA model
model = ARIMA(data['Value'], order=(p, d, q))
model_fit = model.fit()

# Print the model summary
print("\nARIMA Model Summary:")
print(model_fit.summary())

# Step 5: Forecast Future Values
# Forecast for the next 10 periods
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='D')[1:]

# Create a forecast DataFrame
forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)

# Plot the original data and forecast
plt.figure(figsize=(10, 6))
plt.plot(data, label='Original Time Series', color='blue')
plt.plot(forecast_df, label='Forecast', color='red')
plt.title('Time Series Forecasting with ARIMA')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Step 6: Evaluate the Model (Optional)
# Split data into training and test sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fit the ARIMA model on the training set
model_train = ARIMA(train['Value'], order=(p, d, q))
model_train_fit = model_train.fit()

# Forecast on the test set
test_forecast = model_train_fit.forecast(steps=len(test))

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test, test_forecast)
print(f"\nMean Squared Error (MSE) on Test Set: {mse}")

# Plot training, test, and forecast
plt.figure(figsize=(10, 6))
plt.plot(train, label='Training Data', color='blue')
plt.plot(test, label='Test Data', color='orange')
plt.plot(test.index, test_forecast, label='Forecast on Test Data', color='green')
plt.title('ARIMA Model - Training vs Test Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

```
## Outputs :

### Initial Dataset Preview:

![image](https://github.com/user-attachments/assets/14c7dcc9-5747-4263-812a-20ea8bea3a35)


### Stationarity Test (ADF Test):

![image](https://github.com/user-attachments/assets/83de35d0-ccdb-487c-a486-a3130281f8c6)


### ARIMA Model Fitting Summary:

![image](https://github.com/user-attachments/assets/9a475fa4-ce53-4a7b-ba30-4d4b9b8880f4)


### Forecast Output:

![image](https://github.com/user-attachments/assets/364099db-cd52-4e4c-abc7-6488d6159551)

### Evaluation of Model:

![image](https://github.com/user-attachments/assets/07e1bc0a-6b6c-4f04-bbbb-52fcc721b7e8)

![image](https://github.com/user-attachments/assets/79603c1b-f22a-45e0-b33f-83acfbc762f7)

## Result :
The ARIMA model successfully forecasts future values with reasonable accuracy. The evaluation metrics indicate the model's ability to predict trends while maintaining a small margin of error.







