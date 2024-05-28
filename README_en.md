# Analysis and Prediction of Lottery "Mega Millions" Number Frequencies Using Machine Learning Models

## Goals and Objectives

### Goals
The main goal of this research project is to analyze historical lottery data to predict the frequency of number appearances in future using advanced machine learning models, specifically Long Short-Term Memory (LSTM) networks. The project aims to provide a comprehensive comparison between traditional trend analysis and deep learning-based predictions.

### Objectives
1. **Data Collection and Preprocessing**
   - Collect and preprocess historical lottery data focusing on the frequency of number appearances over time.
   - Transform the data into formats suitable for analysis and machine learning model training.

2. **Data Visualization**
   - Visualize historical distributions of number appearance frequencies to identify trends and patterns.
   - Create various charts and heatmaps for effective data representation.

3. **Data Analysis**
   - Create Q-Q plots to check the normality of the distribution of winning frequencies.
   - Conduct the Shapiro-Wilk test to statistically assess the normality of the data.
     
4. **Trend Analysis**
   - Conduct trend analysis to identify significant numbers based on historical data.
   - Calculate slopes, intercepts, and p-values for each lottery number to determine their statistical significance.

5. **Polynomial Regression Model Development**
   - Develop and train polynomial regression models to predict the frequency of number occurrences in the future.
   - Optimize the model using various polynomial degrees to improve prediction accuracy.

6. **Model Evaluation**
   - Evaluate the performance of the polynomial regression model using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
   - Compare the performance of the polynomial regression model with traditional trend analysis.

7. **Prediction and Comparison**
   - Predicting the frequency of appearance of numbers based on their previous trends.
   - Normalization of data to interpret the results as probabilities of choosing each number in a given year.
   - Predict the frequency of number occurrences for future years using the trained polynomial regression model.
   - Compare the forecasts of the polynomial regression model with the results of trend analysis to identify similarities and differences.

## Detailed Methodology
### Tools and Libraries
This project utilized the following tools and libraries:
- **Python:** The main programming language used for data analysis and model building.

- **Pandas:** A library for data manipulation and analysis. Used for working with dataframes.

- **NumPy:** A library for array handling and numerical computations.

- **Seaborn:** A visualization library based on matplotlib. Used for creating statistical plots.

- **Matplotlib:** The primary library for creating plots and visualizations.

- **SciPy:** A library for scientific and technical computations. Used for performing statistical operations.

- **Scikit-learn:** A machine learning library in Python. Used for data preprocessing and model evaluation.
   - MinMaxScaler: For data scaling.
   - mean_absolute_error, mean_squared_error: Metrics for model evaluation.

- **TensorFlow and Keras:** A framework and high-level library for building and training neural networks.
   - Sequential: Used for creating the neural network model.
   - LSTM: Long Short-Term Memory layer for recurrent neural networks.
   - Dense: Fully connected layer in the neural network.
   - Dropout: Regularization layer to prevent overfitting.
  
### Data Collection and Preprocessing
- **Data Source:** The data for this project is taken from https://catalog.data.gov/dataset/lottery-mega-millions-winning-numbers-beginning-2002
- **Database Structure**
The Mega Millions lottery database contains information about winning numbers, including Mega Ball and Multiplier, along with draw dates, totaling 2290 rows with the following details:

   | Draw Date   | Winning Numbers  | Mega Ball | Multiplier |
   |-------------|------------------|-----------|------------|
   | 09/25/2020  | 20 36 37 48 67   | 16        | 2.0        |
   | 09/29/2020  | 14 39 43 44 67   | 19        | 3.0        |
   | 10/02/2020  | 09 38 47 49 68   | 25        | 2.0        |
   | 10/06/2020  | 15 16 18 39 59   | 17        | 3.0        |
   | 10/09/2020  | 05 11 25 27 64   | 13        | 2.0        |
  
   - **Draw Date:** The date of the lottery draw, data type - object.
   - **Winning Numbers:** The numbers that were drawn in the lottery, data type - object.
   - **Mega Ball:** A separate number drawn from a different set of balls, data type - int64.
   - **Multiplier:** An optional feature that increases the winning amounts, data type - float64.

**Transformation and Cleaning for Analysis and Visualization:**
- Splitting the "Winning Numbers" column into five separate columns.
- Converting the "Draw Date" column to datetime format for easier date manipulation.
- Dropping the "Multiplier" column as it is not needed for the analysis.
- Converting the numbers to a numeric format for accurate processing.

```python
df[['Number1', 'Number2', 'Number3', 'Number4', 'Number5']] = df['Winning Numbers'].str.split(expand=True)
df.drop('Winning Numbers', axis=1, inplace=True)
df.drop('Multiplier', axis=1, inplace=True)
columns_to_convert = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Mega Ball']
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, downcast='integer', errors='coerce')
df['Draw Date'] = pd.to_datetime(df['Draw Date'])
```
**Normalization and Scaling for LSTM Model Training:**
- Scaling numeric data to a range of 0 to 1 to improve model performance.
- Transforming data into a format that meets LSTM input requirements.

   
### Data Visualization
#### Heatmap of Winning Numbers Frequency by Year
![HeatMapFrequencyOfWinningNumbersWithYears](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbersYears.png)
From this visualization, it is evident that the dataset should be divided into three time intervals according to changes in game rules to minimize the impact of these changes on number selection trends. The time intervals include:

   - **2006-2012:** A period where the numbers ranged from 1 to 56. This segment provides stability and allows for trend analysis without the influence of changes in the number set.
   - **2014-2017:** A period where the numbers ranged up to 75, allowing for the examination of player behavior with an increased number set.
   - **2018-2023:** The most recent period with numbers up to 70, indicating new trends and changes in player strategies.

Considering each period separately allows for a more accurate assessment of number popularity and trends over time, minimizing distortions caused by game rule changes. This approach lays the foundation for more reliable predictive modeling and strategic planning in the lottery context.

---------------------------------------------------------------------------------------------------------------------------
#### Description of Visualizations for the 56-number Lottery

To analyze the frequency of winning numbers in the 56-number lottery, a horizontal bar chart was created. The bar chart shows how often each number appeared over the entire period considered. This visualization helps identify the numbers that appeared most frequently, which can be useful for further analysis and forecasting. Additionally, a heatmap was created to show the frequency of each number appearing over different years in the 56-number lottery.

![FrequencyOfWinningNumbersLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbers56.png)

![HeatMapFrequencyOfWinninNumbersLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbersYears56.png)

A histogram was also created for the Mega Ball numbers in the 56-number lottery. Mega Ball is a separate number drawn from a different set of balls, namely from 46 balls. This visualization helps identify which Mega Ball numbers appeared most frequently. The heatmap visualizes the frequency of Mega Ball occurrences broken down by year.

![FrequencyOfWinningMegaBallsLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBall56nSort.png)

![HeatMapFrequencyOfWinningMegaBallsLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBallByYears56nHM.png)

![HistFrequencyOfWinningMegaBallsLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/HistFrequencyOfMegaBall56n.png)

---------------------------------------------------------------------------------------------------------------------------

#### Description of Visualizations for the 75-number Lottery

To analyze the frequency of winning numbers in the 75-number lottery, a horizontal bar chart was created. The bar chart shows how often each number appeared over the entire period considered. This visualization helps identify the numbers that appeared most frequently, which can be useful for further analysis and forecasting. Additionally, a heatmap was created to show the frequency of each number appearing over different years in the 75-number lottery.

![FrequencyOfWinningNumbersLoto75](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbers75.png)

![HeatMapFrequencyOfWinninNumbersLoto75](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbersYears75.png)

A histogram was also created for the Mega Ball numbers in the 75-number lottery. Mega Ball is a separate number drawn from a different set of balls, namely from 15 balls. This visualization helps identify which Mega Ball numbers appeared most frequently. The heatmap visualizes the frequency of Mega Ball occurrences broken down by year.

![FrequencyOfWinningMegaBallsLoto75](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBall75nSort.png)

![HeatMapFrequencyOfWinningMegaBallsLoto75](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBallByYears75nHM.png)

![HistFrequencyOfWinningMegaBallsLoto75](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/HistFrequencyOfMegaBall75n.png)

---------------------------------------------------------------------------------------------------------------------------

#### Description of Visualizations for the 70-number Lottery

To analyze the frequency of winning numbers in the 70-number lottery, a horizontal bar chart was created. The bar chart shows how often each number appeared over the entire period considered. This visualization helps identify the numbers that appeared most frequently, which can be useful for further analysis and forecasting. Additionally, a heatmap was created to show the frequency of each number appearing over different years in the 70-number lottery.

![FrequencyOfWinningNumbersLoto70](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbers70.png)

![HeatMapFrequencyOfWinninNumbersLoto70](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbersYears70.png)

A histogram was also created for the Mega Ball numbers in the 75-number lottery. Mega Ball is a separate number drawn from a different set of balls, namely from 25 balls. This visualization helps identify which Mega Ball numbers appeared most frequently. The heatmap visualizes the frequency of Mega Ball occurrences broken down by year.

![FrequencyOfWinningMegaBallsLoto70](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBall70nSort.png)

![HeatMapFrequencyOfWinningMegaBallsLoto70](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBallByYears70nHM.png)

![HistFrequencyOfWinningMegaBallsLoto70](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/HistFrequencyOfMegaBall70n.png)

---------------------------------------------------------------------------------------------------------------------------
### Data Analysis

To check the normality of the distribution of winning frequencies in the lottery, Q-Q plots were created and the Shapiro-Wilk test was performed.

#### Q-Q Plots

The Q-Q plot compares the quantiles of the frequency distribution of the dataset with a theoretical normal distribution. The fact that most of our data points lie along a straight line (with some deviation at the ends) suggests that the frequency of number occurrences approximately follows a normal distribution.

- **Q-Q Plot for the period 2006-2012 for the 56-number Lottery:**

![Q-Q Plot for 2006-2012](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/QQPlotFrequency_for_2006-2012.png)

- **Q-Q Plot for the period 2014-2017 for the 75-number Lottery:**

![Q-Q Plot for 2014-2017](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/QQPlotFrequency_for_2014-2017.png)

- **Q-Q Plot for the period 2018-2023 for the 70-number Lottery:**

![Q-Q Plot for 2018-2023](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/QQPlotFrequency_for_2018-2023.png)

#### Shapiro-Wilk Test

The Shapiro-Wilk test further checks the normality of the data. The result of the test provides a p-value, which can be used to determine whether to reject the null hypothesis that the data is normally distributed. A high p-value (typically greater than 0.05) means that the null hypothesis of normality cannot be rejected, indicating that the data can be considered normally distributed for certain statistical tests and confidence interval constructions that assume normality.

Shapiro-Wilk test results:

- **2006-2012 for the 56-number Lottery:**
  - statistic: 0.9777
  - p-value: 0.3827

- **2014-2017 for the 75-number Lottery:**
  - statistic: 0.9804
  - p-value: 0.2958

- **2018-2023 for the 70-number Lottery:**
  - statistic: 0.9760
  - p-value: 0.1988

Based on the results of the Shapiro-Wilk test, we can conclude that the distribution of winning frequencies is approximately normal for all considered periods, as the p-values in all cases exceed 0.05.

### Trend Analysis
For trend analysis, there is no point in considering lotteries with 56 and 75 numbers, as they were held until 2013. It is more reasonable to focus on analyzing the current lottery with 70 numbers.

#### Data Preparation
   - **DataFrame:** The DataFrame heatmap_df70n is used, where each column represents a separate number, and the rows can represent the frequency of appearance of this number in different time periods.
   - **Adding a Constant:** For each number, an array X is created with a constant and an index (time), allowing the model to account for the intercept.
   - **Regression Model:** An OLS (Ordinary Least Squares) model is built for each number, where the dependent variable y is the frequency of the number, and the independent variables are time.
   - **Saving Results:** The results of the model, including the slope, intercept, and p-value, are saved in the dictionary trends.
   - **Transforming Results into DataFrame:** The dictionary trends is converted into a DataFrame trends_df for easier analysis and data visualization.
This approach allows us to analyze the trends in the frequency of each number over time and determine which numbers are becoming more or less popular.

```python
trends = {}
for column in heatmap_df70n.columns:
    # Checking if a column is numeric
    if pd.api.types.is_numeric_dtype(heatmap_df70n[column]):
        X = sm.add_constant(np.arange(len(heatmap_df70n)))  
        y = heatmap_df70n[column].astype(float).values  
        model = sm.OLS(y, X)
        results = model.fit()
        # Save the results for each number
        trends[column] = {'Slope': results.params[1], 'Intercept': results.params[0], 'P-value': results.pvalues[1]}

# Transform the dictionary into a DataFrame for ease of analysis
trends_df = pd.DataFrame(trends).T
# Set options to display all rows
pd.set_option('display.max_rows', None)
print(trends_df)
```

These results for the linear regression of each lottery number show the slope coefficient Slope and the p-value. The slope coefficient indicates the direction of the trend (a positive value means an increase in frequency, a negative value means a decrease), and the p-value indicates the statistical significance of this trend.

Numbers with significant negative trends and low p-values (for example, number 14 with a Slope of -0.942857 and a P-value of 0.031797) show a statistically significant decrease in frequency over time. Numbers with significant positive trends and low p-values (for example, number 18 with a Slope of 2.028571 and a P-value of 0.037760) show a statistically significant increase in frequency.

### LSTM Model Development
- **Model Architecture:**
1. LSTM layers with dropout for regularization.
2. Dense output layer to predict frequencies.
- **Hyperparameters:**
1. Look-back periods (e.g., 2, 3, 4 years).
2. Batch size and number of epochs.
- **Training and Testing:**
1. Split data into training and test sets.
2. Train the model and evaluate its performance on both sets.

### Model Evaluation
- **Metrics:**
MAE and RMSE for both training and test sets.
- **Comparison:**
Compare the error metrics to determine the best-performing model configuration.

### Prediction and Comparison
#### Forecasting Number Frequencies

Forecasting the frequency of lottery numbers is based on their previous trends (slope and intercept determined through linear regression). For each significant number (i.e., numbers with a p-value less than 0.3), the predicted frequency is calculated based on the trend line equation, where the intercept is the initial value and the slope is the rate of change in frequency over the years. Then, for each year in the given range, the predicted values are calculated. The resulting DataFrame predicted_df70n is normalized so that the values in each year sum to 1, allowing the results to be interpreted as the probabilities of selecting each number in that year.

```python
# Years to forecast
years = np.arange(2024, 2028)
# Converting a dictionary to a DataFrame with the correct columns
trends_df70n = pd.DataFrame.from_dict(trends, orient='index', columns=['Intercept', 'Slope', 'P-value'])
# Determining the threshold for p-values p < 0.3
significant_numbers = trends_df70n[trends_df70n['P-value'] < 0.3]
predicted_frequencies = {}
for index, row in significant_numbers.iterrows():
    predicted_frequencies[index] = row['Intercept'] + row['Slope'] * (years - 2023)
predicted_df70n = pd.DataFrame(predicted_frequencies, index=years)
# Normalizing values
probability_df70n = predicted_df70n.div(predicted_df70n.sum(axis=1), axis=0)
print(probability_df70n)
# Visualization
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for i, year in enumerate(years):
    row = i // 2
    col = i % 2
    probability_df70n.loc[year].plot(kind='bar', ax=axs[row, col], color='indigo')
    axs[row, col].set_title(f'Predicted probabilities for {year} year')
    axs[row, col].set_xlabel('Number')
    axs[row, col].set_ylabel('Probability')

plt.tight_layout()
plt.show()
# Numbers with significant positive trends and low p-values
selected_numbers_df70n = trends_df70n[(trends_df70n['P-value'] < 0.3) & (trends_df70n['Slope'] > 0)]
print("Selected Numbers with Positive Trends and Low P-Values", selected_numbers_df70n.index.tolist())
```

Based on these data, numbers with positive trends and low p-values were selected:
```python
Selected Numbers with Positive Trends and Low P-Values: [3, 8, 15, 18, 19, 21, 36, 45, 47, 50, 51, 52, 55, 61]
```

**Visualizations of projected probabilities for 2024-2027:**

![PredictedProbabilitiesTrends70Numbers](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/PredictedProbabilitiesFor2024_2027.png)

#### Prediction for Mega Ball
A similar process was followed for Mega Ball numbers. As a result, the following Mega Ball numbers with positive trends and low p-values were selected:

```python
Selected Mega Balls with Positive Trends and Low P-Values: [12, 13, 18, 24, 25]
```

**Visualizations of projected Mega Ball probabilities for 2024-2027:**

![PredictedProbabilitiesTrendsMegaBall](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/PredictedProbabilitiesMegaBall.png)

#### Comparison with Trend Analysis:
Visualize and compare the predicted frequencies from the LSTM model and trend analysis.


## Conclusion
The project aims to leverage advanced machine learning techniques to enhance the accuracy of lottery number frequency predictions. By comparing traditional trend analysis with deep learning models, the research seeks to identify the most effective methods for forecasting lottery outcomes. The findings and visualizations provided will offer valuable insights for both researchers and lottery enthusiasts.
