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

3. **Trend Analysis**
   - Conduct trend analysis to identify significant numbers based on historical data.
   - Calculate slopes, intercepts, and p-values for each lottery number to determine their statistical significance.

4. **LSTM Model Development**
   - Develop and train LSTM models to predict future frequencies of number appearances.
   - Optimize the model using various hyperparameters, such as the look-back period, to enhance prediction accuracy.

5. **Model Evaluation**
   - Evaluate the performance of the LSTM model using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
   - Compare the performance of the LSTM model with traditional trend analysis.

6. **Prediction and Comparison**
   - Predict the frequency of number appearances for future years using the trained LSTM model.
   - Compare the LSTM model predictions with trend analysis results to identify similarities and differences.

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
**Heatmap of Winning Numbers Frequency by Year**
![HeatMapFrequencyOfWinningNumbersWithYears](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbersYears.png)
From this visualization, it is evident that the dataset should be divided into three time intervals according to changes in game rules to minimize the impact of these changes on number selection trends. The time intervals include:

   - **2006-2012:** A period where the numbers ranged from 1 to 56. This segment provides stability and allows for trend analysis without the influence of changes in the number set.
   - **2014-2017:** A period where the numbers ranged up to 75, allowing for the examination of player behavior with an increased number set.
   - **2018-2023:** The most recent period with numbers up to 70, indicating new trends and changes in player strategies.

Considering each period separately allows for a more accurate assessment of number popularity and trends over time, minimizing distortions caused by game rule changes. This approach lays the foundation for more reliable predictive modeling and strategic planning in the lottery context.

---------------------------------------------------------------------------------------------------------------------------
**Description of Visualizations for the 56-number Lottery**

To analyze the frequency of winning numbers in the 56-number lottery, a horizontal bar chart was created. The bar chart shows how often each number appeared over the entire period considered. This visualization helps identify the numbers that appeared most frequently, which can be useful for further analysis and forecasting. Additionally, a heatmap was created to show the frequency of each number appearing over different years in the 56-number lottery.

![FrequencyOfWinningNumbersLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbers56.png)

![HeatMapFrequencyOfWinninNumbersLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbersYears56.png)

A histogram was also created for the Mega Ball numbers in the 56-number lottery. Mega Ball is a separate number drawn from a different set of balls. This visualization helps identify which Mega Ball numbers appeared most frequently. The heatmap visualizes the frequency of Mega Ball occurrences broken down by year.

![FrequencyOfWinningMegaBallsLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBall56nSort.png)

![HeatMapFrequencyOfWinningMegaBallsLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBallByYears56nHM.png)

![HistFrequencyOfWinningMegaBallsLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/HistFrequencyOfMegaBall56n.png)

---------------------------------------------------------------------------------------------------------------------------
### Trend Analysis
- **Statistical Analysis:** Calculate slopes, intercepts, and p-values to identify trends.
- **Significance Threshold:** Use p-value < 0.3 to select significant numbers for trend analysis.
  
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
- **Future Predictions:**
Predict lottery number frequencies for the years 2024, 2025, 2026, and 2027.
- **Comparison with Trend Analysis:**
Visualize and compare the predicted frequencies from the LSTM model and trend analysis.


## Conclusion
The project aims to leverage advanced machine learning techniques to enhance the accuracy of lottery number frequency predictions. By comparing traditional trend analysis with deep learning models, the research seeks to identify the most effective methods for forecasting lottery outcomes. The findings and visualizations provided will offer valuable insights for both researchers and lottery enthusiasts.
