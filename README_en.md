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

- **Preprocessing Steps:**
1. Normalize and scale the data for LSTM model training.
2. Reshape the data to fit the LSTM input requirements.
   
### Data Visualization
- **Histogram:** To show the distribution of lottery number frequencies.
- **Heatmaps:** To visualize the frequency of winning numbers over different years.
- **Bar Charts:** To compare predicted frequencies from different models.
  
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
