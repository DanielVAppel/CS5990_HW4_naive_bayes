#-------------------------------------------------------------------------
# AUTHOR: Daniel Appel
# FILENAME: CS5990_HW4_Naive_Bayes
# SPECIFICATION: the Python program (naive_bayes.py) that will read the file weather_training.csv (training set) and classify each test instance from the file weather_test (test set). 
# FOR: CS 5990- Assignment #4
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

# Importing necessary Python libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# 11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

# Hyperparameter values for smoothing
s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

# Reading the training data
df_train = pd.read_csv('weather_training.csv')
X_training = df_train[['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']].values
y_training_original = df_train['Temperature (C)'].values

# Discretize y_training
y_training = np.zeros(len(y_training_original), dtype=int)
for i in range(len(y_training_original)):
    min_diff = float('inf')
    best_class = 0
    for class_value in classes:
        diff = abs(y_training_original[i] - class_value)
        if diff < min_diff:
            min_diff = diff
            best_class = class_value
    y_training[i] = best_class

# Reading the test data
df_test = pd.read_csv('weather_test.csv')
X_test = df_test[['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']].values
y_test_original = df_test['Temperature (C)'].values

# Discretize y_test
y_test = np.zeros(len(y_test_original), dtype=int)
for i in range(len(y_test_original)):
    min_diff = float('inf')
    best_class = 0
    for class_value in classes:
        diff = abs(y_test_original[i] - class_value)
        if diff < min_diff:
            min_diff = diff
            best_class = class_value
    y_test[i] = best_class

# Keep track of the highest accuracy
highest_accuracy = 0.0
best_parameter = None

# Loop over the hyperparameter value (s)
for s in s_values:
    # Fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf.fit(X_training, y_training)
    
    # Initialize accuracy calculation
    correct_predictions = 0
    total_predictions = len(X_test)
    
    # Make the naive_bayes prediction for each test sample and compute its accuracy
    for i, (x_testSample, y_testSample) in enumerate(zip(X_test, y_test_original)):
        # Make prediction
        y_pred = clf.predict([x_testSample])[0]
        
        # Calculate percentage difference
        percent_diff = 100 * abs(y_pred - y_testSample) / abs(y_testSample) if y_testSample != 0 else 100 * abs(y_pred - y_testSample)
        
        # Check if prediction is considered correct (within ±15%)
        if percent_diff <= 15:
            correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    
    # Check if this is the highest accuracy so far
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_parameter = s
        print(f"Highest Naïve Bayes accuracy so far: {highest_accuracy:.2f} Parameter: s = {s}")

print(f"Final best parameter: s = {best_parameter}")
print(f"Highest accuracy achieved: {highest_accuracy:.2f}")