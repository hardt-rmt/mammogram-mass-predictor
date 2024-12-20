import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Process the data
def preprocess_data(datafile):
    # Import mammography_masses data into a Pandas dataframe
    masses_data_header = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity']
    masses_data = pd.read_csv(datafile, na_values='?', names=masses_data_header)
    # Given that there's no correlation, drop the rows with the missing data
    masses_data.dropna(inplace=True)
    # Convert the dataframe into numpy arrays to be use by scikit-learn
    # Create an array that only extracts the feature data we want to work with
    # (age, shape, margin, and density) and another array that contains the classes (severity)
    all_features = masses_data[['age', 'shape', 'margin', 'density']].values
    all_classes = masses_data['severity'].values
    # Normalize the data
    scaler = StandardScaler()
    minMaxScaler = MinMaxScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    all_features_minmax_scaled = minMaxScaler.fit_transform(all_features)
    return all_features_scaled, all_classes, all_features_minmax_scaled


# Process prediction data
def preprocess_prediction_data(datafile):
    # Import mammography_masses data into a Pandas dataframe
    masses_data_header = ['BI-RADS', 'age', 'shape', 'margin', 'density']
    masses_data = pd.read_csv(datafile, na_values='?', names=masses_data_header)
    # Given that there's no correlation, drop the rows with the missing data
    masses_data.dropna(inplace=True)
    # Convert the dataframe into numpy arrays to be use by scikit-learn
    # Create an array that only extracts the feature data we want to work with
    # (age, shape, margin, and density) and another array that contains the classes (severity)
    all_predict_features = masses_data[['age', 'shape', 'margin', 'density']].values
    # Normalize the data
    standardScaler = StandardScaler()
    minMaxScaler = MinMaxScaler()
    all_predicted_features_minmax_scaled = minMaxScaler.fit_transform(all_predict_features)
    all_predict_features_scaled = standardScaler.fit_transform(all_predict_features)
    return all_predict_features_scaled, all_predicted_features_minmax_scaled

