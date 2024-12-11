import numpy as np

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft

from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

import xgboost as xgb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
# Last row of training data for train/test split
train_end_index = 3511
# Logistic regression hyperparameters

def extract_features(data):
    """
    Extract features from 3D sensor data (samples x timesteps x features).
    Each feature captures characteristics specific to activities such as resting, walking, running, and commuting.
    
    Parameters:
        data (numpy.ndarray): Sensor data with shape (samples, timesteps, axes).
    
    Returns:
        numpy.ndarray: Extracted feature vectors for each sample.
    """
    # Statistical Features
    mean_features = np.mean(data, axis=1)  # Mean for each axis
    std_features = np.std(data, axis=1)   # Standard deviation
    range_features = np.ptp(data, axis=1)  # Range (max - min)
    variance_features = np.var(data, axis=1)  # Variance
    skew_features = np.apply_along_axis(lambda x: skew(x, axis=0), axis=1, arr=data)  # Skewness
    kurtosis_features = np.apply_along_axis(lambda x: kurtosis(x, axis=0), axis=1, arr=data)  # Kurtosis

    # Time-Domain Features
    zero_crossings = np.sum(np.diff(np.sign(data), axis=1) != 0, axis=1)  # Zero-crossing rate
    energy_features = np.sum(data ** 2, axis=1)  # Signal energy

    # Frequency Features (FFT)
    fft_coeffs = np.abs(fft(data, axis=1))
    dominant_freqs = np.argmax(fft_coeffs, axis=1)  # Dominant frequency index
    spectral_entropy = -np.sum((fft_coeffs / np.sum(fft_coeffs, axis=1, keepdims=True)) *
                               np.log2(fft_coeffs + 1e-12), axis=1)  # Spectral entropy

    # Shape Features
    peak_counts = np.array([np.sum([len(find_peaks(row[:, axis])[0]) for axis in range(row.shape[1])]) for row in data])
    peak_counts = np.tile(peak_counts[:, None], (1, 6))

    '''
    # Ensure all features are 2D
    zero_crossings = zero_crossings[:, None]  # Convert to 2D
    energy_features = energy_features[:, None]
    dominant_freqs = dominant_freqs[:, None]
    spectral_entropy = spectral_entropy[:, None]
    peak_counts = peak_counts[:, None]'''

    # Combine all features
    combined_features = np.hstack([
        mean_features, std_features, range_features, variance_features,
        skew_features, kurtosis_features, zero_crossings,
        energy_features, dominant_freqs, spectral_entropy,
        peak_counts
    ])

    return combined_features

def predict_test(train_data, train_labels, test_data):
    train_labels -= 1

    train_features = extract_features(train_data)
    test_features = extract_features(test_data)

    # Standardize features and train a logistic regression model
    scaler = StandardScaler()
    train_features_std = scaler.fit_transform(train_features)
    test_features_std = scaler.transform(test_features)

    best_model = hyperparameter_tuning(train_features_std, train_labels)

    preds = best_model.predict(test_features_std)

    preds += 1

    return preds

def hyperparameter_tuning(train_data, train_labels):
    xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

    # Define the parameter grid
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.1, 0.2, 0.3, 0.4],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'min_child_weight': [1, 2, 3, 4],
        'reg_lambda': [50, 100, 200],
        'reg_alpha': [50, 100, 200]
    }

    # Define the scorer (F1 macro)
    f1_scorer = make_scorer(f1_score, average='macro')

    # Define RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=50,  # Number of parameter settings sampled
        scoring=f1_scorer,
        cv=5,  # 5-fold cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1,  # Use all available cores
    )

    # Fit RandomizedSearchCV
    random_search.fit(train_data, train_labels)

    # Best hyperparameters and best score
    print("Best Hyperparameters:", random_search.best_params_)
    print("Best F1 Macro Score:", random_search.best_score_)

    return random_search.best_estimator_


# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    # Load labels and training sensor data into 3-D array
    labels = np.loadtxt('labels_train_1.csv', dtype='int')
    data_slice_0 = np.loadtxt(sensor_names[0] + '_train_1.csv',
    delimiter=',')
    data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1],
    len(sensor_names)))
    data[:, :, 0] = data_slice_0
    del data_slice_0
    for sensor_index in range(1, len(sensor_names)):
        data[:, :, sensor_index] = np.loadtxt(
        sensor_names[sensor_index] + '_train_1.csv', delimiter=',')
    # Split into training and test by row index. Do not use a random split as
    # rows are not independent!
    train_data = data[:train_end_index+1, :, :]
    train_labels = labels[:train_end_index+1]
    test_data = data[train_end_index+1:, :, :]
    test_labels = labels[train_end_index+1:]
    test_outputs = predict_test(train_data, train_labels, test_data)
    # Compute micro and macro-averaged F1 scores
    micro_f1 = f1_score(test_labels, test_outputs, average='micro')
    macro_f1 = f1_score(test_labels, test_outputs, average='macro')
    print(f'Micro-averaged F1 score: {micro_f1}')
    print(f'Macro-averaged F1 score: {macro_f1}')
    # Examine outputs compared to labels
    n_test = test_labels.size
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(n_test), test_labels, 'b.')
    plt.xlabel('Time window')
    plt.ylabel('Target')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(n_test), test_outputs, 'r.')
    plt.xlabel('Time window')
    plt.ylabel('Output (predicted target)')
    plt.show()