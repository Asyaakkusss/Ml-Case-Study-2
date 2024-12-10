import numpy as np

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

def predict_test(train_data, train_labels, test_data):
    train_labels -= 1

    # Feature extraction: compute mean and standard deviation of each row for
    # each sensor and concatenate across sensors to form the feature vector
    mean_train_feature = np.mean(train_data, axis=1)
    std_train_feature = np.std(train_data, axis=1)
    train_features = np.hstack((mean_train_feature, std_train_feature))
    mean_test_feature = np.mean(test_data, axis=1)
    std_test_feature = np.std(test_data, axis=1)
    test_features = np.hstack((mean_test_feature, std_test_feature))

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