import tensorflow as tf
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model as Model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
# Last row of training data for train/test split
train_end_index = 3511

def predict_test(train_data, train_labels, test_data):

    train_labels -= 1

    tf.random.set_seed(0)

    model = tf.keras.Sequential(
        [
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(),

        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
        ]
    )

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    checkpoint_filepath = '/tmp/checkpoint.keras'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True)

    
    callbacks = [
                EarlyStopping(patience=2),
                model_checkpoint_callback,
    ]
    
    # Step 1: Initialize the scaler
    scaler = StandardScaler()

    # Step 2: Reshape training data to 2D for scaling
    train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])  # Flatten timesteps into samples

    # Step 3: Fit the scaler on training data and scale
    train_data_scaled = scaler.fit_transform(train_data_reshaped)

    # Step 4: Reshape scaled training data back to 3D
    train_data_scaled = train_data_scaled.reshape(train_data.shape)

    # Step 5: Repeat for test data (using the same scaler)
    test_data_reshaped = test_data.reshape(-1, test_data.shape[-1])  # Flatten timesteps into samples
    test_data_scaled = scaler.transform(test_data_reshaped)
    test_data_scaled = test_data_scaled.reshape(test_data.shape)
    
    history = model.fit(train_data_scaled,train_labels, epochs=100, callbacks=callbacks)

    test_outputs = model.predict(test_data_scaled)
    test_outputs = np.argmax(test_outputs, axis=1)  # Convert probabilities to class labels

    test_outputs += 1

    return test_outputs

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