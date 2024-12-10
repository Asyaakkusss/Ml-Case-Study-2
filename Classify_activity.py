import tensorflow as tf
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Model as Model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
# Last row of training data for train/test split
train_end_index = 3511


def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()

class ScikitLearnF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_data, val_labels = self.validation_data
        val_predictions = self.model.predict(val_data)
        val_predictions = np.argmax(val_predictions, axis=1)

        # Calculate F1 scores using scikit-learn
        micro_f1 = f1_score(val_labels, val_predictions, average='micro')
        macro_f1 = f1_score(val_labels, val_predictions, average='macro')

        # Log metrics
        print(f"\nEpoch {epoch + 1} F1 Scores - Micro: {micro_f1:.4f}, Macro: {macro_f1:.4f}")
        logs['val_micro_f1'] = micro_f1
        logs['val_macro_f1'] = macro_f1

        # Optional: Print a classification report for more insights
        if (epoch + 1) % 10 == 0:  # Print every 10 epochs for readability
            print("\nClassification Report:\n", classification_report(val_labels, val_predictions))

def predict_test(train_data, train_labels, test_data):
    train_labels -= 1

    tf.random.set_seed(0)

    '''
    tf.keras.layers.Conv1D(64, 4, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(64, 4, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Conv1D(128, 4, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(128, 4, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation="softmax")
    '''

    model = tf.keras.Sequential([
        Conv1D(64, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling1D(pool_size=2),
        Conv1D(128, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling1D(pool_size=2),
        Conv1D(256, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        GlobalMaxPooling1D(),
        Flatten(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])

    model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
    )

    # Define callbacks
    checkpoint_filepath = '/tmp/checkpoint.keras'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',  # Monitor validation loss instead of custom F1
        mode='min',
        save_best_only=True,
    )

    # Standardize data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
    test_data_scaled = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)

    # Split validation data
    validation_split = 0.4
    split_idx = int(train_data_scaled.shape[0] * (1 - validation_split))
    val_data = train_data_scaled[split_idx:]
    val_labels = train_labels[split_idx:]
    train_data_scaled = train_data_scaled[:split_idx]
    train_labels = train_labels[:split_idx]

    model.summary()

    # ScikitLearnF1Callback
    sklearn_f1_callback = ScikitLearnF1Callback(validation_data=(val_data, val_labels))

    callbacks = [EarlyStopping(patience=5), model_checkpoint_callback, sklearn_f1_callback]

    # Train the model
    history = model.fit(
        train_data_scaled,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=50,
        callbacks=callbacks
    )

    # Evaluate model on test data
    test_outputs = model.predict(test_data_scaled)
    test_outputs = np.argmax(test_outputs, axis=1)

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
    