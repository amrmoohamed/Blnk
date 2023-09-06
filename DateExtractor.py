import cv2  # Import the OpenCV library for image processing
import os  # Import the os module for file operations
import numpy as np  # Import NumPy for array operations
import tensorflow as tf  # Import TensorFlow for building and working with deep learning models
import tensorflow.keras as keras  # Import Keras, a high-level neural networks API
import tensorflow.keras.backend as K  # Import Keras backend for custom loss functions
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Reshape,
    LSTM,
    BatchNormalization,
    Dense,
)  # Import specific layers from Keras
from sklearn.model_selection import train_test_split  # Import train_test_split for data splitting
from sklearn.metrics import accuracy_score  # Import accuracy_score for evaluating model performance
import tf2onnx  # Import tf2onnx for converting the model to ONNX format
import tensorflow.contrib.tensorrt as trt  # Import TensorRT for optimizing and converting models to TensorRT format

class CRNNModel:
    def __init__(self, input_shape, num_classes):
        # Initialize the CRNNModel with input shape and number of classes
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_crnn_model()  # Build the CRNN model upon initialization

    def build_crnn_model(self):
        # Create a Sequential Keras model
        model = keras.Sequential()
        
        # Convolutional layers
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        # Add a 2D convolutional layer with 64 filters, 3x3 kernel, ReLU activation, and 'same' padding
        model.add(MaxPooling2D(pool_size=(2, 2)))  # Add a max-pooling layer with a 2x2 pool size
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # Add another convolutional layer
        model.add(MaxPooling2D(pool_size=(2, 2)))  # Add another max-pooling layer

        model.add(Reshape(target_shape=((self.input_shape[0]//4), -1)))  # Reshape the data

        # Recurrent layers
        model.add(LSTM(256, return_sequences=True))  # Add an LSTM layer with 256 units and return sequences
        model.add(BatchNormalization())  # Add batch normalization
        model.add(LSTM(256))  # Add another LSTM layer with 256 units
        model.add(BatchNormalization())  # Add batch normalization

        # Separate components for day, month, and year predictions
        day_pred = Dense(32, activation='softmax', name='day_pred')(model.layers[-1].output)  # Day prediction
        month_pred = Dense(12, activation='softmax', name='month_pred')(model.layers[-1].output)  # Month prediction
        year_pred = Dense(self.num_classes, activation='softmax', name='year_pred')(model.layers[-1].output)  # Year prediction

        model = keras.Model(inputs=model.inputs, outputs=[day_pred, month_pred, year_pred])  # Create the final model
        
        return model

    def preprocess_image(self, image_path):
        # Load an image from a file using OpenCV
        image = cv2.imread(image_path)

        # Resize the image to a fixed size (250x40)
        image = cv2.resize(image, (250, 40))

        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize the pixel values to the range [0, 1]
        image = image / 255.0

        # Expand the dimensions of the image for model input
        image = np.expand_dims(image, axis=-1)

        return image

    def load_label(self, label_path):
        # Read a label from a text file and preprocess it
        with open(label_path, 'r') as file:
            label = file.readline().strip()  # Read the date from the text file

        # Create a mapping from Arabic to English numerals
        arabic_numerals = '٠١٢٣٤٥٦٧٨٩'
        english_numerals = '0123456789'
        numeral_mapping = str.maketrans(arabic_numerals, english_numerals)

        # Use translate to replace Arabic numerals with English numerals in the label
        label = label.translate(numeral_mapping)

        # Split the date into day, month, and year components
        year, month, day = label.split('/')

        # Convert the components to integers
        year = int(year)
        month = int(month) - 1  # Subtract 1 to match Python's 0-based indexing for months
        day = int(day)

        year -= 1950  # Offset the year by 1950 to get the class index

        # Return the day, month, and year as a list
        return [day, month, year]

    def load_data(self, data_path):
        # Load images and labels from a directory
        image_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        images = []
        labels = []

        # Load images and encode the labels
        for file in image_files:
            image_path = os.path.join(data_path, file)
            label_path = os.path.join(data_path, file[:-4] + '.txt')

            # Check if the label file exists
            if os.path.exists(label_path):
                day, month, year = self.load_label(label_path)
                # Preprocess the image and load the label
                image = self.preprocess_image(image_path)
                labels.append([day, month, year])
            else:
                continue

            images.append(image)

        # Convert lists to NumPy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Create the train and test sets
        return images, labels

    def split_dataset(self, images, labels, test_size=0.2, validation_size=0.25, random_state=42):
        # Split the data into train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_model(self, model_path):
        # Save the Keras model to a file
        self.model.save(model_path)

    def load_model(self, model_path):
        # Load a Keras model from a file
        self.model = keras.models.load_model(model_path)

    def predict_date(self, image_path):
        # Predict a date from an image
        image = self.preprocess_image(image_path)  # Implement image preprocessing according to your requirements
        image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch

        predictions = self.model.predict(image)  # Make predictions on the image

        day_idx = np.argmax(predictions[0])
        month_idx = np.argmax(predictions[1])
        year_idx = np.argmax(predictions[2])

        predicted_day = day_idx  # Extract the predicted day
        predicted_month = month_idx + 1  # Add 1 to match human-readable month representation
        predicted_year = 1950 + year_idx  # Assuming we have labels ranging from 1950 to 2023

        return predicted_day, predicted_month, predicted_year

    def predict_and_evaluate(self, X_test, y_test):
        # Predict on test data and evaluate the model's performance
        predictions = self.model.predict(X_test)  # Make predictions on the test data

        # Extract individual predictions
        predicted_day = np.argmax(predictions[0], axis=1)
        predicted_month = np.argmax(predictions[1], axis=1)
        predicted_year = np.argmax(predictions[2], axis=1)

        # Calculate accuracy for each component
        accuracy_day = accuracy_score(y_test[:, 0], predicted_day)
        accuracy_month = accuracy_score(y_test[:, 1], predicted_month)
        accuracy_year = accuracy_score(y_test[:, 2], predicted_year)

        print(f"Accuracy of Day prediction is {accuracy_day}\n")
        print(f"Accuracy of Month prediction is {accuracy_month}\n")
        print(f"Accuracy of Year prediction is {accuracy_year}")

        predictions[1] = predictions[1] + 1  # Add 1 to predicted months to match human-readable representation
        predictions[2] = predictions[1] + 1950  # Offset predicted years by 1950

        return {
            "accuracy_day": accuracy_day,
            "accuracy_month": accuracy_month,
            "accuracy_year": accuracy_year,
        }, predictions

    def train(self, X_train, X_val, y_train, y_val, epochs):
        # Compile and train the model
        self.model.compile(optimizer='adam', 
                           loss={
                               'day_pred': 'sparse_categorical_crossentropy',
                               'month_pred': 'sparse_categorical_crossentropy',
                               'year_pred': 'sparse_categorical_crossentropy'
                           },
                           metrics={
                               'day_pred': 'sparse_categorical_accuracy',
                               'month_pred': 'sparse_categorical_accuracy',
                               'year_pred': 'sparse_categorical_accuracy'
                           })

        self.model.fit(X_train, {'day_pred': y_train[:, 0], 'month_pred': y_train[:, 1], 'year_pred': y_train[:, 2]},
                       validation_data=(X_val, {'day_pred': y_val[:, 0], 'month_pred': y_val[:, 1], 'year_pred': y_val[:, 2]}),
                       epochs=epochs,
                       batch_size=16)

    def convert_to_onnx(self, model, onnx_path):
        # Convert the Keras model to ONNX format
        spec = (tf.TensorSpec((None,) +self.input_shape, tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_path)

    def convert_to_trt(self, keras_model, trt_path):
        # Convert the Keras model to TensorRT format
        tf.saved_model.save(keras_model, 'model_dir')

        params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode='FP16')  # Specify precision mode (FP16)

        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir='model_dir',
            conversion_params=params)
        converter.convert()
        converter.save(trt_path)

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

