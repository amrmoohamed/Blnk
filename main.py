import os
from DateExtractor import CRNNModel

def main():
    # Specify the input shape and number of classes
    input_shape = (40, 250, 1)
    num_classes = 124 - 50  # Adapt this according to your dataset

    # Specify the path to your data
    data_path = 'OCR_Dates/'

    # Create an instance of the CRNNModel class
    crnn_model = CRNNModel(input_shape, num_classes)

    # Load data from the specified path
    images, labels = crnn_model.load_data(data_path)

    # Split the dataset into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = crnn_model.split_dataset(images, labels)

    # User menu
    while True:
        print("\nOptions:")
        print("1. Train the CRNN model")
        print("2. Save the model")
        print("3. Load the model")
        print("4. Test the CRNN model on test data")
        print("5. Convert the model to ONNX format")
        print("6. Convert the model to TensorRT format")
        print("7. Predict a date from an image")
        print("8. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            # Train the CRNN model
            epochs = int(input("Enter the number of epochs: "))
            crnn_model.train(X_train, X_val, y_train, y_val, epochs=epochs)
        elif choice == "2":
            # Save the model
            model_path = input("Enter the path to save the model: ")
            crnn_model.save_model(model_path)
        elif choice == "3":
            # Load the model
            model_path = input("Enter the path to load the model: ")
            crnn_model.load_model(model_path)
        elif choice == "4":
            # Test the CRNN model on test data
            metrics, predictions = crnn_model.predict_and_evaluate(X_test, y_test)
        elif choice == "5":
            # Convert the model to ONNX format
            onnx_path = input("Enter the path to save the ONNX model: ")
            crnn_model.convert_to_onnx(crnn_model.model, onnx_path)
        elif choice == "6":
            # Convert the model to TensorRT format
            trt_path = input("Enter the path to save the TensorRT model: ")
            crnn_model.convert_to_trt(crnn_model.model, trt_path)
        elif choice == "7":
            # Predict a date from an image
            test_image_path = input("Enter the path to the test image: ")
            predicted_day, predicted_month, predicted_year = crnn_model.predict_date(test_image_path)
            print("Predicted Date: {}/{}/{}".format(predicted_day, predicted_month, predicted_year))
        elif choice == "8":
            # Exit the program
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
