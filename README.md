# Date Extraction from Identification Documents

## Overview:

This project is designed to extract date information from identification documents using a CRNN-based OCR model. It includes a Python script (main.py) for training, testing, and using the model, as well as a Django web application for real-time date extraction.

## Getting Started:

### Prerequisites:
- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV
- Django (for running the web app)

### Installation:

1. Clone this repository to your local machine:
   git clone https://github.com/amrmoohamed/Blnk.git

2. Install the required Python packages:
   pip install -r requirements.txt

Usage:

Using main.py:

1. Specify the input shape and number of classes in the main.py file:
   input_shape = (40, 250, 1)
   
   num_classes = 74 # From 1950 to 2023  

3. Specify the path to your data:
   data_path = 'OCR_Dates/'

4. Open a terminal and navigate to the repository's root directory.

5. Run the main.py script:
   python main.py

6. Follow the on-screen menu to train the CRNN model, save/load the model, test it on test data, convert it to ONNX or TensorRT formats, and predict dates from images.

Running the Django Web App:

1. Navigate to the OCR directory:
   cd OCR

2. Run the Django development server:
   ./runocrserver.sh

3. A new browser tab will automatically open with the URL http://127.0.0.1:8000/ExtractDate/, allowing you to use the web app for date extraction.

4. To manually stop the server, press Ctrl+C in the terminal.

Opening the Django App in a New Browser Tab:

To run the Django app and open it in a new browser tab with a single command, you can use the provided script:

1. In the root directory of your project, run the following command:
   ./open_browser.sh

This script starts the Django server in the background and automatically opens a new browser tab with the Django app's URL.

License:

This project is licensed under the MIT License (LICENSE).
