from django.shortcuts import render, redirect
from django.http import JsonResponse
import cv2
import numpy as np
import onnxruntime as ort 

# Load the ONNX model for OCR date prediction
onnx_model = ort.InferenceSession('/Users/amrmohamed/Downloads/Blnk/model.onnx')

def index(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image'].read()
        image = cv2.imdecode(np.frombuffer(uploaded_image, np.uint8), -1)
        
        # Preprocess the image (similar to your CRNNModel preprocess_image function)
        image = cv2.resize(image, (250, 40))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.expand_dims(image, axis=-1)

        # Perform inference using the ONNX model
        input_name = onnx_model.get_inputs()[0].name
        output_names = [output.name for output in onnx_model.get_outputs()]
        input_data = {input_name: image.astype(np.float32)}
        results = onnx_model.run(output_names, input_data)

        # Extract predictions from the results
        predicted_day, predicted_month, predicted_year = results

        # Find the index of the maximum probability for each prediction
        day_idx = np.argmax(predicted_day)
        month_idx = np.argmax(predicted_month) + 1
        year_idx = np.argmax(predicted_year) + 1950

        date_idx = str(day_idx) + '/' + str(month_idx) + '/' + str(year_idx)

        arabic_numerals = '٠١٢٣٤٥٦٧٨٩'
        english_numerals = '0123456789'
        numeral_mapping = str.maketrans(english_numerals, arabic_numerals)

        # Use translate to replace Arabic numerals with English numerals in the label
        date_idx = date_idx.translate(numeral_mapping)
        day_arabic, month_arabic, year_arabic = date_idx.split('/')

        # Redirect to the result page with predictions
        return redirect('result', day=day_idx, month=month_idx, year=year_idx,
        day_arabic = day_arabic, month_arabic = month_arabic, year_arabic = year_arabic)

    return render(request, 'index.html')

def result(request, day, month, year, day_arabic, month_arabic, year_arabic):
    context = {
        'predicted_day': day,
        'predicted_month': month,
        'predicted_year': year,
        'predicted_day_arabic': day_arabic,
        'predicted_month_arabic': month_arabic,
        'predicted_year_arabic': year_arabic,
    }
    #print(context)
    return render(request, 'result.html', {'prediction': context})
