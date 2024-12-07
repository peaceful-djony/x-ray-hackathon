import streamlit as st
import pandas as pd
import numpy as np
import pydicom
import cv2
import pickle
from PIL import Image
import io
import os

print(os.getcwd())

# Загрузка обученной модели
model_path = './models/logreg_densenet_model.pkl'  # Укажите правильный путь
model=pickle.load(open(model_path,'rb'))
IMG_SIZE = 224  # Размер изображения, используемый при обучении модели


def preprocess_image(file, img_size):
    # Определение типа файла и его чтение
    if file.type == 'application/dicom' or file.type == 'application/octet-stream':
        dicom = pydicom.dcmread(file)
        img = dicom.pixel_array
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # Используем PIL для открытия любых изображений
        image = Image.open(file)
        img = np.array(image.convert('RGB'))

    # Изменение размера и нормализация
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def predict_image(file):
    preprocessed_image = preprocess_image(file, IMG_SIZE)
    prediction = model[0].predict(preprocessed_image)
    return prediction[0]


st.title("Image Prediction")

uploaded_files = st.file_uploader("Upload image files (DICOM, JPG, PNG)", accept_multiple_files=True)

if uploaded_files:
    results = []
    for idx, uploaded_file in enumerate(uploaded_files):
        prediction = predict_image(uploaded_file)
        foreign_bodies_prob = prediction[0]
        clavicle_fracture_prob = prediction[1]

        results.append({
            'index': idx + 1,
            'study_instance_anon': uploaded_file.name,
            'result_fracture': clavicle_fracture_prob,
            'result_medimp': foreign_bodies_prob
        })

    results_df = pd.DataFrame(results)

    # Отображение DataFrame в Streamlit
    st.dataframe(results_df)

    # Скачивание результата в Excel
    excel_bytes = io.BytesIO()
    with pd.ExcelWriter(excel_bytes, engine='xlsxwriter') as writer:
        results_df.to_excel(writer, index=False)

    st.download_button(label="Download Results as Excel",
                       data=excel_bytes.getvalue(),
                       file_name='predictions.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

