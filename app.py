import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# Загрузка модели


# Интерфейс Streamlit
st.title("Классификация степени заболевания")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # Предсказание сразу после загрузки изображения
    label, probabilities = predict(image)

    # Проверка вероятности предсказания
    min_probability = 0.5
    if max(probabilities) < min_probability:
        st.error("Ошибка: Максимальная вероятность предсказания меньше 0.5.")
    else:
        st.write(f"Предсказанный класс: {label}")
        if label == 0:
            st.write("Класс 0: Здоровый")
        elif label == 1:
            st.write("Класс 1: Stage 2-3")
        elif label == 2:
            st.write("Класс 2: Plus")

        # Вывод вероятностей для каждого класса
        st.write("Вероятности классов:")
        st.write(f"Здоровый: {probabilities[0]:.4f}")
        st.write(f"Stage 2-3: {probabilities[1]:.4f}")
        st.write(f"Plus: {probabilities[2]:.4f}")