import streamlit as st
import numpy as np
import cv2

def grayscale_conversion(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def black_and_white_conversion(image, threshold):
    _, bw_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return bw_image

def resize_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    resized_image = cv2.resize(image, (width, height))
    return resized_image

def main():
    # Introduction
    st.title("Image Conversion App")
    st.write("Welcome to the Image Conversion App!")
    st.write("This app allows you to perform various image conversions.")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Conversion options
        conversion_options = st.multiselect(
            "Select conversion options",
            ["Grayscale", "Black and White", "Resize to 1/2 Size", "Resize to 1/4 Size"]
        )

        if "Grayscale" in conversion_options:
            grayscale_image = grayscale_conversion(image)
            st.subheader("Grayscale Image")
            st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)

        if "Black and White" in conversion_options:
            threshold = st.slider("Threshold", 0, 255, 128)
            bw_image = black_and_white_conversion(image, threshold)
            st.subheader("Black and White Image")
            st.image(bw_image, caption="Black and White Image", use_column_width=True)

        if "Resize to 1/2 Size" in conversion_options:
            if "Grayscale" not in conversion_options:
                grayscale_image = grayscale_conversion(image)
            half_size_image = resize_image(grayscale_image, 0.5)
            st.subheader("Grayscale Image (1/2 Size)")
            st.image(half_size_image, caption="Grayscale Image (1/2 Size)", use_column_width=True)

        if "Resize to 1/4 Size" in conversion_options:
            if "Grayscale" not in conversion_options:
                grayscale_image = grayscale_conversion(image)
            quarter_size_image = resize_image(grayscale_image, 0.25)
            st.subheader("Grayscale Image (1/4 Size)")
            st.image(quarter_size_image, caption="Grayscale Image (1/4 Size)", use_column_width=True)

if __name__ == "__main__":
    main()

