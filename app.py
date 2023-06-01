import streamlit as st
import cv2
import numpy as np
import tempfile

def grayscale_conversion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def black_white_conversion(image):
    gray = grayscale_conversion(image)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return bw

def gray_half_conversion(image):
    gray = grayscale_conversion(image)
    dither = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return dither

def gray_quarter_conversion(image):
    gray = grayscale_conversion(image)
    dither = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return dither

def download_image(image, format):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}') as temp:
        temp.write(image)
        return temp.name

def main():
    st.title("Image Converter App")
    st.write("Upload an image and select the conversion type.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "gif"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.subheader("Original Image")
        st.image(image, channels="BGR")

        if st.button("Convert to Grayscale"):
            gray = grayscale_conversion(image)
            st.subheader("Grayscale Image")
            st.image(gray, cmap="gray")

            converted_image = cv2.imencode(".jpg", gray)[1].tobytes()
            download_link = download_image(converted_image, "jpg")
            st.download_button("Download Grayscale Image", download_link)

        if st.button("Convert to Black and White"):
            bw = black_white_conversion(image)
            st.subheader("Black and White Image")
            st.image(bw, cmap="gray")

            converted_image = cv2.imencode(".jpg", bw)[1].tobytes()
            download_link = download_image(converted_image, "jpg")
            st.download_button("Download Black and White Image", download_link)

        if st.button("Convert to Gray 1/2"):
            half = gray_half_conversion(image)
            st.subheader("Gray 1/2 Image")
            st.image(half, cmap="gray")

            converted_image = cv2.imencode(".jpg", half)[1].tobytes()
            download_link = download_image(converted_image, "jpg")
            st.download_button("Download Gray 1/2 Image", download_link)

        if st.button("Convert to Gray 1/4"):
            quarter = gray_quarter_conversion(image)
            st.subheader("Gray 1/4 Image")
            st.image(quarter, cmap="gray")

            converted_image = cv2.imencode(".jpg", quarter)[1].tobytes()
            download_link = download_image(converted_image, "jpg")
            st.download_button("Download Gray 1/4 Image", download_link)

if __name__ == "__main__":
    main()
