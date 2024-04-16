import streamlit as stream
import cv2 as cv
from PIL import Image
import numpy as num
import matplotlib.pyplot as plotting


# SETTING THE PAGE CONFIGURATION
stream.set_page_config(
    page_title="IMAGEPROCESSING",
    page_icon=":camera",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FUNCTIONS TO PERFROM IMAGE PROCESSING OPERATIONS
def process_image(image, operations):
    processed_image = image.copy()
    for operation in operations:
        if operation == 'Resize':
            new_size = (stream.sidebar.number_input('NEW WIDTH:', min_value=1), stream.sidebar.number_input('NEW HEIGHT:', min_value=1))
            processed_image = cv.resize(processed_image, new_size, interpolation=cv.INTER_LINEAR)
        elif operation == 'Crop':
            x1 = stream.sidebar.number_input('X1 COORDINATES:', min_value=0, max_value=image.shape[1]-1)
            y1 = stream.sidebar.number_input('Y1 COORDINATES:', min_value=0, max_value=image.shape[0]-1)
            x2 = stream.sidebar.number_input('X2 COORDINATES:', min_value=x1+1, max_value=image.shape[1])
            y2 = stream.sidebar.number_input('Y2 COORDINATES:', min_value=y1+1, max_value=image.shape[0])
            processed_image = processed_image[y1:y2, x1:x2]
        elif operation == 'Rotate':
            angle = stream.sidebar.slider('ANGLE :', min_value=0, max_value=360, value=0)
            rows, cols = processed_image.shape[:2]
            M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            processed_image = cv.warpAffine(processed_image, M, (cols, rows))
        elif operation == 'Translate':
            tx = stream.sidebar.slider('TRANSLATION IN X:', min_value=-image.shape[1], max_value=image.shape[1], value=0)
            ty = stream.sidebar.slider('TRANSLATION IN Y:', min_value=-image.shape[0], max_value=image.shape[0], value=0)
            M = num.float32([[1, 0, tx], [0, 1, ty]])
            processed_image = cv.warpAffine(processed_image, M, (image.shape[1], image.shape[0]))
        elif operation == 'Blur':
            blur_type = stream.sidebar.selectbox('BLUR TYPE:', ['Gaussian Blur', 'Median Blur', 'Average Blur'])
            if blur_type == 'Gaussian Blur':
                blur_size = stream.sidebar.slider('BLUR SIZE:', min_value=1, max_value=31, step=2, value=5)
                processed_image = cv.GaussianBlur(processed_image, (blur_size, blur_size), 0)
            elif blur_type == 'Median Blur':
                blur_size = stream.sidebar.slider('Kernel Size:', min_value=1, max_value=31, step=2, value=5)
                processed_image = cv.medianBlur(processed_image, blur_size)
            elif blur_type == 'Average Blur':
                blur_size = stream.sidebar.slider('Kernel Size:', min_value=1, max_value=31, step=2, value=5)
                processed_image = cv.blur(processed_image, (blur_size, blur_size))
        elif operation == 'Change Color':
            channel = stream.sidebar.selectbox('Select Color Channel:', ['Red', 'Green', 'Blue'])
            value = stream.sidebar.slider('New Value:', min_value=0, max_value=255, value=0)
            if channel == 'Red':
                processed_image[:, :, 2] = value
            elif channel == 'Green':
                processed_image[:, :, 1] = value
            elif channel == 'Blue':
                processed_image[:, :, 0] = value
        elif operation == 'Thresholding':
            threshold_type = stream.sidebar.selectbox('Threshold Type:', ['Binary', 'Binary Inverted', 'Truncated', 'To Zero', 'To Zero Inverted'])
            threshold_value = stream.sidebar.slider('Threshold Value:', min_value=0, max_value=255, value=128)
            gray_image = cv.cvtColor(processed_image, cv.COLOR_BGR2GRAY)
            _, processed_image = cv.threshold(gray_image, threshold_value, 255, getattr(cv, f'THRESH_{threshold_type.upper()}'))
        elif operation == 'Grayscale':
            processed_image = cv.cvtColor(processed_image, cv.COLOR_BGR2GRAY)

    return processed_image

# MAIN FUNCTION
def main():

    # TITLE AND FILE UPLOAD 
    stream.title("IMAGE PROCESSING")
    uploaded_file = stream.file_uploader("UPLOAD AN IMAGE", type=["jpg", "jpeg", "png"])

    # SIDEBAR OPTIONS
    stream.sidebar.header("IMAGE PROCESSING OPTIONS")
    operations = stream.sidebar.multiselect("OPERATIONS:", ['Resize', 'Crop', 'Rotate', 'Translate', 'Blur', 'Change Color', 'Thresholding', 'Grayscale'])

    if uploaded_file is not None:
        image = num.array(Image.open(uploaded_file))
        processed_image = process_image(image, operations)

        stream.subheader("ORIGINAL IMAGE")
        stream.image(image, caption='Original Image', use_column_width=True)

        stream.subheader("PROCESSED IMAGE")
        stream.image(processed_image, caption='Processed Image', use_column_width=True)

# MAIN OPERATION INITIALITION POINT 
if __name__ == "__main__":
    main()

