import streamlit as st
from PIL import Image
import numpy as np
import cv2
# import tensorflow as
import tensorflow as tf
import os
import cv2
import time
from PIL import Image


def process_image(img,img_array,imageName):
    # model = tf.keras.models.load_model(os.path.join(os.getcwd(),"files","non aug","unet-non-aug.h5"))
    model = tf.keras.models.load_model(os.path.join(os.getcwd(),"files","unet-aug.h5"))
    img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(os.path.join(os.getcwd(),"input",imageName),img_array_bgr)
    x = img_array_bgr/255.0
    x = np.expand_dims(x,axis=0)
    start_time = time.time()
    p = model.predict(x)[0]
    end_time = time.time()
    print("time taken : ",end_time-start_time)
    p = p>0.5
    p = p*128
    cv2.imwrite("output.jpg",p)
    output = Image.open("output.jpg")
    return output


def main():
    st.set_page_config(layout="wide")

    st.title("Vine net")
    st.markdown("""<style>
    .exg6vvm15 .edgvbvh10{
        display : none
    }
    .block-container{
        padding:2rem
    }
    .exg6vvm0{
        padding : 0rem 3rem
    }
    # .e1tzin5v4{
    #     padding : 0rem 6rem
    # }
    .stButton{
        # display: flex;
        # flex-direction: row-reverse;
        padding: 0rem 3rem;
    }
    .e19lei0e1{
        display : none
    }
    .e1tzin5v4{
        padding : 0rem 4rem
    }
    .etr89bj1{
        padding : 0rem 3rem
    }
    .e16nr0p30{
        padding : 0rem 3rem
    }
    </style>""", unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        original_image = Image.open(uploaded_image)
        # st.image(original_image)
        img_array = np.array(original_image)

    col1, col2 = st.columns(2)
    if 'original_image' in locals():
        with col1:
            st.image(original_image, caption="Original Image", use_column_width=True)
            if st.button("Process Image"):
                processed_img = process_image(original_image,img_array,uploaded_image.name)

    # Display processed image if it exists
    if 'processed_img' in locals():
        with col2:
            st.image(processed_img, caption="Processed Image", use_column_width=True)

    # Display original image if it exists
   
            # st.image(processed_img)
            # st.balloons()



if __name__ == "__main__":
    main()
