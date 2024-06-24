import streamlit as st
import numpy as np 
import tensorflow as tf 

# Creating Model 
def model_prediction(test_image):

    model = tf.keras.models.load_model('trained_model.keras')

    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    print(input_arr.shape)
    prediction = model.predict(input_arr)
    result_index1 = np.argmax(prediction)
    return result_index1
# Sidebar for Website
st.sidebar.title("VIPS-TC")
app_mode = st.sidebar.radio("Menu",["Home","About","Disease Detection"])       

# Designing Each page
if(app_mode == "Home"):
    st.header("PLANT DISEASE DETECTION SYSTEM")
    st.toast('Welcome to Disease Detection System!')
    image_path = "UI\Ai.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
Welcome to our Plant Disease Detection System, a cutting-edge solution for identifying plant diseases through the power of image recognition. With our platform, farmers, gardeners, and researchers can easily scan leaf images to quickly diagnose plant diseases, allowing for faster treatment and prevention of further damage.

## How It Works
Our system uses advanced machine learning algorithms to analyze leaf images and detect signs of various plant diseases. Simply upload an image of a plant leaf, and our system will identify any potential issues.

### Key Features
- **Simple and Fast:** Quickly upload an image and receive an analysis within seconds.
- **Wide Range of Diseases:** Detects a variety of common plant diseases, including fungal infections, bacterial infections, and viral diseases.
- **High Accuracy:** Trained on a large dataset of plant leaf images, our system provides accurate results.
- **Recommendations for Treatment:** Along with the diagnosis, receive recommendations on how to treat the detected diseases.

## Get Started
To start using our Plant Disease Detection System, follow these simple steps:
1. **Capture an Image:** Take a clear photo of the affected leaf. Ensure the image is in focus and well-lit.
2. **Upload the Image:** Use our website's upload feature to submit your image for analysis.
3. **Receive Diagnosis:** Within seconds, get a report detailing any detected diseases and suggested treatments.

## Supported Plants and Diseases
Our system currently supports a wide range of plants, including:
- Tomatoes
- Potatoes
- Citrus fruits
- Roses
- And more...

We also detect common diseases such as:
- Powdery mildew
- Black spot
- Leaf rust
- Leaf curl
- And many others...

---
                
© 2024 Plant Disease Detection System. All rights reserved. 
                """)
    
# About Page
elif(app_mode == "About"):
    st.image("UI\Disease image.png")
    st.markdown("""
    **This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.**
    
    \n
    1.Train images = 17289\n
    2.Valid images = 18240\n
    3.Test Images = 33\n
    
## Contact Us
If you have any questions or need support, please reach out to our team:
- Email: support@plantdiseasedetection.com
- Phone: (123) 456-7890

Follow us on [Twitter](https://twitter.com/plantdiseasedetection) and [Instagram](https://instagram.com/plantdiseasedetection) for the latest updates.

## Privacy Policy
We take your privacy seriously. Please review our [Privacy Policy](privacy-policy.html) to understand how we handle your data.

## Terms of Service
By using our service, you agree to our [Terms of Service](terms-of-service.html).            

""")
    

# Prediction Page
elif(app_mode=="Disease Detection"):
    st.header("PLANT DISEASE DETECTION SYSTEM")
    st.image("UI\PlantImange.jpg",use_column_width=True)
    test_image = st.file_uploader("Choose the image :") 
    if(st.button("Show Image ")):
        st.image(test_image,use_column_width=True)
        
    if(st.button("Predict Disease")):
        with st.spinner("Model is working !"):
            result_index = model_prediction(test_image)
            #Define Classes
            class_name = ['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']
            st.success("Detected Disease is {}".format(class_name[result_index]))