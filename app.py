
import torch.nn.functional as F
from functools import partial
from collections import defaultdict, Counter
from contextlib import contextmanager
from pathlib import Path
import shutil
import time
import math
import os
import PIL
from torch.nn.parameter import Parameter
import torchvision.models as models
from torch.optim import Adam, SGD
import torch.nn as nn

from torchvision.models import ResNeXt50_32X4D_Weights
import gc
import base64
import pickle
import streamlit as st
import numpy as np
import pyparsing
import pandas as pd
import matplotlib as plt
import seaborn as sns
from PIL import Image, ImageOps
import tensorflow
from tensorflow.keras.models import load_model
import tensorflow as tf
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image
from streamlit_option_menu import option_menu
import torch
from torchvision.models import resnext50_32x4d

from torch.utils.data import DataLoader, Dataset
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as MobileNetV2_preprocess_input
import random
import warnings
warnings.filterwarnings("ignore")

# import albumentations as A
# from albumentations.pytorch import ToTensorV2


st.set_page_config(
    page_title="Leaf Disease Detection"
)
with st.sidebar:
    selection = option_menu(
        menu_title='Leaf Disease Detection System',
        options=['Introduction', 'Potato',
                 'Tomato',
                 'Wheat',
                 'Corn Leaf',
                 ],

        default_index=0,

    )

if (selection == 'Introduction'):
    st.title('Leaf Disease Detection')
    # st.subheader('this is the sub header')
    # st.write('this is small font')
    st.image("int.jpeg")
    st.header('Causes of Leaf Diseases')
    st.subheader('Fungal Infections:')
    st.write('Fungi are among the most common causes of leaf diseases. They can infect leaves through spores, which are spread by wind, water, or pests. Fungal diseases include powdery mildew, leaf spot, rust, and blight.')
    st.subheader('Bacterial Infections:')
    st.write('Bacteria can infect leaves through wounds or natural openings, causing diseases such as bacterial leaf spot, bacterial blight, and bacterial canker.')
    st.subheader('Viral Infections:')
    st.write('Viruses can infect leaves and cause diseases such as mosaic patterns, yellowing, and distorted growth. Viruses are often spread by insects, contaminated tools, or infected plant material.')
    st.subheader('Poor Cultural Practices:')
    st.write('Improper watering, over-fertilization, inadequate air circulation, overcrowding, and improper pruning can create conditions favorable for disease development. For example, overhead watering can promote fungal diseases by creating a moist environment on the leaves.')
if (selection == 'Potato'):
    # @st.cache(allow_output_mutation=True)
    def load_model():
        model = tf.keras.models.load_model('models/potato_model.h5')
        return model
    with st.spinner('Model is being loaded..'):
        model = load_model()

    st.write("""
            # Potato Leaf Disease Detection
            """
             )

    file = st.file_uploader("", type=["jpg", "png"])

    def import_and_predict(image_data, model):
        size = (256, 256)
        image = ImageOps.fit(image_data, size, Image.BICUBIC)
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        return prediction
    if file is None:
        st.warning("Please upload an image file")

    else:
        if st.button("Process"):
            image = Image.open(file)
            st.image(image, use_column_width=True)
            predictions = import_and_predict(image, model)
            class_names = ['Early blight', 'Late blight', 'Healthy']
            string = "Prediction : " + class_names[np.argmax(predictions)]
            if class_names[np.argmax(predictions)] == 'Healthy':
                st.success(string)
            elif class_names[np.argmax(predictions)] == 'Early blight':
                st.warning(string)
                st.write("Remedy")
                st.info("Early blight can be minimized by maintaining optimum growing conditions, including proper fertilization, irrigation, and management of other pests. Grow later maturing, longer season varieties. Fungicide application is justified only when the disease is initiated early enough to cause economic loss.")
            elif class_names[np.argmax(predictions)] == 'Late blight':
                st.warning(string)
                st.write("Remedy")
                st.info("Late blight is controlled by eliminating cull piles and volunteer potatoes, using proper harvesting and storage practices, and applying fungicides when necessary. Air drainage to facilitate the drying of foliage each day is important.")


if (selection == 'Tomato'):
    def prepare(file):
        img_array = file/255
        return img_array.reshape(-1, 128, 128, 3)

    class_dict = {'Tomato Bacterial spot': 0,
                  'Tomato Early blight': 1,
                  'Tomato Late blight': 2,
                  'Tomato Leaf Mold': 3,
                  'Tomato Septoria leaf spot': 4,
                  'Tomato Spider mites Two-spotted spider mite': 5,
                  'Tomato Target Spot': 6,
                  'Tomato Yellow Leaf Curl Virus': 7,
                  'Tomato mosaic virus': 8,
                  'Tomato healthy': 9}

    def prediction_cls(prediction):
        for key, clss in class_dict.items():
            if np.argmax(prediction) == clss:

                return key

    @st.cache_resource
    def load_image(image_file):
        img = Image.open(image_file)
        img = img.resize((128, 128))

        return img

    def main():
        st.title("Tomato Leaf Disease Prediction")
        image_file = st.file_uploader(
            "Upload Image", type=["png", "jpg", "jpeg"])

        if image_file == None:
            st.warning("Please upload an image first")
        else:

            if st.button("Process"):
                img = load_image(image_file)

                img = tf.keras.preprocessing.image.img_to_array(img)
                model = tf.keras.models.load_model("models/model_vgg19.h5")

                img = prepare(img)

                st.image(img, caption="Uploaded Image")
                st.subheader("Detected Disease :")
                if (prediction_cls(model.predict(img))) == 'Tomato healthy':
                    st.success("The plant is healthy")
                else:
                    st.warning(prediction_cls(model.predict(img)))

                if (prediction_cls(model.predict(img))) == 'Tomato Bacterial spot':
                    st.subheader("Remedies :")
                    st.write("Hot water treatment can be used to kill bacteria on and in seed. For growers producing their own seedlings, avoid over-watering and handle plants as little as possible. Disinfect greenhouses, tools, and equipment between seedling crops with a commercial sanitizer.")

                elif (prediction_cls(model.predict(img))) == 'Tomato Early blight':
                    st.subheader("Remedies :")
                    st.write("Cover the soil under the plants with mulch, such as fabric, straw, plastic mulch, or dried leaves. Water at the base of each plant, using drip irrigation, a soaker hose, or careful hand watering. Pruning the bottom leaves can also prevent early blight spores from splashing up from the soil onto leaves.")

                elif (prediction_cls(model.predict(img))) == 'Tomato Late blight':
                    st.subheader("Remedies :")
                    st.write("Spraying fungicides is the most effective way to prevent late blight. For conventional gardeners and commercial producers, protectant fungicides such as chlorothalonil (e.g., Bravo, Echo, Equus, or Daconil) and Mancozeb (Manzate) can be used.")

                elif (prediction_cls(model.predict(img))) == 'Tomato Leaf Mold':
                    st.subheader("Remedies :")
                    st.write("Applying fungicides when symptoms first appear can reduce the spread of the leaf mold fungus significantly. Several fungicides are labeled for leaf mold control on tomatoes and can provide good disease control if applied to all the foliage of the plant, especially the lower surfaces of the leaves.")

                elif (prediction_cls(model.predict(img))) == 'Tomato Septoria leaf spot':
                    st.subheader("Remedies :")
                    st.write("Fungicides are very effective for control of Septoria leaf spot and applications are often necessary to supplement the control strategies previously outlined. The fungicides chlorothalonil and mancozeb are labeled for homeowner use.")

                elif (prediction_cls(model.predict(img))) == 'Tomato Spider mites Two-spotted spider mite':
                    st.subheader("Remedies :")
                    st.write("Most spider mites can be controlled with insecticidal/miticidal oils and soaps. The oils—both horticultural oil and dormant oil—can be used. Horticultural oils can be used on perennial and woody ornamentals during the summer but avoid spraying flowers, which can be damaged.")

                elif (prediction_cls(model.predict(img))) == 'Tomato Target Spot':
                    st.subheader("Remedies :")
                    st.write("Many fungicides are registered to control of target spot on tomatoes. Growers should consult regional disease management guides for recommended products. Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control of target spot in research trials.")

                elif (prediction_cls(model.predict(img))) == 'Tomato Yellow Leaf Curl Virus':
                    st.subheader("Remedies :")
                    st.write("There is no treatment for virus-infected plants. Removal and destruction of plants is recommended. Since weeds often act as hosts to the viruses, controlling weeds around the garden can reduce virus transmission by insects.")

                elif (prediction_cls(model.predict(img))) == 'Tomato mosaic virus':
                    st.subheader("Remedies :")
                    st.write("There's no way to treat a plant with tomato spotted wilt virus. However, there are several preventative measures you should take to control thrips—the insects that transmit tomato spotted wilt virus. Weed, weed, and weed some more. Ensure that your garden is free of weeds that thrips are attracted to.")

    if __name__ == "__main__":
        main()

if (selection == 'Corn Leaf'):
    st.title('Corn Leaf Disease Detection')

    uploaded_image = st.file_uploader('', type=['jpg', 'png', 'jpeg'])

    # Image preprocessing
    if not uploaded_image:
        st.warning('Please upload an image before preceeding!')
        st.stop()
    else:
        img_as_bytes = uploaded_image.read()

        st.image(img_as_bytes, use_column_width=True)
        img = tf.io.decode_image(img_as_bytes, channels=3)
        img = tf.image.resize(img, (224, 224))

        img_arr = tf.keras.preprocessing.image.img_to_array(
            img)  # Convert image to array
        img_arr = tf.expand_dims(img_arr, 0)

        img = MobileNetV2_preprocess_input(img_arr)

        model = tf.keras.models.load_model("models/traing_model.h5")

        preds = model.predict(img)
        preds_class = model.predict(img).argmax()

        dict_class = {
            'Corn Leaf Condition': ['Northern Leaf Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy'],
            'Confiance': [0, 0, 0, 0]
        }

        df_results = pd.DataFrame(
            dict_class, columns=['Corn Leaf Condition', 'Confiance'])

        def predictions(preds):
            df_results.loc[df_results['Corn Leaf Condition'].index[0],
                           'Confiance'] = preds[0][0]
            df_results.loc[df_results['Corn Leaf Condition'].index[1],
                           'Confiance'] = preds[0][1]
            df_results.loc[df_results['Corn Leaf Condition'].index[2],
                           'Confiance'] = preds[0][2]
            df_results.loc[df_results['Corn Leaf Condition'].index[3],
                           'Confiance'] = preds[0][3]

            return (df_results)

        st.dataframe(predictions(preds))
        map_class = {
            0: 'Northern Leaf Blight',
            1: 'Common Rust',
            2: 'Gray Leaf Spot',
            3: 'Healthy'
        }

        if (map_class[preds_class] == "Northern Leaf Blight") or (map_class[preds_class] == "Common Rust") or (map_class[preds_class] == "Gray Leaf Spot"):

            st.subheader(" {} disease".format(map_class[preds_class]))
        if (map_class[preds_class] == "Northern Leaf Blight"):
            st.markdown('''Northern Leaf Blight (NLB) or Corn Blight is a fungal disease affecting corn plants. Its causes include:
                                    
            1. **Fungal Infection:** Blight is primarily caused by the fungus *Exserohilum turcicum* (formerly known as *Helminthosporium turcicum*).
            2. **Moisture and Temperature:** Warm, humid conditions favor the development of NLB.
            3. **Susceptible Corn Varieties:** Certain corn varieties are more prone to NLB infection.

            Remedies and treatments for NLB often involve a combination of cultural practices, fungicides, and resistant varieties:

            1. **Cultural Practices:**
            - **Crop Rotation:** Rotate crops to break the disease cycle as the fungus can survive on crop debris.
            - **Tillage:** Deep plowing can bury infected residue, reducing the fungal load.
            - **Spacing:** Proper plant spacing allows for better air circulation, minimizing moisture on leaves.
            - **Remove Infected Leaves:** Remove and destroy infected leaves to prevent further spread.

            2. **Resistant Varieties:** Planting corn varieties that are genetically resistant to NLB can significantly reduce the disease impact.

            3. **Fungicides:** Fungicides can be used to manage NLB, but their effectiveness depends on timing and the severity of the infection. Commonly used fungicides for NLB include:
            - **Azoxystrobin** (e.g., Quadris)
            - **Trifloxystrobin** (e.g., Flint)
            - **Propiconazole** (e.g., Tilt)

            It's crucial to follow the recommended application rates and safety precautions when using fungicides.

            Always consult with local agricultural extension services or professionals for specific product recommendations, application timing, and regulations in your area.''')
        elif (map_class[preds_class] == "Common Rust"):

            st.markdown('''Common Rust is a fungal disease affecting corn leaves. Its causes often involve environmental factors like high humidity, warm temperatures, and leaf wetness. Here are some suggestions for both preventing and treating Common Rust:

            ### Prevention:
            1. **Crop Rotation:** Avoid planting corn in the same area every year. Rotate crops to reduce disease buildup in the soil.
            2. **Resistant Varieties:** Choose corn varieties resistant to rust diseases.
            3. **Proper Spacing:** Ensure adequate spacing between plants to promote air circulation, reducing humidity around the leaves.
            4. **Fungicides:** Applying fungicides preventatively can help protect plants from rust. Fungicides like azoxystrobin, propiconazole, or tebuconazole are commonly used. Always follow the instructions on the product label.

            ### Treatment:
            1. **Fungicides:** Apply fungicides at the first signs of infection. Repeat applications as directed on the product label.
            2. **Pruning Infected Leaves:** Remove and destroy infected leaves to prevent the spread of spores.
            3. **Cultural Practices:** Implement good agricultural practices, such as removing crop debris after harvest to reduce overwintering sites for the fungus.

            However, when it comes to specific medicine names or products, it's essential to consult with agricultural extension services, local experts, or agronomists. They can provide recommendations based on your location, the severity of the infection, and any local regulations regarding fungicide use.

            Always read and follow the instructions and safety guidelines provided by the manufacturer when using any agricultural chemicals.''')

        elif (map_class[preds_class] == "Gray Leaf Spot"):
            st.markdown('''Grey leaf spot is a fungal disease that commonly affects corn plants. It's caused by the fungus *Cercospora zeae-maydis*. The primary causes of this disease include:

            1. **Moisture:** Extended periods of leaf wetness due to rain, irrigation, or high humidity create favorable conditions for fungal growth.
            2. **Warm temperatures:** Optimal temperatures between 75°F to 85°F (24°C to 29°C) encourage the development and spread of the fungus.
            3. **Residue management:** Infected crop debris left in the field can harbor the fungus, facilitating its recurrence in subsequent plantings.

            Remedies for grey leaf spot typically involve a combination of cultural and chemical methods:

            1. **Cultural practices:**
            - **Crop rotation:** Avoid planting corn in the same area repeatedly to reduce fungal buildup in the soil.
            - **Residue removal:** Remove and destroy infected plant debris after harvest to minimize overwintering of the fungus.
            - **Spacing:** Plant corn at recommended distances to promote air circulation, which can help reduce humidity around plants.

            2. **Chemical control:**
            - **Fungicides:** Application of fungicides containing active ingredients like azoxystrobin, trifloxystrobin, or chlorothalonil can help manage the disease. Specific product names and dosages may vary based on location and regulations, so it is essential to consult with a local agricultural extension office or expert for recommendations tailored to your area.When considering fungicide use, it is crucial to follow the manufacturer\'s instructions regarding application timing, dosage, and safety precautions. Always remember to prioritize preventive measures and integrated pest management strategies to minimize the reliance on chemicals and reduce the risk of fungicide resistance. If you are dealing with grey leaf spot or any specific plant disease, contacting a local agricultural extension service or a plant pathology expert can provide tailored advice for your region and the current condition of your crop.''')

        else:
            st.subheader("The Corn Leaf is {}".format(map_class[preds_class]))

if (selection == 'Wheat'):
    def prediction_cls(prediction):
        for key, clss in class_names.items():
            if np.argmax(prediction) == clss:
                return key

    st.title("Wheat Leaf Disease Detection")

    file = st.file_uploader("", type=["jpg", "png"])
    if not file:
        st.warning("Please Upload an Image")
        st.stop()

    else:
        img_as_bytes_wheat = file.read()

        model = tf.keras.models.load_model(
            'models/wheat.h5')

        def import_and_predict(image_data, model):
            size = (300, 300)
            image = ImageOps.fit(image_data, size)
            img = np.asarray(image)
            img_reshape = img[np.newaxis, ...]
            prediction = model.predict(img_reshape)
            return prediction
        if file is None:
            st.warning("Please upload an image file")
        else:
            if st.button("Process"):
                image = Image.open(file)
                st.image(image, use_column_width=True)
                predictions = import_and_predict(image, model)

                class_names = ['Healthy', 'spetoria', 'stripe_rust']

                string = "Detected Disease : " + \
                    class_names[np.argmax(predictions)]
                if class_names[np.argmax(predictions)] == 'Healthy':
                    st.balloons()
                    st.success(string)

                elif class_names[np.argmax(predictions)] == 'spetoria':
                    st.warning(string)
                    st.markdown("## Remedy")
                    st.info("These include a combination of triazoles + chlorothalonil at wheat stage T1 (1-2 node stage) or triazole + SDHI at wheat stage T2 (last leaf stage). However, new solutions are also available. These are perfectly appropriate for use in a septoria control strategy in combination with a triazole and/or chlorothalonil.")

                elif class_names[np.argmax(predictions)] == 'stripe_rust':
                    st.warning(string)
                    st.markdown("## Remedy")
                    st.info("Aviator® Xpro® and Prosaro® are both protective and curative fungicides, unlike some other fungicides which only offer protective properties against stripe rust. They are both registered for the control of stripe rust in wheat.")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
