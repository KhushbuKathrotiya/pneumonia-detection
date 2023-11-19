# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import torch
import pydicom
import numpy as np
from PIL import Image
from torchvision import transforms

LOGGER = get_logger(__name__)


def load_and_preprocess_dcm(file):
    # dcm = pydicom.dcmread(file)
    # image = dcm.pixel_array.astype(np.float32)
    dcm = pydicom.read_file(file).pixel_array
    # dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)
    resized_image = Image.fromarray(dcm / 255).resize((224, 224))
    display = Image.fromarray(dcm)
    display = display.convert("L")
    
    # Preprocess your image as needed (e.g., resizing, normalization)
    # For simplicity, let's assume a model that takes a fixed-size input
    # and normalizes pixel values to the range [0, 1].
    preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.49, 0.248)])
    img = preprocess(resized_image)
    # img = np.expand_dims(img, axis=0)
    return np.expand_dims(img, axis=0), display

# Function to load your pre-trained model and make predictions
def predict(image):
    # Load your pre-trained model here
    # Replace the following line with your actual model loading code
    device = torch.device("cpu")
    model = torch.jit.load('model_scripted.pt').to(device)
    model.eval()
    # st.write("Model loaded...")
    with torch.no_grad():
        data = torch.from_numpy(image)
        pred_prob = torch.sigmoid(model(data)[0].cpu())

    return pred_prob

def main():
    st.title("RSNA Pneumonia Detection")

    uploaded_file = st.file_uploader("Choose a DICOM file", type=["dcm"])

    if uploaded_file is not None:
        # st.image(uploaded_file)

        # Process the DICOM image
        processed_image, display = load_and_preprocess_dcm(uploaded_file)

        # Make predictions
        prediction = predict(processed_image)
        if prediction<0.5:
            st.write("Prediction: No Pneumonia")
        else:
            st.write("Pneumonia detected with <b>"+str(round(prediction.item(), 4))+"</b> probabilty.", unsafe_allow_html=True)  # Display the prediction result
        st.image(display, caption='DICOM Image uploaded', use_column_width=True)

if __name__ == "__main__":
    main()
