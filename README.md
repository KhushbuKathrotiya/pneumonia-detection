# Pneumonia Detection on Chest RadioGraph Using Streamlit
**Description:**
This Streamlit code defines a web application for RSNA pneumonia detection. It uses the Streamlit library to create a user interface.It serves as an easily deployable web interface to upload DICOM files, preprocess the images, and make predictions using a pre-trained model. The main functionalities include loading and preprocessing DICOM images using the load_and_preprocess_dcm function, predicting pneumonia presence with the predict function, and displaying the result alongside the uploaded DICOM image. The application is encapsulated in the main function, utilizing Streamlit for building the web interface. Users can run the application locally, and the predictions, along with the DICOM image, are dynamically displayed in the web interface. Additionally, it references loading a pre-trained model from a TorchScript file ('model_scripted.pt') and incorporates basic image preprocessing using the PyTorch and Pillow libraries.


Step 1: Run the command pip install -t requirement.txt, which installs all the required libraries.

Step 2: Run hello.py file -> shows interfece as below

<img width="1440" alt="Screenshot 2023-11-18 at 9 07 42 PM" src="https://github.com/KhushbuKathrotiya/pneumonia-detection/assets/63412542/5c4f8b60-7e75-4e1c-b93c-840952e11e58">


Step 3: Upload any.dicom file, preprocess your image as needed, and pass the image to the predtct function, in which we already loaded the model using the torch.jit.load function. Then the model predicts the probability: if the probability is less than 0.5, it predicts "no pneumonia"; otherwise, it shows "pneumonia detected with probability".

  case 1:
<img width="1440" alt="Screenshot 2023-11-18 at 3 17 02 PM" src="https://github.com/KhushbuKathrotiya/pneumonia-detection/assets/63412542/2ad1d7ca-0f9b-4027-85e8-bb3bdc025e11">

  case 2:
<img width="1440" alt="Screenshot 2023-11-18 at 3 16 48 PM" src="https://github.com/KhushbuKathrotiya/pneumonia-detection/assets/63412542/b47e0537-6db3-480a-b382-5f79fe62be87">
