# Pneumonia Detection GUI using Streamlit

This Streamlit code defines a web application for RSNA Pneumonia Detection.It uses the Streamlit library to create a user interface.It serves as an easily deployable web interface to upload DICOM files, preprocess the images, and make predictions using a pre-trained model

The main functionalities include loading and preprocessing DICOM images using the load_and_preprocess_dcm function, predicting pneumonia presence with the predict function, and displaying the result alongside the uploaded DICOM image.

The application is encapsulated in the main function, utilizing Streamlit for building the web interface. Users can run the application locally, and the predictions, along with the DICOM image, are dynamically displayed in the web interface.

Additionally, it references loading a pre-trained model from a TorchScript file ('model_scripted.pt') and incorporates basic image preprocessing using the PyTorch and Pillow libraries.
