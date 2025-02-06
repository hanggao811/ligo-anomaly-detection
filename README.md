# LIGO Anomaly Detection Project

## Overview
This project detects anomalies in LIGO gravitational wave data using a custom TensorFlow/Keras ensemble model. The model combines predictions from three pre-trained autoencoders with learned weights. The details of the challenge can be found in [this Codabench competition](https://www.codabench.org/competitions/2626/).

---

## Setup
1. **Clone this repository**:
   ```bash
   git clone https://github.com/hanggao811/ligo-anomaly-detection.git
   cd ligo-anomaly-detection

2. Install dependencies:
    pip install -r requirements.txt

Data Preparation
Place your input data (e.g., trial_data.hdf5) in the data/ folder.

Directory and File Descriptions
Submission_file/
Contains files required for submission
model_(91.0)(91.1).keras
Pre-trained model version 1 for anomaly detection in LIGO data.
model_(91.9)(88.8).keras
Pre-trained model version 2 for anomaly detection in LIGO data.
model_(94.0)(90.8).keras
Pre-trained model version 3 for anomaly detection in LIGO data.
model.keras
A model designed to distinguish anomalies from background noise using the predictions from models 1, 2, and 3.
model.ipynb
A Jupyter notebook for testing the pre-trained models using custom input data, guiding users through loading the models and making predictions.
data/
Directory containing input datasets. Some files were deleted due to size constraints but can be downloaded at the challenge website as needed:
background.npz: Contains background noise data used for training or testing.
bbh_for_challenge.npy: Data representing binary black hole signals for model training/testing.
sglf_for_challenge.npy: Simulated sine-Gaussian signals used for anomaly detection training.
requirements.txt
List of Python dependencies necessary to run the models and notebooks. 
TransformerHDR1.1.ipynb
A Jupyter notebook for training the model 1,2,3 though the parameters need be tuned
TransformerHDRfinal.ipynb
A Jupyter notebook for training the transformer-based anomaly detection model from scratch using the provided datasets.
README.md

Instructions for setting up the project, installing dependencies, and testing the models using input data.
To Test with Your Data
1.Open model.ipynb
2.Modify the input path to your data(The data, due to the requirements of the challenge, should be in the shape of (N, 200, 2),
where 200 represents the time steps, and 2 corresponds to the LIGO Livingston and LIGO Hanford data.)
input_data = load_hdf5_data("../trial_data.hdf5")
3.Initialize the model and load the pre-trained weights or configuration
model = Model()
model.load()
4.Predict using the model and the provided input data
predictions = model.predict(input_data)