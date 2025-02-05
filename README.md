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

Directory structure after setup:
ligo-anomaly-detection/
├── Submission_file/ #file consist the pre-trained models and the codes
│   ├── model_(91.0)(91.1).keras    # Pre-trained model 1
│   ├── model_(91.9)(88.8).keras    # Pre-trained model 2
│   ├── model_(94.0)(90.8).keras    # Pre-trained model 3
│   ├── model_2.keras    # model classifies the anomaly from the background with model 1,2,3
│   ├── model.ipynb    # Notebook for testing with your data
│   └── TransformerHDRfinal.ipynb    # Notebook for training the model
├── data/                
│   ├── background.npz    # was deleted due to file size, can be downloaded through the challenge website if needed
│   ├── bbh_for_challenge.npy   # was deleted due to file size, can be downloaded through the challenge website if needed
│   └── sglf_for_challenge.npy    #was deleted due to file size, can be downloaded through the challenge website if needed
├── requirements.txt         # Python dependencies, will be added soon🤔
└── README.md

Usage
Test with Your Data
Open model.ipynb
# Modify the input path to your data
# The data, due to the requirements of the challenge, should be in the shape of (N, 200, 2),
# where 200 represents the time steps, and 2 corresponds to the LIGO Livingston and LIGO Hanford data.
input_data = load_hdf5_data("../trial_data.hdf5")

# Initialize the model and load the pre-trained weights or configuration
model = Model()
model.load()

# Predict using the model and the provided input data
predictions = model.predict(input_data)