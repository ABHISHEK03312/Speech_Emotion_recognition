# Speech Emotion Recognition in Conversations
This repository aims to recognise the various emotions in conversations exhibited by various speakers.


## Setup
Install Dependencies from ```requirements.txt```
```
pip install -r requirements.txt
```
## Experiments, Models and Data Preperation
The dataset used for the model is the [RAVDESS Dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio). Rename the downloaded directory to audio-files, outside the root folder, only if you wish to retrain.

## Run Final Predictions
```predict.py``` contains the functions to provide predictions for a
```
python predict.py
```

## Experiments, Models and Data Preperation
The various experiments, data generation scripts and other functions can be found in the various notebooks. 
We run the notebooks in the following order if you wish to retrain models or regenerate data. do note that this may take a while to run. Data generation can take upto 45 minutes or more, and model training can take around 30 minutes. 

####The training data and many other data files are not included in the repository owing to large size.

```
Preprocessing.ipynb
Generating Mixed Clips.ipynb
Training SER Model.ipynb
Training Siamese Model.ipynb
```

