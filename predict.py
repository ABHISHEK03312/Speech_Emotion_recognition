import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # filters CUDA and TF logs
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from split_and_cluster import split_and_cluster

def load_SER_model(filepath, verbose=1):
    model = keras.models.load_model(filepath)
    if verbose:
        print("\nLoaded SER Model with summary:")
        print(model.summary())
    return model

def split_and_predict(audio_filepath, SER_model_filepath="models/ser_175", siamese_model_filepath="models/base_siamese_175", verbose=1):
    scaled_features, timestamps, clustered_speakers = split_and_cluster(audio_filepath, siamese_model_filepath, verbose)
    SER_model = load_SER_model(SER_model_filepath, verbose)
    model_input = np.expand_dims(scaled_features, axis=2)
    preds = SER_model(model_input)
    pred_emotions = []
    emotion_dict = {1:"neutral", 2:"calm", 3:"happy", 4:"sad", 5:'angry', 6:'fearful', 7:'disgust', 8:'surprised'}
    for probs in preds:
        pred = np.argmax(probs)
        pred_emotions.append(pred+1)
    emotion_list = [emotion_dict[p] for p in pred_emotions]

    reverse_map = [None for _ in range(len(scaled_features))]
    for p_id, c_list in clustered_speakers.items():
        for c_id in c_list:
            reverse_map[c_id] = p_id
    
    print("\nThe conversation is structured as follows:")
    for c_id in range(len(emotion_list)):
        print("Person #{}: {} emotion from {:.2f}s to {:.2f}s".format((reverse_map[c_id]+1), emotion_list[c_id], timestamps[c_id][0], timestamps[c_id][1]))
    print()


if __name__ == "__main__":
    # audio_filepath = "mixed_data\mixed_clip_15_happy_18_sad_18_disgust_24_fearful.wav"
    # split_and_predict(audio_filepath, verbose=0)

    # highly inefficient for large number of data, as model gets loaded repeatedly each time
    for subdir, dirs, files in os.walk('mixed_data'):
        for file in files:
            temp_path = os.path.join(subdir, file)
            print(file)  # filename contains truth value
            split_and_predict(temp_path, verbose=0)
            print('------------------------------------------------------------------')
