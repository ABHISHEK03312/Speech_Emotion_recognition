import librosa
from librosa import feature
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.cluster import DBSCAN


def get_feature_vector(y, sr):
    feature_vector = []
    
    # multi-dim features
    feature_vector.extend(np.mean(feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1))
    feature_vector.extend(np.mean(feature.chroma_stft(y=y, sr=sr), axis=1))
    feature_vector.extend(np.mean(feature.spectral_contrast(y=y, sr=sr), axis=1))
    feature_vector.extend(np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr),axis=1))
    feature_vector.extend(np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1))
    
    # single-dim features with special requirements
    feature_vector.append(np.mean(feature.rms(y=y)))
    feature_vector.append(np.mean(feature.zero_crossing_rate(y=y)))
    feature_vector.extend([np.mean(x) for x in librosa.piptrack(y=y, sr=sr)])
    
    # single-dim features
    feat_list = [
        librosa.onset.onset_strength,
        feature.spectral_rolloff,
        feature.melspectrogram,
        feature.spectral_centroid,
        feature.spectral_bandwidth
    ]
    
    for temp_func in feat_list:
        feature_vector.append(np.mean(temp_func(y=y, sr=sr)))
    
    return feature_vector


def get_audio_split(filepath):
    x, sr=librosa.load(filepath, sr=None)
    # split audio with any audio signal lesser than 20db as mute. Minimum length of silence must be 0.5s
    nonMuteSections = librosa.effects.split(x, frame_length=sr//2)

    return x, sr, nonMuteSections


def get_audio_features(x, sr, nonMuteSections):
    audio_features = []
    audio_slices = []
    for i in range(nonMuteSections.shape[0]):
        current_slice_len = nonMuteSections[i][1]-nonMuteSections[i][0]
        ideal_len = int(3.7*sr)  # 3.7 second ideal clip len
        # finding maximum possible delta to add to both sides to make a clip have as close to 3s length as possible
        if ideal_len>current_slice_len:
            delta = int((ideal_len-current_slice_len)/2)
            delta = min(delta, nonMuteSections[i][0])
            delta = min(delta, (len(x)-1-nonMuteSections[i][1]))
            # making sure we don't add more than 0.5s, since that could spill over into another person's voice
            delta = min(sr//2, delta)  # assuming split was made on 0.5s of silence as minimum
        else:
            delta = 0
        if current_slice_len+2*delta < 2*sr:  # probably random noise, clip is too small even with delta
            continue
        slice=x[nonMuteSections[i][0]-delta:nonMuteSections[i][1]+delta]
        audio_slices.append([nonMuteSections[i][0]-delta, nonMuteSections[i][1]+delta])
        feature_vector = get_feature_vector(slice, sr)
        audio_features.append(feature_vector)
    
    return audio_features, audio_slices


def rescale_audio_features(audio_features):
    minMax = pd.read_pickle('Speaker_Classification_data/minMax.df')  # contains min and max for every feature
    scaled_features=[]
    for i in range(len(audio_features)):
        # calculating scaled features for a single slice
        scaled_slice=[]
        for j in range(len(audio_features[i])):
            # scaling each feature with the appropriate min and max
            scaled=(audio_features[i][j]-minMax[j]['min'])/(minMax[j]['max']-minMax[j]['min'])
            scaled_slice.append(scaled)
        scaled_features.append(scaled_slice)
    return scaled_features


def load_best_model(filepath, verbose=1):
    model = keras.models.load_model(filepath)
    if verbose:
        print("Summary of loaded model:")
        print(model.summary())
    return model


def cluster_speakers(model, scaled_features, verbose=1):
    model_inputs = [np.array([f]) for f in scaled_features]
    # initialize distance matrix with 1s
    distance_matrix = [[1 for _ in range(len(model_inputs))] for _ in range(len(model_inputs))]

    for i in range(len(model_inputs)):
        for j in range(i, len(model_inputs)):
            # getting cosine distance = 1 - cosine similarity (predicted by siamese model)
            temp_dist = 1 - model([model_inputs[i], model_inputs[j]])[0][0]
            temp_dist = max(temp_dist, 0)  # keeping everything >=0, sometimes dips below due to rounding errors otherwise
            distance_matrix[i][j] = temp_dist
            distance_matrix[j][i] = temp_dist

    if verbose:
        print("The distance matrix was computed as:")
        for row in distance_matrix:
            for item in row:
                print('%.2f'%item, end='\t')
            print()

    cluster = DBSCAN(eps=0.25, min_samples=1, metric='precomputed').fit(distance_matrix)
    pred_clusters = cluster.labels_
    
    clustered_speakers = {}  # dictionary mapping person id to list of clip ids
    for c_id, p_id in enumerate(pred_clusters):
        if p_id in clustered_speakers:
            clustered_speakers[p_id].append(c_id)
        else:
            clustered_speakers[p_id] = [c_id]
    
    return clustered_speakers


def split_and_cluster(audio_filepath, model_filepath="models/base_siamese_175", verbose=1):
    """Driver function for split_and_cluster.py"""
    x, sr, nonMuteSections = get_audio_split(audio_filepath)
    audio_features, audio_slices = get_audio_features(x, sr, nonMuteSections)
    scaled_features = rescale_audio_features(audio_features)
    model = load_best_model(model_filepath, verbose)
    clustered_speakers = cluster_speakers(model, scaled_features, verbose)

    if verbose:
        print('\nThe following speakers were found:')
        for p_id, c_list in clustered_speakers.items():
            print('Person #{} speaks between:'.format(p_id), end=' ')
            for c_id in c_list:
                slice_frames = audio_slices[c_id]
                timestamps = [fr/sr for fr in slice_frames]
                print("{0:.2f}s to {1:.2f}s;".format(timestamps[0], timestamps[1]), end=' ')
            print('\n')
    
    overall_timestamps = [[fr/sr for fr in slice_frames] for slice_frames in audio_slices]
    return scaled_features, overall_timestamps, clustered_speakers


if __name__ == "__main__":
    audio_filepath = "mixed_data\mixed_clip_15_happy_18_sad_18_disgust_24_fearful.wav"
    _ = split_and_cluster(audio_filepath)
