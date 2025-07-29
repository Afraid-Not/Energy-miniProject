import numpy as np
import pandas as pd
import random 
import tensorflow as tf
import keras.backend as K

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def smape_tf(y_true, y_pred):
    numerator = K.abs(y_pred - y_true)
    denominator = (K.abs(y_pred) + K.abs(y_true)) + K.epsilon()
    return 200.0 * K.mean(numerator / denominator)

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)

    Parameters:
    - y_true: 정답값 (array-like)
    - y_pred: 예측값 (array-like)

    Returns:
    - SMAPE score (%)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) + 1e-8  # 분모 0 방지
    diff = np.abs(y_pred - y_true)
    smape = np.mean(2.0 * diff / denominator) * 100
    return smape
    
def create_sequences_stride(data, target_cols, time_steps=24, stride=24):
    X, Y = [], []
    feature_cols = [col for col in data.columns if col not in target_cols]

    for i in range(0, len(data) - time_steps, stride):
        x_seq = data[feature_cols].iloc[i:i+time_steps].values
        y_seq = data[target_cols].iloc[i:i+time_steps].values
        X.append(x_seq)
        Y.append(y_seq)

    return np.array(X), np.array(Y)

data_path = './Energy/data/'

train = pd.read_csv(data_path + 'train_new.csv')
test = pd.read_csv(data_path + 'test_new.csv')
building = pd.read_csv('./Energy/building_info.csv')
submit = pd.read_csv('./Energy/sample_submission.csv')

print(train.shape)
print(train.columns)
print(test.shape)
print(test.columns)
print(building.columns)
















