import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from discharge_model_tf import mish


def load_Severson(training=True, part='discharge'):
    """
    import the training and testing set
    """
    folder_path = '../Severson_Dataset/feature_selector_discharge/'
    index = range(500) if part == 'charge' else range(500, 1000)
    if training:
        feature, target = np.load(folder_path+'trn_features.npy')[:, :, index], np.load(folder_path+'trn_targets.npy')
    else:
        feature, target = np.load(folder_path+'val_features.npy')[:, :, index], np.load(folder_path+'val_targets.npy')
    target[:, 0] = np.log2(target[:, 0]) # 對EOL取對數
    return normalize(feature).transpose(0, 2, 1), normalize(target)


def get_scaler(pred_target='EOL'):
    """
    use training set to fit the scaler
    """
    assert pred_target=='EOL' or pred_target=='chargetime' or pred_target=='both' 
    scaler_x, scaler_y = MinMaxScaler(), StandardScaler()
    feature =  np.load('../Severson_Dataset/feature_selector_discharge/trn_features.npy')[:, :, 500:]
    target = np.load('../Severson_Dataset/feature_selector_discharge/trn_targets.npy')
    scaler_x.fit(feature.transpose((0, 2, 1)).reshape(-1, 4))
    target[:, 0] = np.log2(target[:, 0]) # 對EOL取對數

    if pred_target=='EOL':
        target = target[:, 0].reshape(-1, 1)
    elif pred_target=='chargetime':
        target = target[:, 1].reshape(-1, 1)

    scaler_y.fit(target)
    return scaler_x, scaler_y


def normalize(data, length=500):
    scaler_x, scaler_y = get_scaler('both')
    c = data.shape[1] # channel
    if c>2: # feature
        data = scaler_x.transform(data.transpose((0, 2, 1)).reshape(-1, c)).reshape(-1, length, c).transpose((0, 2, 1))
    else: # target
        data = scaler_y.transform(data)
    return data


def predictor1_preprocessing(ep=[20, 14], norm=True):
    folder='../Severson_Dataset/feature_selector_discharge/'
    model_path = [f'checkpoints/Dim_Reduction_1_ep{ep[0]}.h5', f'checkpoints/Dim_Reduction_2_ep{ep[1]}.h5']
    # load model
    selector1 = tf.keras.models.load_model(model_path[0], custom_objects={'mish': mish}, compile=False)
    selector2 = tf.keras.models.load_model(model_path[1], custom_objects={'mish': mish}, compile=False)
    # load dataset
    trn_set = load_Severson(training=True)[0]
    val_set = load_Severson(training=False)[0]
    trn_summary = np.load(folder+'trn_summary.npy')
    val_summary = np.load(folder+'val_summary.npy')
    trn_target = np.load(folder+'trn_targets.npy')
    val_target = np.load(folder+'val_targets.npy')
    trn_size, val_size = len(trn_summary), len(val_summary)
    trn_feature = np.zeros((trn_size, 100, 8))
    val_feature = np.zeros((val_size, 100, 8))
    trn_feature[:, :, :6] = trn_summary.transpose(0, 2, 1)
    val_feature[:, :, :6] = val_summary.transpose(0, 2, 1)
    target_index = [np.arange(trn_size)*100, np.arange(val_size)*100]
    target_scaler = get_scaler('both')[1]
    # feature generate
    for i in range(len(trn_summary)):
        trn_feature[i, :, 6] = selector1.predict(trn_set[(100*i):(100*(i+1))]).reshape(-1)
        trn_feature[i, :, 7] = selector2.predict(trn_set[(100*i):(100*(i+1))]).reshape(-1)
    for i in range(len(val_summary)):
        val_feature[i, :, 6] = selector1.predict(val_set[(100*i):(100*(i+1))]).reshape(-1)
        val_feature[i, :, 7] = selector2.predict(val_set[(100*i):(100*(i+1))]).reshape(-1)
    if norm:
        feature_max, feature_min = np.max(np.max(trn_feature, axis=0), axis=0), np.min(np.min(trn_feature, axis=0), axis=0)
        for i in range(8):
            trn_feature[:, :, i] = (trn_feature[:, :, i]-feature_min[i])/(feature_max[i]-feature_min[i])
            val_feature[:, :, i] = (val_feature[:, :, i]-feature_min[i])/(feature_max[i]-feature_min[i])
    # target preprocessing
    trn_target[:, 0] = np.log2(trn_target[:, 0])
    val_target[:, 0] = np.log2(val_target[:, 0])
    trn_target, val_target = target_scaler.transform(trn_target[target_index[0]]), target_scaler.transform(val_target[target_index[1]])
    return trn_feature, val_feature, trn_target, val_target


# predictor1_preprocessing()