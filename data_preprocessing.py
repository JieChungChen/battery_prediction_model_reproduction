import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class Feature_Selector_Dataset(Dataset):
    def __init__(self, train=True, pred_target='EOL', part='discharge', norm=True):
        """
        train(bool): training or testing set
        pred_target(str): 'EOL' or 'chargetime'
        part(str): 'discharge' or 'charge'
        norm(bool): apply normalizarion to target 
        """
        self.train = train
        self.pred_target = pred_target
        self.input, self.target = load_Severson(training=train, norm=norm, part=part)

    def __getitem__(self, index):
        target_id = 0 if self.pred_target=='EOL' else 1
        feature, target = self.input[index], self.target[index, target_id]
        return feature, target

    def __len__(self):
        return len(self.input)

    def visualize(self, index, feature_id):
        feature_list =  ['Voltage', 'Discharge capacity', 'Current', 'Temperature']
        curve = self.input[index, feature_id, :]
        plt.plot(np.arange(len(curve)), curve, c='red')
        plt.ylabel(feature_list[feature_id], fontsize=14)
        plt.xlabel('time', fontsize=14)
        plt.show()
        plt.close()


class Predictor1_Dataset(Dataset):
    def __init__(self, train=True, last_padding=True):
        self.train = train
        self.fix = False
        self.f_length = 0
        folder = 'Severson_Dataset/feature_selector_discharge/'
        self.trn_input, self.trn_target = np.load(folder+'predictor1_trn_feature.npy'), np.load(folder+'trn_targets.npy')
        self.val_input, self.val_target = np.load(folder+'predictor1_val_feature.npy'), np.load(folder+'val_targets.npy')
        trn_size, val_size = len(self.trn_input), len(self.val_input)
        target_index = [np.arange(trn_size)*100, np.arange(val_size)*100]
        target_scaler = get_scaler('both')[1]
        feature_mean, feature_std = np.mean(np.mean(self.trn_input[:, :6, :], axis=0), axis=1), np.std(self.trn_input[:, :6, :].transpose((1, 0, 2)).reshape(6, -1), axis=1)
        for i in range(6):
            self.trn_input[:, i, :] = (self.trn_input[:, i, :].copy()-feature_mean[i])/feature_std[i]
            self.val_input[:, i, :] = (self.val_input[:, i, :].copy()-feature_mean[i])/feature_std[i]
        self.trn_target, self.val_target = target_scaler.transform(self.trn_target[target_index[0]]), target_scaler.transform(self.val_target[target_index[1]])
        # self.trn_target, self.val_target = self.trn_target[target_index[0]], self.val_target[target_index[1]]

        if last_padding: # full last padding
            aug_trn_input, aug_trn_target = [], []
            for i in range(trn_size):
                for cycle_length in range(100):
                    after_padding = self.trn_input[i].copy()
                    after_padding[:, cycle_length:] = after_padding[:, cycle_length].reshape(-1, 1).repeat(100-cycle_length, axis=1)
                    aug_trn_input.append(after_padding)
                    aug_trn_target.append(self.trn_target[i, :])
            self.trn_input, self.trn_target = np.stack(aug_trn_input, axis=0), np.stack(aug_trn_target, axis=0)

    def __getitem__(self, index):
        if self.train:
            feature, target = self.trn_input[index], self.trn_target[index]
        else:
            feature, target = self.val_input[index], self.val_target[index]
        return feature, target

    def __len__(self):
        if self.train:
            return len(self.trn_input)
        return len(self.val_input)

    def visualize(self, index, feature_id):
        feature_list =  ['charge capacity', 'discharge capacity', 'chargetime', 'TAvg', 'TMin', 'TMax', 'EOL feature', 'chargetime feature']
        curve = self.trn_input[index, feature_id, :]
        plt.plot(np.arange(len(curve)), curve, c='red')
        plt.ylabel(feature_list[feature_id], fontsize=14)
        plt.xlabel('time', fontsize=14)
        plt.show()
        plt.close()


class Predictor3_Dataset(Dataset):
    def __init__(self, train=True, last_padding=True):
        self.train = train
        self.fix = False
        self.f_length = 0
        folder = 'Severson_Dataset/feature_selector_discharge/'
        self.trn_input, self.trn_target = np.load(folder+'predictor3_trn_feature.npy'), np.load(folder+'trn_targets.npy')
        self.val_input, self.val_target = np.load(folder+'predictor3_val_feature.npy'), np.load(folder+'val_targets.npy')
        trn_size, val_size = len(self.trn_input), len(self.val_input)
        target_index = [np.arange(trn_size)*100, np.arange(val_size)*100]
        target_scaler = get_scaler('both')[1]
        feature_mean, feature_std = np.mean(np.mean(self.trn_input[:, :6, :], axis=0), axis=1), np.std(self.trn_input[:, :6, :].transpose((0, 2, 1)).reshape(-1, 6), axis=0)
        for i in range(6):
            self.trn_input[:, i, :] = (self.trn_input[:, i, :]-feature_mean[i])/feature_std[i]
            self.val_input[:, i, :] = (self.val_input[:, i, :]-feature_mean[i])/feature_std[i]
        self.trn_target, self.val_target = target_scaler.transform(self.trn_target[target_index[0]]), target_scaler.transform(self.val_target[target_index[1]])
        # self.trn_target, self.val_target = self.trn_target[target_index[0]], self.val_target[target_index[1]]

        if last_padding: # full last padding
            aug_trn_input, aug_trn_target = [], []
            for i in range(trn_size):
                for cycle_length in range(100):
                    after_padding = self.trn_input[i].copy()
                    after_padding[:, cycle_length:] = after_padding[:, cycle_length].reshape(-1, 1).repeat(100-cycle_length, axis=1)
                    aug_trn_input.append(after_padding)
                    aug_trn_target.append(self.trn_target[i, :])
            self.trn_input, self.trn_target = np.stack(aug_trn_input, axis=0), np.stack(aug_trn_target, axis=0)

    def __getitem__(self, index):
        if self.train:
            feature, target = self.trn_input[index], self.trn_target[index, 0]
        else:
            feature, target = self.val_input[index], self.val_target[index, 0]
        return feature, target

    def __len__(self):
        if self.train:
            return len(self.trn_input)
        return len(self.val_input)

    def visualize(self, index, feature_id):
        feature_list =  ['charge capacity', 'discharge capacity', 'chargetime', 'TAvg', 'TMin', 'TMax', 'EOL feature', 'chargetime feature']
        curve = self.trn_input[index, feature_id, :]
        plt.plot(np.arange(len(curve)), curve, c='red')
        plt.ylabel(feature_list[feature_id], fontsize=14)
        plt.xlabel('time', fontsize=14)
        plt.show()
        plt.close()


def load_Severson(training=True, norm=True, part='discharge'):
    folder_path = 'Severson_Dataset/feature_selector_discharge/'
    index = range(500) if part == 'charge' else range(500, 1000, 1)
    if training:
        feature, target = np.load(folder_path+'trn_features.npy')[:, :, index], np.load(folder_path+'trn_targets.npy')
    else:
        feature, target = np.load(folder_path+'val_features.npy')[:, :, index], np.load(folder_path+'val_targets.npy')
    if norm:
        return normalize(feature), normalize(target)
    else:
        return normalize(feature), target
 

def get_scaler(pred_target='EOL'):
    """get the normalize scaler in training set"""
    assert pred_target=='EOL' or pred_target=='chargetime' or pred_target=='both' 
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    feature =  np.load('Severson_Dataset/feature_selector_discharge/trn_features.npy')[:, :, 500:]
    target = np.load('Severson_Dataset/feature_selector_discharge/trn_targets.npy')
    scaler_x.fit(feature.transpose((0, 2, 1)).reshape(-1, 4))

    if pred_target=='EOL':
        target = target[:, 0].reshape(-1, 1)
    elif pred_target=='chargetime':
        target = target[:, 1].reshape(-1, 1)

    scaler_y.fit(target)
    return scaler_x, scaler_y


def normalize(data):
    scaler_x, scaler_y = get_scaler('both')
    c = data.shape[1] # channel
    if c>2: # feature
        data = scaler_x.transform(data.transpose((0, 2, 1)).reshape(-1, c)).reshape(-1, 500, c).transpose((0, 2, 1))
    else: # target
        data = scaler_y.transform(data.reshape(-1, c))
    return data

