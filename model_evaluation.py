import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_preprocessing import get_scaler, Predictor1_Dataset


def main():
    model_path = 'predictor1_finetuned.pth'
    predictor1 = torch.load(model_path)
    predictor1.eval()
    _, scaler_y = get_scaler()
    trn_rmse, test_rmse = [], []
    trn_set = Predictor1_Dataset(train=True, last_padding=False)
    test_set = Predictor1_Dataset(train=False, last_padding=False)
    visulasized_id = [0, 50, 99]
    for cycles in range(100):
        trn_loader = DataLoader(trn_set, batch_size=92, num_workers=0, drop_last=False, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=23, num_workers=0, drop_last=False, shuffle=False)
        with torch.no_grad():
            for inputs, targets in test_loader:
                for i in range(len(inputs)):
                    inputs[i, :, cycles:] = inputs[i, :, cycles].reshape(-1, 1).repeat(1, 100-cycles)
                outputs = predictor1(inputs.cuda().float())
                pred = outputs.detach().cpu().numpy()
                gt, pred = scaler_y.inverse_transform(targets), scaler_y.inverse_transform(pred)
                rmse = np.sqrt(np.mean((gt[:, 0]-pred[:, 0])**2))
                mape = np.mean(np.abs(pred[:, 0]-gt[:, 0])/gt[:, 0])
                test_rmse.append(rmse)

            if cycles in visulasized_id:
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.plot([0, 2000], [0, 2000], ls='--', c='black', alpha=0.5)
                plt.scatter(gt[:, 0], pred[:, 0], c='red', s=7, label='testing', alpha=0.7)
                plt.subplot(1, 3, 3)
                plt.hist(np.sqrt((gt[:, 0]-pred[:, 0])**2), color='red', alpha=0.5)
                plt.xlabel('MSE')

            for inputs, targets in trn_loader:
                for i in range(len(inputs)):
                    inputs[i, :, cycles:] = inputs[i, :, cycles].reshape(-1, 1).repeat(1, 100-cycles)
                outputs = predictor1(inputs.cuda().float())
                pred = outputs.detach().cpu().numpy()
                gt, pred = scaler_y.inverse_transform(targets), scaler_y.inverse_transform(pred)
                rmse = np.sqrt(np.mean((gt[:, 0]-pred[:, 0])**2))
                mape = np.mean(np.abs(pred[:, 0]-gt[:, 0])/gt[:, 0])
                trn_rmse.append(rmse)

            if cycles in visulasized_id:
                plt.subplot(1, 3, 1)
                plt.scatter(gt[:, 0], pred[:, 0], c='blue', s=7, label='training', alpha=0.7)
                plt.legend()
                plt.xlabel('ground truth', fontsize=14)
                plt.ylabel('prediction', fontsize=14)
                plt.subplot(1, 3, 2)
                plt.hist(np.sqrt((gt[:, 0]-pred[:, 0])**2), color='blue', alpha=0.5)
                plt.xlabel('MSE')
                plt.show()
                plt.close()

    plt.plot(range(100), trn_rmse, c='blue', label='training')
    plt.plot(range(100), test_rmse, c='red', label='testing')
    plt.xlabel('input cycle(s)', fontsize=14)
    plt.ylabel('RMSE(EOL)', fontsize=14)
    plt.legend()
    plt.show()
    plt.close()


if __name__=='__main__':
    main()