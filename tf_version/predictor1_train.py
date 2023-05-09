import numpy as np
import argparse
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from discharge_model_tf import Predictor_1, mish
from data_preprocessing import get_scaler, predictor1_preprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


def get_args_parser():
    parser = argparse.ArgumentParser('Discharge Model Feature Selector training', add_help=False)
    parser.add_argument('--input_shape', default=(100, 8), type=tuple)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    # Model parameters
    parser.add_argument('--model_name', default='Predictor_1', type=str) 
    parser.add_argument('--finetune', default=False, type=bool)   
    parser.add_argument('--last_padding', default=True, type=bool)   
    parser.add_argument('--load_checkpoint', default='Dim_Reduction_2_seed67.pth', type=str)                  

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    return parser


def main(args):
    model = Predictor_1(args.input_shape, 0.5)
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=args.lr, amsgrad=True)
    loss_function = keras.losses.MeanSquaredError()
    criterion = keras.losses.MeanAbsoluteError()
    checkpoint = ModelCheckpoint(filepath='Predictor1.h5', monitor='val_loss', save_best_only=True)
    model.compile(optimizer=opt, loss=loss_function, metrics=[criterion])
    x_train, x_test, y_train, y_test = predictor1_preprocessing(ep=[17, 14]) #18
    if args.last_padding:
        aug_trn_input, aug_trn_target = [], []
        for i in range(len(x_train)):
            for cycle_length in range(100):
                after_padding = x_train[i].copy()
                after_padding[cycle_length:, :] = after_padding[cycle_length, :].reshape(1, -1).repeat(100-cycle_length, axis=0)
                aug_trn_input.append(after_padding)
                aug_trn_target.append(y_train[i, :])
        trn_input, trn_target = np.stack(aug_trn_input, axis=0), np.stack(aug_trn_target, axis=0)

    history = model.fit(trn_input, trn_target, epochs=args.epochs,
                        batch_size=args.batch_size, shuffle=True, callbacks=[checkpoint],
                        validation_data=(x_test, y_test), verbose=2)
    
    plt.plot(history.history['loss'], label='train', c='blue', ls='--')
    plt.plot(history.history['val_loss'], label='test', c='red', ls='--')
    plt.title('model loss, fontsize=14')
    plt.ylabel('loss', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.legend(loc='upper right')
    plt.savefig('save_fig/loss_profile.png')
    plt.close()

    model = keras.models.load_model('Predictor1.h5', custom_objects={'mish': mish}, compile=False)
    trn_rmse, test_rmse = predictor1_model_evaluation(model, [x_train, x_test, y_train, y_test])
    print('training set RMSE 1 cycle: %d, 5 cycle: %d, 100 cycle: %d' %
                (trn_rmse[0], trn_rmse[1], trn_rmse[2]))
    print('testing set RMSE 1 cycle: %d, 5 cycle: %d, 100 cycle: %d' %
                (test_rmse[0], test_rmse[1], test_rmse[2]))
    

def predictor1_model_evaluation(model, dataset, eval_length=[1, 5, 100]):
    """
    根據不同input length評估Predictor1之預測誤差
    """
    scaler_y = get_scaler('both')[1]
    x_train, x_test, y_train, y_test = dataset
    trn_rmse, test_rmse = [], []
    for cycles in eval_length:
        cycles-=1
        inputs = x_train.copy()
        for i in range(len(x_train)):
            inputs[i, cycles:, :] = inputs[i, cycles, :].reshape(1, -1).repeat(100-cycles, axis=0)
        pred = model.predict(inputs)
        gt, pred = scaler_y.inverse_transform(y_train), scaler_y.inverse_transform(pred)
        gt, pred = np.power(2, gt), np.power(2, pred)
        trn_rmse.append(root_mean_square_err(gt[:, 0], pred[:, 0]))

        ax = plt.gca()
        ax.set_aspect(1)
        plt.xlim(200, 2000)
        plt.ylim(200, 2000)
        plt.plot([0, 2000], [0, 2000], ls='--', c='black')
        plt.scatter(gt[:, 0], pred[:, 0], c='red', s=6, label='training')

        inputs = x_test.copy()
        for i in range(len(inputs)):
            inputs[i, cycles:, :] = inputs[i, cycles, :].reshape(1, -1).repeat(100-cycles, axis=0)
        pred = model.predict(inputs)
        gt, pred = scaler_y.inverse_transform(y_test), scaler_y.inverse_transform(pred)
        gt, pred = np.power(2, gt), np.power(2, pred)
        test_rmse.append(root_mean_square_err(gt[:, 0], pred[:, 0]))

        plt.scatter(gt[:, 0], pred[:, 0], c='blue', s=6, label='testing')
        plt.legend()
        plt.xlabel('ground truth', fontsize=14)
        plt.ylabel('prediction', fontsize=14)
        plt.savefig('save_fig/'+str(cycles+1)+'-cycle prediction.png')
        plt.close()     
    return trn_rmse, test_rmse


def root_mean_square_err(gt, pred):
    return np.sqrt(np.mean((gt-pred)**2))


if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args) 

