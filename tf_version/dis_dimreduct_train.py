import numpy as np
import argparse
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import discharge_model_tf
from data_preprocessing import load_Severson
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


def get_args_parser():
    parser = argparse.ArgumentParser('Discharge Model Feature Selector training', add_help=False)
    parser.add_argument('--input_shape', default=(500, 4), type=tuple)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--seed', default=97, type=int)

    # Model parameters
    parser.add_argument('--model_name', default='Dim_Reduction_1', type=str) 
    parser.add_argument('--pred_target', default='EOL', type=str) 
    parser.add_argument('--finetune', default=False, type=bool)   
    parser.add_argument('--load_checkpoint', default='Dim_Reduction_2_seed67.pth', type=str)                  

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    return parser


def main(args):
    model = discharge_model_tf.__dict__[args.model_name](args.input_shape)
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=args.lr, amsgrad=True)
    loss_function = keras.losses.MeanSquaredError()
    criterion = keras.losses.MeanAbsoluteError()
    checkpoint = ModelCheckpoint(filepath='checkpoints/'+args.model_name+'_ep{epoch:d}.h5', monitor='val_loss', save_best_only=False)
    model.compile(optimizer=opt, loss=loss_function, metrics=[criterion])
    x_train, y_train = load_Severson(training=True)
    x_test, y_test = load_Severson(training=False)
    f_id = 0 if args.pred_target=='EOL' else 1
    history = model.fit(x_train, y_train[:, f_id], epochs=args.epochs,
                        batch_size=args.batch_size, shuffle=True, callbacks=[checkpoint],
                        validation_data=(x_test, y_test[:, f_id]), verbose=2)

    plt.plot(history.history['loss'], label='train', c='blue', ls='--')
    plt.plot(history.history['val_loss'], label='test', c='red', ls='--')
    plt.title('model loss, fontsize=14')
    plt.ylabel('loss', fontsize=14)
    plt.xlabel('epoch', fontsize=14)
    plt.legend(loc='upper left')
    plt.savefig('save_fig/loss_profile.png')
    plt.close()
    result_plot(model, [x_train, x_test, y_train, y_test], f_id)
    


def result_plot(model, dataset, f_id):
    x_train, x_test, y_train, y_test = dataset
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    ax = plt.gca()
    ax.set_aspect(1)
    for i in range(len(pred_train)//100):
        s, e = i*100, (i+1)*100
        plt.scatter(y_train[s:e, f_id], pred_train[s:e, 0], c=range(100), cmap='coolwarm', s=4, alpha=0.7)
    plt.plot([np.min(y_train), np.max(y_train)], [np.min(y_train), np.max(y_train)], ls='--', c='black')
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('prediction', fontsize=14)
    plt.colorbar()
    plt.title('trnset result', fontsize=16)
    plt.savefig('save_fig/trn_real_result.png')
    plt.close()

    for i in range(len(pred_test)//100):
        s, e = i*100, (i+1)*100
        plt.scatter(y_test[s:e, f_id], pred_test[s:e, 0], c=range(100), cmap='coolwarm', s=4, alpha=0.7)
    plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], ls='--', c='black')
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('prediction', fontsize=14)
    plt.colorbar()
    plt.title('valset result', fontsize=16)
    plt.savefig('save_fig/val_real_result.png')
    plt.close()


if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args) 