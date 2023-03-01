import argparse
import matplotlib as mpl
mpl.use('Agg')
import math
from random import randint
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary
from torch.utils.data import DataLoader
from data_preprocessing import Severson_Dataset_Training
from discharge_model import Dim_Reduction_1, Dim_Reduction_2, init_weights
from SeversonDataset_preprocess import train_val_split
from utils import *


def get_args_parser():
    parser = argparse.ArgumentParser('Feature Selector 1 & 2 training', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--seed', default=39, type=int)
    parser.add_argument('--detail_step', default=10, type=int)

    # Model parameters
    parser.add_argument('--model_name', default='Feature_Selector2', type=str) 
    parser.add_argument('--pred_target', default='chargetime', type=str) 
    parser.add_argument('--finetune', default=False, type=bool)   
    parser.add_argument('--load_checkpoint', default='Feature_Selector2_best_seed39.pth', type=str)                  

    # Hyperparameters
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR')
    parser.add_argument('--lr_schedule', type=bool, default=True, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=5e-5, metavar='LR')
    parser.add_argument('--warm_up', type=int, default=10, metavar='LR')
    parser.add_argument('--delta', type=int, default=1)

    return parser

 
def main(args):
    if torch.cuda.is_available():
        print(" -- GPU is available -- ")

    # 根據random seed，隨機分割訓練及測試集
    train_val_split(seed=args.seed)

    # pytorch_dataset_preprocessing(seed=args.seed, folder='Severson_Dataset/')
    trn_set = Severson_Dataset_Training(train=True, pred_target=args.pred_target)
    val_set = Severson_Dataset_Training(train=False, pred_target=args.pred_target)
    trn_loader = DataLoader(trn_set, batch_size=args.batch_size, num_workers=0, drop_last=False, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0, drop_last=False, shuffle=False)

    if args.pred_target == 'EOL': # for EOL
        model = Dim_Reduction_1(4, 1).apply(init_weights).cuda()
        summary(model, (4, 500))
    else: # for chargetime
        model = Dim_Reduction_2(4, 1).apply(init_weights).cuda()
        summary(model, (4, 500))

    if args.finetune:
        model = torch.load(args.load_checkpoint).cuda()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.anneal_period, eta_min=args.min_lr)
    criterion = nn.MSELoss()
    # criterion = nn.HuberLoss(delta=args.delta)

    best_rmse = 1000
    trn_loss_record, val_loss_record = [], []
    for epoch in range(args.epochs):
        model.train()
        step = 0
        n_minibatch = math.ceil(len(trn_set)/args.batch_size)
        if args.lr_schedule:
            adjust_learning_rate(optimizer, args.epochs, epoch+1, args.warm_up, args.lr, args.min_lr)
        for inputs, targets in trn_loader:
            step += 1
            optimizer.zero_grad()
            outputs = model(inputs.cuda().float())
            loss = criterion(outputs, targets.reshape(-1, 1).cuda().float())
            loss.backward()
            optimizer.step()
            if step%args.detail_step==0:
                print('epoch:[%d / %d] batch:[%d / %d] loss: %.3f lr: %e' % (epoch+1, args.epochs, step, n_minibatch, loss, optimizer.param_groups[0]["lr"]))

        # model evaluation per epoch
        model.eval()
        with torch.no_grad():
            trn_loss, trn_batch = 0, 0
            for inputs, targets in trn_loader:
                output = model(inputs.cuda().float())
                loss = criterion(output, targets.reshape(-1, 1).cuda().float())
                trn_loss += loss
                trn_batch += 1
            val_loss, val_batch = 0, 0
            for inputs, targets in val_loader:
                output = model(inputs.cuda().float())
                loss = criterion(output, targets.reshape(-1, 1).cuda().float())
                val_loss += loss
                val_batch += 1
            trn_loss_record.append((trn_loss/trn_batch).cpu())
            val_loss_record.append((val_loss/val_batch).cpu())
            print('trn_loss: %.3f, val_loss: %.3f' % ((trn_loss/trn_batch), (val_loss/val_batch)))

        # inverse transform to real RUL
        rmse, mape = real_RMSE_and_MAPE(model, val_loader, args.pred_target)
        print('real RMSE: %.3f, MAPE: %.3f' % (rmse, mape))
        if rmse < best_rmse:
            best_rmse, best_mape = rmse, mape
            pred_result(model, trn_set, 'trn', args.pred_target)        
            pred_result(model, val_set, 'val', args.pred_target)
            torch.save(model, args.model_name+'_best_seed'+str(args.seed)+'.pth')

    # training finished
    loss_profile(trn_loss_record, val_loss_record)
    print('best RMSE: %.3f, MAPE: %.3f' % (best_rmse, best_mape))


if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args) 