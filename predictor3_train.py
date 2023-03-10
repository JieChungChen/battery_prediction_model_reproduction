import matplotlib as mpl
mpl.use('Agg')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from random import randint
import argparse
import torch
from torch import nn, optim
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_preprocessing import Predictor3_Dataset
from full_model import Predictor_3
from utils import *


def get_args_parser():
    parser = argparse.ArgumentParser('Predictor3 training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--seed', default=39, type=int)

    # Model parameters
    parser.add_argument('--model_name', default='Predictor_3', type=str) 
    parser.add_argument('--finetune', default=True, type=bool)   
    parser.add_argument('--load_checkpoint', default='predictor3_best_model.pth', type=str)                  

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--lr_schedule', type=bool, default=False, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR')
    parser.add_argument('--warm_up', type=int, default=10, metavar='LR')
    parser.add_argument('--delta', type=int, default=1)
    return parser


def main(args):
    if torch.cuda.is_available():
        print(" -- GPU is available -- ")

    trn_set = Predictor3_Dataset(train=True, last_padding=False)
    trn_loader = DataLoader(trn_set, batch_size=92, num_workers=0, drop_last=False, shuffle=False)
    val_set = Predictor3_Dataset(train=False, last_padding=False)
    val_loader = DataLoader(val_set, batch_size=23, num_workers=0, drop_last=False, shuffle=False)
    print(len(trn_set), len(val_set))

    model = Predictor_3(10, 1, 0.25).apply(init_weights).cuda()
    if args.finetune:
        model.load_state_dict(torch.load(args.load_checkpoint))
    summary(model, (10, 100)) # architecture visualization

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.anneal_period, eta_min=args.min_lr)
    # criterion = nn.HuberLoss(delta=args.delta)
    criterion = nn.L1Loss()
    loss_function  = nn.MSELoss()

    best_rmse = 1000
    trn_loss_record, val_loss_record = [], []
    seq_len = [20, 40, 60, 80, 100]
    for epoch in range(args.epochs):
        trn_set_rand = Predictor3_Dataset(train=True, last_padding=True)
        trn_loader_rand = DataLoader(trn_set_rand, batch_size=args.batch_size, num_workers=0, drop_last=False, shuffle=True)
        model.train()
        batch = 0
        n_minibatch = (len(trn_set_rand)//args.batch_size)
        if args.lr_schedule:
            adjust_learning_rate(optimizer, args.epochs, epoch+1, args.warm_up, args.lr, args.min_lr)
        for inputs, targets in trn_loader_rand:
            batch += 1
            optimizer.zero_grad()
            output = model(inputs.cuda().float())
            loss = criterion(output, targets.reshape(-1, 1).cuda().float())
            loss.backward()
            optimizer.step()
            if batch%50==1:
                print('epoch:[%d / %d] batch:[%d / %d] loss= %.3f lr= %e' % 
                    (epoch + 1, args.epochs, batch, n_minibatch, loss.mean(), optimizer.param_groups[0]["lr"]))

        # model evaluation per epoch
        model.eval()
        with torch.no_grad():
            trn_loss, val_loss = 0, 0
            for inputs, targets in trn_loader:
                output = model(inputs.cuda().float())
                loss = loss_function(output , targets.reshape(-1, 1).cuda().float())
                trn_loss += loss.mean()
            for inputs, targets in val_loader:
                output = model(inputs.cuda().float())
                loss = loss_function(output , targets.reshape(-1, 1).cuda().float())
                val_loss += loss.mean()
            trn_loss_record.append(trn_loss.cpu())
            val_loss_record.append(val_loss.cpu())
            print('trn_loss: %.3f, val_loss: %.3f' % ((trn_loss), (val_loss)))

        # inverse transform to real EOL
        trn_rmse, test_rmse = predictor3_model_evaluation(model, best_rmse, eval_length=[0, 19, 99])
        print('training set RMSE 1 cycle: %.3f, 20 cycle: %.3f, 100 cycle: %.3f' %
                (trn_rmse[0], trn_rmse[1], trn_rmse[2]))
        print('testing set RMSE 1 cycle: %.3f, 20 cycle: %.3f, 100 cycle: %.3f' %
                (test_rmse[0], test_rmse[1], test_rmse[2]))

        # save best testing loss      
        if test_rmse[2]<best_rmse:
            best_rmse = test_rmse[2]
            if args.finetune:
                torch.save(model.state_dict(), 'predictor3_finetuned.pth')
            else:
                torch.save(model.state_dict(), 'predictor3_best_model.pth')

    # training finished
    loss_profile(trn_loss_record, val_loss_record)
    print('best RMSE: %.3f' % (best_rmse))


if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)