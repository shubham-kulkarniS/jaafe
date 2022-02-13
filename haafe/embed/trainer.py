
import math
import os
import time

import numpy as np

import torch
from torch import nn,optim
from sklearn.metrics import roc_auc_score

# 95% Confidence Interval for AUC. Hanley and McNeil (1982). https://gist.github.com/doraneko94/e24643136cfb8baf03ef8a314ab9615c
def roc_auc_score_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = math.sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return AUC, (lower, upper)



class Trainer:
    def __init__(self,dataloaders,model,args=None,**kwargs) -> None:
        self.model = model
        self.args = args        
        # define optimizer
        if args.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        criterion = nn.BCEWithLogitsLoss(reduction='mean') 
        # This loss combines a Sigmoid layer and the BCELoss in one single class.
        self.criterion = criterion
        self.optimiser = optimizer
        self.dataloaders = dataloaders
    
    def train_step(self,epoch,verbose=True):
        optimizer = self.optimiser
        model = self.model
        criterion = self.criterion
        loader = self.dataloaders['train']

        model.train()
        global_epoch_loss = 0
        samples = 0
        for batch_idx, (_, data, target) in enumerate(loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
            optimizer.step()
            global_epoch_loss += loss.data.item() * len(target)
            samples += len(target)
            if verbose and (batch_idx % self.args.log_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, samples, len(loader.dataset), 100*samples/len(loader.dataset), global_epoch_loss/samples))
        return global_epoch_loss / samples


        
    def test_step(self, loader, verbose=True, data_set='Test', save=None):
        # optimizer = self.optimiser
        model = self.model
        criterion = self.criterion

        model.eval()
        test_loss = 0
        tpred = []
        ttarget = []

        if save is not None:
            csv = open(save, 'wt')
            print('index,prob', file=csv)

        with torch.no_grad():
            for keys, data, target in loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                pred = output.sigmoid()
                tpred.append(pred.cpu().numpy())

                # if target[0] != -1:
                loss = criterion(output, target.float()).data.item()
                test_loss += loss * len(target) # sum up batch loss 
                ttarget.append(target.cpu().numpy())

                if save is not None:
                    for i, key in enumerate(keys):
                        print(f'{key},{pred[i]}', file=csv)
        
        if len(ttarget) > 0:
            test_loss /= len(loader.dataset)
            auc, auc_ci = roc_auc_score_ci(np.concatenate(ttarget), np.concatenate(tpred))
            if verbose:
                print('\n{} set: Average loss: {:.4f}, AUC: {:.1f}% ({:.1f}% - {:.1f}%)\n'.format(
                    data_set, test_loss, 100 * auc, auc_ci[0]*100, auc_ci[1]*100))

            return test_loss, auc

    def best_train(self):        
        ## saving   
        if not os.path.isdir(self.args.checkpoint):
            os.mkdir(self.args.checkpoint)

        best_valid_auc = 0
        iteration = 0
        epoch = 1
        best_epoch = epoch
        # trainint with early stopping
        t0 = time.time()
        while (epoch < self.args.epochs + 1) and (iteration < self.args.patience):
            
            self.train_step(epoch)
            # weight=train_dataset.weight)
            valid_loss, valid_auc = self.test_step(self.dataloaders['val'], data_set='Validation')


            if valid_auc > best_valid_auc:
                print('Saving state')
                iteration = 0
                best_valid_auc = valid_auc
                best_epoch = epoch
                state = {
                    'valid_auc': valid_auc,
                    'valid_loss': valid_loss,
                    'epoch': epoch,
                }
                
                torch.save(state, './{}/ckpt{:03d}.pt'.format(self.args.checkpoint, epoch))
                torch.save(self.model.state_dict(), './{}/model{:03d}.pt'.format(self.args.checkpoint, epoch))

            else:
                iteration += 1
                print('AUC was not improved, iteration {0}'.format(str(iteration)))

            epoch += 1
            print(f'Elapsed seconds: ({time.time() - t0:.0f}s)')
        print(f'Best AUC: {best_valid_auc*100:.1f}% on epoch {best_epoch}')

    def complete(self):
        pass












def train(loader, model, criterion, optimizer, epoch, cuda, log_interval, weight=None, max_norm=1, verbose=True):
    model.train()
    global_epoch_loss = 0
    samples = 0
    for batch_idx, (_, data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        global_epoch_loss += loss.data.item() * len(target)
        samples += len(target)
        if verbose and (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, samples, len(loader.dataset), 100*samples/len(loader.dataset), global_epoch_loss/samples))
    return global_epoch_loss / samples




def test(loader, model, criterion, cuda, verbose=True, data_set='Test', save=None):
    model.eval()
    test_loss = 0
    tpred = []
    ttarget = []

    if save is not None:
        csv = open(save, 'wt')
        print('index,prob', file=csv)

    with torch.no_grad():
        for keys, data, target in loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.sigmoid()
            tpred.append(pred.cpu().numpy())

            # if target[0] != -1:
            loss = criterion(output, target.float()).data.item()
            test_loss += loss * len(target) # sum up batch loss 
            ttarget.append(target.cpu().numpy())

            if save is not None:
                for i, key in enumerate(keys):
                    print(f'{key},{pred[i]}', file=csv)
    
    if len(ttarget) > 0:
        test_loss /= len(loader.dataset)
        auc, auc_ci = roc_auc_score_ci(np.concatenate(ttarget), np.concatenate(tpred))
        if verbose:
            print('\n{} set: Average loss: {:.4f}, AUC: {:.1f}% ({:.1f}% - {:.1f}%)\n'.format(
                data_set, test_loss, 100 * auc, auc_ci[0]*100, auc_ci[1]*100))

        return test_loss, auc