from tqdm import tqdm
import utils
import os
import random
import argparse
import numpy as np
import sys

from torch.utils import data
from datasets.CBASDataset import CBASDataset

from metrics import StreamSegMetrics
from network.HCM import HCM
import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter


def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--trainset_path", type=str, 
        default='../Datasets/CBAS/Train',
        help="path to Dataset")
    parser.add_argument("--testset_path", type=str, 
        default='../Datasets/CBAS/Test',
        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='CBAS', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help='num_classes')
    parser.add_argument("--model", type=str, default='HCM',
        help='model name:[HCM]')

    parser.add_argument("--epochs", type=int, default=60,
                        help="epoch number (default: 60)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
  
    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size ')
    parser.add_argument("--trainsize", type=int, default=512)
    parser.add_argument("--n_cpu", type=int, default=4,
                        help="download datasets")
    
    parser.add_argument("--ckpt", type=str,
            default=None, help="restore from checkpoint")
    parser.add_argument("--loss_type", type=str, default='cse', 
                        help="loss type:[cse, focal]")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--save_ep", type=int, default=5,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--save_path", type=str, 
                default='./CHKP/')
    parser.add_argument('--log_dir', type=str, default='./logs/',)
    return parser


def get_dataset(opts):
    train_dst = CBASDataset(is_train=True,voc_dir=opts.trainset_path)
    val_dst = CBASDataset(is_train=False,voc_dir=opts.testset_path)
    return train_dst, val_dst


def validate(opts, model, loader, device,  metrics):
    metrics.reset()
    with torch.no_grad():
        for step, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            S=model(images)
              
            outputs = S[0]
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

        score = metrics.get_results()
    return score


def main():
    
    opts = get_argparser().parse_args()
    os.makedirs(opts.save_path, exist_ok=True)
    opts.log_dir=opts.log_dir+'{}_{}_{}e'.format(opts.model, opts.dataset, opts.epochs)
    tb_writer = SummaryWriter(opts.log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.cuda.empty_cache()
    
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts)
    opts.total_itrs=opts.epochs * (len(train_dst) // opts.batch_size)
    print('opts:',opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu,
        drop_last=True)  
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))


    model = eval(opts.model)()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=opts.weight_decay)
    metrics=StreamSegMetrics(opts.num_classes)

 
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    if opts.loss_type == 'focal':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cse':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

        
    cur_epoch=0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)  
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory

    model=model.to(device)

   
    for epoch in range(cur_epoch, opts.epochs):
        model.train()
        cur_itrs=0
        data_loader = tqdm(train_loader, file=sys.stdout)
        running_loss = 0.0
        
        for (images, gts) in data_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            gts = gts.to(device, dtype=torch.long)

            gts=gts.squeeze()
            optimizer.zero_grad()
            # print('images.shape:', images.shape)
            
            s1,s2,s3,s4= model(images)
            
            loss1 = criterion(s1, gts) 
            loss2 = criterion(s2, gts) 
            loss3 = criterion(s3, gts)
            loss4 = criterion(s4, gts) 
    
            total_loss = loss1 + loss2/2 + loss3/4 +loss4/8 
            
            running_loss += total_loss.data.item()

            total_loss.backward()
            optimizer.step()

            data_loader.desc = "Epoch {}/{}, loss={:.4f}".format(epoch, opts.epochs, running_loss/cur_itrs)
            
            scheduler.step()


        print("validation...")
        model.eval()
        val_score = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
        print('val_score:',val_score)
        tags = ["train_loss", "learning_rate","Mean_Acc","Mean_IoU","BA_IoU"]

        tb_writer.add_scalar(tags[0], (running_loss/cur_itrs), epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[2], val_score['Mean Acc'], epoch)
        tb_writer.add_scalar(tags[3], val_score['Mean IoU'], epoch)
        tb_writer.add_scalar(tags[4], val_score['Class IoU'][1], epoch)

     
        if (epoch+1) % opts.save_ep == 0:
            save_name=opts.save_path+'lastest_{}_{}.pth'.format(opts.model, opts.dataset)
            torch.save(model.state_dict(), save_name)
            if (epoch+1) >= 40 and (epoch+1)%10 ==0:
                save_name=opts.save_path+'{}_{}_ep{}.pth'.format(opts.model, opts.dataset, (epoch+1))
                torch.save(model.state_dict(), save_name)
           
if __name__ == '__main__':
    main()
