# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="192.168.8.89" --master_port=12345 train-DDP.py --use_mix_precision True
# Watch Training Log：tensorboard --logdir=tensorboard_dir
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import time
import os
import torch.optim
import torch.utils.data
import torch.nn as nn
from collections import OrderedDict
from model import CustomMobileNetV3
from getdata import MyData
from torch.cuda.amp import GradScaler
from option import get_args
opt = get_args()
dist.init_process_group(backend='nccl', init_method='env://')

os.makedirs(opt.checkpoints, exist_ok=True)


def train(gpu):
    rank = dist.get_rank()
    model = CustomMobileNetV3()
    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().to(gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    scaler = GradScaler(enabled=opt.use_mix_precision)  

    dataloaders = MyData()
    train_loader = dataloaders['train']
    test_loader = dataloaders['val']
    
    if opt.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    start_time = time.time()
    best_val_acc = 0.0
    no_improve_epochs = 0
    early_stopping_patience = 10  # Early Stopping Patience
    
    """breakckpt resume"""
    if opt.resume:
        checkpoint = torch.load(opt.resume_ckpt)
        print('Loading checkpoint from:', opt.resume_ckpt)
        new_state_dict = OrderedDict()      # Create a new ordered dictionary and remove prefixes
        for k, v in checkpoint['model'].items():
            name = k[7:]                    # Remove 'module.' To match the original model definition
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)     # Load a new state dictionary
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']                       # Set the starting epoch
        if opt.use_lr_scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        
    for epoch in range(start_epoch + 1, opt.epochs):
        tqdm_trainloader = tqdm(train_loader, desc=f'Epoch {epoch}')
        running_loss, running_correct_top1, running_correct_top3, running_correct_top5 = 0.0, 0.0, 0.0, 0.0
        total_samples = 0
        for i, (images, target) in enumerate(tqdm_trainloader if rank == 0 else train_loader, 0):
            images = images.to(gpu)
            target = target.to(gpu)

            with torch.cuda.amp.autocast(enabled=opt.use_mix_precision):

                output = model(images)
                loss = criterion(output, target)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update() 

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(output.data, 1)
                running_correct_top1  += (predicted == target).sum().item()
                _, predicted_top3 = torch.topk(output.data, 3, dim=1)
                _, predicted_top5 = torch.topk(output.data, 5, dim=1)
                running_correct_top3 += (predicted_top3[:, :3] == target.unsqueeze(1).expand_as(predicted_top3)).sum().item()
                running_correct_top5 += (predicted_top5[:, :5] == target.unsqueeze(1).expand_as(predicted_top5)).sum().item()
                total_samples += target.size(0)
            
        state = {'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}
        
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0] if opt.use_lr_scheduler else opt.lr
            print(f'[Epoch {epoch}]  '
                    f'[Train Loss: {running_loss / len(train_loader.dataset):.6f}]  '
                    f'[Train Top-1 Acc: {running_correct_top1 / len(train_loader.dataset):.6f}]  '
                    f'[Train Top-3 Acc: {running_correct_top3 / len(train_loader.dataset):.6f}]  '
                    f'[Train Top-5 Acc: {running_correct_top5 / len(train_loader.dataset):.6f}]  '
                    f'[Learning Rate: {current_lr:.6f}]  '
                    f'[Time: {time.time() - start_time:.6f} seconds]')
            writer.add_scalar('Train/Loss', running_loss / len(train_loader.dataset), epoch)
            writer.add_scalar('Train/Top-1 Accuracy', running_correct_top1 / len(train_loader.dataset), epoch)
            writer.add_scalar('Train/Top-3 Accuracy', running_correct_top3 / len(train_loader.dataset), epoch)
            writer.add_scalar('Train/Top-5 Accuracy', running_correct_top5 / len(train_loader.dataset), epoch)
            writer.add_scalar('Train/Learning Rate', current_lr, epoch)
            
            torch.save(state, f'{opt.checkpoints}model_epoch_{epoch}.pth')
            # dist.barrier()
            
        tqdm_trainloader.close()
        
        if opt.use_lr_scheduler:    # Learning-rate Scheduler
            scheduler.step()
        
        acc_top1 = valid(test_loader, model, epoch, gpu, rank)
        if acc_top1 is not None:
            if acc_top1 > best_val_acc:
                best_val_acc = acc_top1
                no_improve_epochs = 0
                torch.save(state, f'{opt.checkpoints}/model_best.pth')
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stopping_patience:
                    print(f'Early stopping triggered after {early_stopping_patience} epochs without improvement.')
                    break
        else:
            print("Warning: acc_top1 is None, skipping this epoch.")
        
    dist.destroy_process_group()

def valid(val_loader, model, epoch, gpu, rank):
    model.eval()
    correct_top1, correct_top3, correct_top5, total = torch.tensor(0.).to(gpu), torch.tensor(0.).to(gpu), torch.tensor(0.).to(gpu), torch.tensor(0.).to(gpu)
    with torch.no_grad():
        tqdm_valloader = tqdm(val_loader, desc=f'Epoch {epoch}')
        for i, (images, target) in enumerate(tqdm_valloader, 0) :
            images = images.to(gpu)
            target = target.to(gpu)
            output = model(images)
            total += target.size(0)
            correct_top1  += (output.argmax(1) == target).type(torch.float).sum()
            _, predicted_top3 = torch.topk(output, 3, dim=1)
            _, predicted_top5 = torch.topk(output, 5, dim=1)
            correct_top3 += (predicted_top3[:, :3] == target.unsqueeze(1).expand_as(predicted_top3)).sum().item()
            correct_top5 += (predicted_top5[:, :5] == target.unsqueeze(1).expand_as(predicted_top5)).sum().item()
            
    dist.reduce(total, 0, op=dist.ReduceOp.SUM)     # Group communication reduce operation (change to allreduce if Gloo)
    dist.reduce(correct_top1, 0, op=dist.ReduceOp.SUM)
    dist.reduce(correct_top3, 0, op=dist.ReduceOp.SUM)
    dist.reduce(correct_top5, 0, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f'[Epoch {epoch}]  '
                f'[Val Top-1 Acc: {correct_top1 / total:.6f}]  '
                f'[Val Top-3 Acc: {correct_top3 / total:.6f}]  '
                f'[Val Top-5 Acc: {correct_top5 / total:.6f}]')
        writer.add_scalar('Validation/Top-1 Accuracy', correct_top1 / total, epoch)
        writer.add_scalar('Validation/Top-3 Accuracy', correct_top3 / total, epoch)
        writer.add_scalar('Validation/Top-5 Accuracy', correct_top5 / total, epoch)
    
    return float(correct_top1 / total)  # Return top 1 precision
    tqdm_valloader.close()


def main():
    train(opt.local_rank)


if __name__ == '__main__':
    writer = SummaryWriter(log_dir=opt.tensorboard_dir)
    main()
    writer.close()
