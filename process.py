import torch
from torch.cuda.amp import autocast, GradScaler
import xgboost as xgb
from tqdm import tqdm
from utils import write_log

def train_epoch(model, loader, loss_func, optimizer, args):
    model.train()
    if args.amp: scaler = GradScaler()
    total_loss, num_correct, num_samples = 0., 0, 0
    for batch in loader:
        batch = batch.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_func(output, batch.y)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        num_correct += (output.argmax(dim=1) == batch.y).sum().item()
        num_samples += len(batch.y)
    torch.cuda.empty_cache()
    return total_loss / num_samples, num_correct / num_samples

@torch.no_grad()
def test_epoch(model, loader, args):
    model.eval()
    num_correct, num_samples = 0, 0
    for batch in loader:
        batch = batch.to(args.device)
        output = model(batch.x, batch.edge_index, batch.batch)
        num_correct += (output.argmax(dim=1) == batch.y).sum().item()
        num_samples += len(batch.y)
    torch.cuda.empty_cache()
    return num_correct / num_samples

def process(model, train_loader, test_loader, loss_func, optimizer, args):
    tqdm_meter = tqdm(desc='[Training GAT]')
    write_log(args.dirlog, '\n'+str(args.__dict__)+'\n')
    best_train_acc, best_test_acc = 0., 0.

    for epoch in range(args.nepoch):
        loss, train_acc = train_epoch(model, train_loader, loss_func, optimizer, args)
        test_acc = test_epoch(model, test_loader, args)
        tqdm_meter.set_postfix(
            Epoch='%3d' % (epoch + 1),
            Loss ='%6f' % loss,
            TrainAcc='%6.2f%%' % (train_acc * 100),
            TestAcc ='%6.2f%%' % (test_acc * 100))
        tqdm_meter.update()
        write_log(args.dirlog, 'Epoch %03d, Loss %.6f, TrainAcc %6.2f%%, TestAcc %6.2f%%\n' 
                    % (epoch + 1, loss, train_acc * 100, test_acc * 100))

        if test_acc > best_test_acc or (test_acc == best_test_acc and train_acc > best_train_acc):
            best_test_acc = test_acc
            best_train_acc = train_acc
            torch.save(model, args.dirmodel)
        torch.cuda.empty_cache()
    tqdm_meter.close()

