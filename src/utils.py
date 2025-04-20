import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
import glob
from datetime import datetime


"""
Define task metrics, loss functions and model trainer here.
"""

def save_checkpoint(model, optimizer, scheduler, epoch, directory='models', run_id=None, model_name="model"):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f'{model_name}_checkpoint_epoch_{epoch:03d}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'run_id': run_id,
    }, filename)

def load_checkpoint(model, optimizer, scheduler, directory='models', specific_file=None, model_name="model"):
    if specific_file:
        path = specific_file
    else:
        # Trova l'ultimo file salvato
        files = glob.glob(os.path.join(directory, f'{model_name}_checkpoint_epoch_*.pth'))
        if not files:
            print("Nessun checkpoint trovato nella directory.")
            return 0, None
        path = max(files, key=os.path.getmtime)  # più recente

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    run_id = checkpoint['run_id'] if 'run_id' in checkpoint else None
    print(f"Checkpoint caricato da {path}")
    return checkpoint['epoch'], run_id

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_fit(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss

# Legacy: compute mIoU and Acc. for each image and average across all images.

# def compute_miou(x_pred, x_output):
#     _, x_pred_label = torch.max(x_pred, dim=1)
#     x_output_label = x_output
#     batch_size = x_pred.size(0)
#     class_nb = x_pred.size(1)
#     device = x_pred.device
#     for i in range(batch_size):
#         true_class = 0
#         first_switch = True
#         invalid_mask = (x_output[i] >= 0).float()
#         for j in range(class_nb):
#             pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).long().to(device))
#             true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).long().to(device))
#             mask_comb = pred_mask.float() + true_mask.float()
#             union = torch.sum((mask_comb > 0).float() * invalid_mask)  # remove non-defined pixel predictions
#             intsec = torch.sum((mask_comb > 1).float())
#             if union == 0:
#                 continue
#             if first_switch:
#                 class_prob = intsec / union
#                 first_switch = False
#             else:
#                 class_prob = intsec / union + class_prob
#             true_class += 1
#         if i == 0:
#             batch_avg = class_prob / true_class
#         else:
#             batch_avg = class_prob / true_class + batch_avg
#     return batch_avg / batch_size
#
#
# def compute_iou(x_pred, x_output):
#     _, x_pred_label = torch.max(x_pred, dim=1)
#     x_output_label = x_output
#     batch_size = x_pred.size(0)
#     for i in range(batch_size):
#         if i == 0:
#             pixel_acc = torch.div(
#                 torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
#                 torch.sum((x_output_label[i] >= 0).float()))
#         else:
#             pixel_acc = pixel_acc + torch.div(
#                 torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
#                 torch.sum((x_output_label[i] >= 0).float()))
#     return pixel_acc / batch_size


# New mIoU and Acc. formula: accumulate every pixel and average across all pixels in all images
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item(), acc.item()


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
           (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)


"""
=========== Universal Multi-task Trainer =========== 
"""


def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, opt, total_epoch=200, resume=False, log_dir='runs', use_wandb=True):
    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = 0
    if resume:
        start_epoch, run_id = load_checkpoint(multi_task_model, optimizer, scheduler, directory='models')
        if use_wandb:
            if run_id:
                wandb.init(project="multi-task", config=opt.__dict__, dir=log_dir, id=run_id, resume='must')
            else:
                wandb.init(project="multi-task", config=opt.__dict__, dir=log_dir)
    elif use_wandb:
        wandb.init(project="multi-task", config=opt.__dict__, dir=log_dir)
    run_id = wandb.run.id if use_wandb else None

    train_batch = len(train_loader)
    test_batch = len(test_loader)
    T = opt.temp
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    lambda_weight = np.ones([3, total_epoch])
    for index in trange(start_epoch, total_epoch, desc="Training Epochs"):
        cost = np.zeros(24, dtype=np.float32)

        # apply Dynamic Weight Average
        if opt.weight == 'dwa':
            if index == 0 or index == 1:
                lambda_weight[:, index] = 1.0
            else:
                w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
                w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
                w_3 = avg_cost[index - 1, 6] / avg_cost[index - 2, 6]
                lambda_weight[0, index] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                lambda_weight[1, index] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                lambda_weight[2, index] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

        # iteration for all batches
        multi_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        for k in tqdm(range(train_batch), desc=f"Epoch {index+1}", leave=False):
            train_data, train_label, train_depth, train_normal = next(train_dataset) #train_dataset.next()
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred, logsigma = multi_task_model(train_data)

            optimizer.zero_grad()
            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                          model_fit(train_pred[1], train_depth, 'depth'),
                          model_fit(train_pred[2], train_normal, 'normal')]

            if opt.weight == 'equal' or opt.weight == 'dwa':
                loss = sum([lambda_weight[i, index] * train_loss[i] for i in range(3)])
            else:
                loss = sum(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(3))

            loss.backward()
            optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
            avg_cost[index, :12] += cost[:12] / train_batch

        # compute mIoU and acc
        avg_cost[index, 1:3] = np.array(conf_mat.get_metrics())

        # evaluating test data
        multi_task_model.eval()
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = next(test_dataset)
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred, _ = multi_task_model(test_data)
                test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                             model_fit(test_pred[1], test_depth, 'depth'),
                             model_fit(test_pred[2], test_normal, 'normal')]

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred[2], test_normal)
                avg_cost[index, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 13:15] = np.array(conf_mat.get_metrics())

        scheduler.step()
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
            .format(index+1, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                    avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                    avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))
        writer.add_scalars('Loss/train', {'semantic': avg_cost[index, 0],
                                          'depth': avg_cost[index, 3],
                                          'normal': avg_cost[index, 6]}, index)
        writer.add_scalars('Loss/test', {'semantic': avg_cost[index, 12],
                                         'depth': avg_cost[index, 15],
                                         'normal': avg_cost[index, 18]}, index)
        if use_wandb:
            wandb.log({
                'epoch': index+1,
                'train/semantic_loss': avg_cost[index, 0],
                'train/semantic_miou': avg_cost[index, 1],
                'train/semantic_acc': avg_cost[index, 2],
                'train/depth_loss': avg_cost[index, 3],
                'train/depth_abs_err': avg_cost[index, 4],
                'train/depth_rel_err': avg_cost[index, 5],
                'train/normal_loss': avg_cost[index, 6],
                'train/normal_mean_err': avg_cost[index, 7],
                'train/normal_median_err': avg_cost[index, 8],
                'train/normal_mean_11.25': avg_cost[index, 9],
                'train/normal_mean_22.5': avg_cost[index, 10],
                'train/normal_mean_30': avg_cost[index, 11],
                'test/semantic_loss': avg_cost[index, 12],
                'test/semantic_miou': avg_cost[index, 13],
                'test/semantic_acc': avg_cost[index, 14],
                'test/depth_loss': avg_cost[index, 15],
                'test/depth_abs_err': avg_cost[index, 16],
                'test/depth_rel_err': avg_cost[index, 17],
                'test/normal_loss': avg_cost[index, 18],
                'test/normal_mean_err': avg_cost[index, 19],
                'test/normal_median_err': avg_cost[index, 20],
                'test/normal_mean_11.25': avg_cost[index, 21],
                'test/normal_mean_22.5': avg_cost[index, 22],
                'test/normal_mean_30': avg_cost[index, 23]
            })
        save_checkpoint(multi_task_model, optimizer, scheduler, index + 1, directory='models', run_id=run_id, model_name=f"SegNetMTAN_multitask_{opt.weight}")
    # saving the results of the last epoch in a csv file, the name of the file will be different for each run
    os.makedirs('results', exist_ok=True)
    opt.task = "multi_task"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(":", "-").replace(" ", "_").replace("-", "_").replace(".", "_")

    filename = os.path.join('results', f'{opt.task}_lambda_{opt.weight}_T_{opt.temp}_{current_time}.csv')
    with open(filename, 'w') as f:
        # saving every field in the first row
        f.write('epoch,train_semantic_loss,train_semantic_miou,train_semantic_acc,'
                'train_depth_loss,train_depth_abs_err,train_depth_rel_err,'
                'train_normal_loss,train_normal_mean_err,train_normal_median_err,'
                'train_normal_mean_11.25,train_normal_mean_22.5,train_normal_mean_30,'
                'test_semantic_loss,test_semantic_miou,test_semantic_acc,'
                'test_depth_loss,test_depth_abs_err,test_depth_rel_err,'
                'test_normal_loss,test_normal_mean_err,test_normal_median_err,'
                'test_normal_mean_11.25,test_normal_mean_22.5,test_normal_mean_30\n')
        for i in range(total_epoch):
            f.write(f'{i+1},{avg_cost[i, 0]},{avg_cost[i, 1]},{avg_cost[i, 2]},'
                    f'{avg_cost[i, 3]},{avg_cost[i, 4]},{avg_cost[i, 5]},'
                    f'{avg_cost[i, 6]},{avg_cost[i, 7]},{avg_cost[i, 8]},'
                    f'{avg_cost[i, 9]},{avg_cost[i, 10]},{avg_cost[i, 11]},'
                    f'{avg_cost[i, 12]},{avg_cost[i, 13]},{avg_cost[i, 14]},'
                    f'{avg_cost[i, 15]},{avg_cost[i, 16]},{avg_cost[i, 17]},'
                    f'{avg_cost[i, 18]},{avg_cost[i, 19]},{avg_cost[i, 20]},'
                    f'{avg_cost[i, 21]},{avg_cost[i, 22]},{avg_cost[i, 23]}\n')
    print(f"Results saved in {filename}")




"""
=========== Universal Single-task Trainer =========== 
"""


def single_task_trainer(train_loader, test_loader, single_task_model, device, optimizer, scheduler, opt, total_epoch=200, resume=False, log_dir='runs', use_wandb=True):
    writer = SummaryWriter(log_dir=log_dir)
    
    start_epoch = 0
    if resume:
        start_epoch, run_id = load_checkpoint(single_task_model, optimizer, scheduler, directory='models')
        if use_wandb:
            if run_id:
                wandb.init(project=f"single-task-{opt.task}", config=opt.__dict__, dir=log_dir, id=run_id, resume='must')
            else:
                wandb.init(project=f"single-task-{opt.task}", config=opt.__dict__, dir=log_dir)
    elif use_wandb:
        wandb.init(project=f"single-task-{opt.task}", config=opt.__dict__, dir=log_dir)
    run_id = wandb.run.id if use_wandb else None

    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    for index in trange(start_epoch, total_epoch, desc="Training"):
        cost = np.zeros(24, dtype=np.float32)

        # iteration for all batches
        single_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(single_task_model.class_nb)
        for k in tqdm(range(train_batch), desc=f"Epoch {index+1}", leave=False):
            train_data, train_label, train_depth, train_normal = next(train_dataset)
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_preds, _ = single_task_model(train_data)
            optimizer.zero_grad()

            if opt.task == 'semantic':
                train_pred = train_preds[0]
                train_loss = model_fit(train_pred, train_label, opt.task)
                train_loss.backward()
                optimizer.step()

                conf_mat.update(train_pred.argmax(1).flatten(), train_label.flatten())
                cost[0] = train_loss.item()

            if opt.task == 'depth':
                train_pred = train_preds[1]
                train_loss = model_fit(train_pred, train_depth, opt.task)
                train_loss.backward()
                optimizer.step()
                cost[3] = train_loss.item()
                cost[4], cost[5] = depth_error(train_pred, train_depth)

            if opt.task == 'normal':
                train_pred = train_preds[2]
                train_loss = model_fit(train_pred, train_normal, opt.task)
                train_loss.backward()
                optimizer.step()
                cost[6] = train_loss.item()
                cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred, train_normal)

            avg_cost[index, :12] += cost[:12] / train_batch

        if opt.task == 'semantic':
            avg_cost[index, 1:3] = np.array(conf_mat.get_metrics())

        # evaluating test data
        single_task_model.eval()
        conf_mat = ConfMatrix(single_task_model.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = next(test_dataset)
                test_data, test_label = test_data.to(device),  test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred, _ = single_task_model(test_data)

                if opt.task == 'semantic':
                    test_pred = test_pred[0]
                    test_loss = model_fit(test_pred, test_label, opt.task)

                    conf_mat.update(test_pred.argmax(1).flatten(), test_label.flatten())
                    cost[12] = test_loss.item()

                if opt.task == 'depth':
                    test_pred = test_pred[1]
                    test_loss = model_fit(test_pred, test_depth, opt.task)
                    cost[15] = test_loss.item()
                    cost[16], cost[17] = depth_error(test_pred, test_depth)

                if opt.task == 'normal':
                    test_pred = test_pred[2]
                    test_loss = model_fit(test_pred, test_normal, opt.task)
                    cost[18] = test_loss.item()
                    cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred, test_normal)

                avg_cost[index, 12:] += cost[12:] / test_batch
            if opt.task == 'semantic':
                avg_cost[index, 13:15] = np.array(conf_mat.get_metrics())

        scheduler.step()
        if opt.task == 'semantic':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f}'
              .format(index+1, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 12], avg_cost[index, 13], avg_cost[index, 14]))
        if opt.task == 'depth':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f}'
              .format(index+1, avg_cost[index, 3], avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17]))
        if opt.task == 'normal':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
              .format(index+1, avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11],
                      avg_cost[index, 18], avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))
            
        if opt.task == 'semantic':
            writer.add_scalars('Loss', {
                'train': avg_cost[index, 0],
                'test': avg_cost[index, 12]
            }, index)
            if use_wandb:
                wandb.log({
                    'epoch': index+1,
                    'train_loss': avg_cost[index, 0],
                    'train_miou': avg_cost[index, 1],
                    'train_acc': avg_cost[index, 2],
                    'test_loss': avg_cost[index, 12],
                    'test_miou': avg_cost[index, 13],
                    'test_acc': avg_cost[index, 14]
                })
        elif opt.task == 'depth':
            writer.add_scalars('Loss', {
                'train': avg_cost[index, 3],
                'test': avg_cost[index, 15]
            }, index)
            if use_wandb:
                wandb.log({
                    'epoch': index+1,
                    'train_loss': avg_cost[index, 3],
                    'train_abs_err': avg_cost[index, 4],
                    'train_rel_err': avg_cost[index, 5],
                    'test_loss': avg_cost[index, 15],
                    'test_abs_err': avg_cost[index, 16],
                    'test_rel_err': avg_cost[index, 17]
                })
        elif opt.task == 'normal':
            writer.add_scalars('Loss', {
                'train': avg_cost[index, 6],
                'test': avg_cost[index, 18]
            }, index)
            if use_wandb:
                wandb.log({
                    'epoch': index+1,
                    'train_loss': avg_cost[index, 6],
                    'train_mean_err': avg_cost[index, 7],
                    'train_median_err': avg_cost[index, 8],
                    'train_mean_11.25': avg_cost[index, 9],
                    'train_mean_22.5': avg_cost[index, 10],
                    'train_mean_30': avg_cost[index, 11],
                    'test_loss': avg_cost[index, 18],
                    'test_mean_err': avg_cost[index, 19],
                    'test_median_err': avg_cost[index, 20],
                    'test_mean_11.25': avg_cost[index, 21],
                    'test_mean_22.5': avg_cost[index, 22],
                    'test_mean_30': avg_cost[index, 23]
                })

        save_checkpoint(single_task_model, optimizer, scheduler, index + 1, directory='models', run_id=run_id, model_name=opt.task)
    # saving the results of the last epoch in a csv file, the name of the file will be different for each run
    os.makedirs('results', exist_ok=True)
    opt.task = "single_task"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(":", "-").replace(" ", "_").replace("-", "_").replace(".", "_")
    filename = os.path.join('results', f'{opt.task}_lambda_{opt.weight}_T_{opt.temp}_{current_time}.csv')
    with open(filename, 'w') as f:
        if opt.task == 'semantic':
            f.write('epoch,train_semantic_loss,train_semantic_miou,train_semantic_acc,'
                    'test_semantic_loss,test_semantic_miou,test_semantic_acc\n')
        elif opt.task == 'depth':
            f.write('epoch,train_depth_loss,train_abs_err,train_rel_err,'
                    'test_depth_loss,test_abs_err,test_rel_err\n')
        elif opt.task == 'normal':
            f.write('epoch,train_normal_loss,train_mean_err,train_median_err,'
                    'train_mean_11.25,train_mean_22.5,train_mean_30,'
                    'test_normal_loss,test_mean_err,test_median_err,'
                    'test_mean_11.25,test_mean_22.5,test_mean_30\n')
        for i in range(total_epoch):
            if opt.task == 'semantic':
                f.write(f'{i+1},{avg_cost[i, 0]},{avg_cost[i, 1]},{avg_cost[i, 2]},'
                        f'{avg_cost[i, 12]},{avg_cost[i, 13]},{avg_cost[i, 14]}\n')
            elif opt.task == 'depth':
                f.write(f'{i+1},{avg_cost[i, 3]},{avg_cost[i, 4]},{avg_cost[i, 5]},'
                        f'{avg_cost[i, 15]},{avg_cost[i, 16]},{avg_cost[i, 17]}\n')
            elif opt.task == 'normal':
                f.write(f'{i+1},{avg_cost[i, 6]},{avg_cost[i, 7]},{avg_cost[i, 8]},'
                        f'{avg_cost[i, 9]},{avg_cost[i, 10]},{avg_cost[i, 11]},'
                        f'{avg_cost[i, 18]},{avg_cost[i, 19]},{avg_cost[i, 20]},'
                        f'{avg_cost[i, 21]},{avg_cost[i, 22]},{avg_cost[i, 23]}\n')
    print(f"Results saved in {filename}")