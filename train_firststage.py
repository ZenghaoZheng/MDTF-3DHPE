import argparse
import os
import sys
import numpy as np
import pkg_resources
import torch
import wandb
from torch import optim
from tqdm import tqdm

from data.reader.motion_dataset import MotionDataset3D
from utils.data import flip_data
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from utils.learning import load_anglemodel, AverageMeter, decay_lr_exponentially
from utils.tools import count_param_numbers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.logging import Logger

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/h36m/first.yaml",
                        help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint/stage1/',
                        help='new checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=16, type=int, help='Number of CPU cores')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--use-log', action='store_true')
    parser.add_argument('--wandb-name', default=None, type=str)
    parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    opts = parser.parse_args()
    return opts


def classify_angles_15(angle_sequence, anglerange):
    lastindex = anglerange // 30
    shifted_angles = (angle_sequence - 15) % anglerange
    categories = (shifted_angles // 30) + 1
    categories[(angle_sequence >= 0) & (angle_sequence < 15)] = 0
    categories[(shifted_angles >= lastindex * 30)] = lastindex + 1
    return categories


def get_angles(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 17)
    '''
    limbs_id = [[0, 1], [1, 2], [2, 3],
                [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [9, 10],
                [8, 11], [11, 12], [12, 13],
                [8, 14], [14, 15], [15, 16]
                ]
    eps = 1e-7
    limbs = x[:, :, limbs_id, :]
    limbs = limbs[:, :, :, 1, :] - limbs[:, :, :, 0, :]

    polar_angle = torch.atan2(torch.sqrt(limbs[:,:,:,0] ** 2 + limbs[:,:,:,1] ** 2), limbs[:,:,:,2]) / torch.pi *180    #b,f,16
    zero = torch.zeros((polar_angle.shape[0], polar_angle.shape[1], 1), device=polar_angle.device)      #b,f,1
    polar_angle = torch.cat([zero, polar_angle], dim=2)     #b,f,17
    polar_angle = polar_angle //30       #b,f,17
    return polar_angle


def compute_loss(output, y):

    b, f, n, _ = output.shape
    y = y.view(-1)
    output = output.view(-1, output.shape[-1])
    criterion = CrossEntropyLoss()
    loss = criterion(output, y)

    return loss


def train_one_epoch(args, model, train_loader, optimizer, device, losses):
    model.train()
    for x, y in tqdm(train_loader):
        batch_size = x.shape[0]
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            if args.root_rel:
                y = y - y[..., 0:1, :]
            else:
                y[..., 2] = y[..., 2] - y[:, 0:1, 0:1, 2]  # Place the depth of first frame root to be 0
            y = get_angles(y).to(torch.long)

        pred = model(x)  # (N, T, 17, num_class)

        optimizer.zero_grad()

        loss_3d_pos = compute_loss(pred, y)

        loss_total = loss_3d_pos

        losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)

        loss_total.backward()
        optimizer.step()


def evaluate(args, model, test_loader, device):
    print("[INFO] Evaluation")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            if args.root_rel:
                y = y - y[..., 0:1, :]
            y = get_angles(y).to(torch.long)

            predicted_3d_pos = model(x)
            preds = torch.argmax(predicted_3d_pos, dim=-1)

            preds[:, :, 0] = 0
            all_preds.append(preds.view(-1).cpu())
            all_labels.append(y.view(-1).cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return accuracy, precision, recall, f1



def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, max_accuracy, wandb_id):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'max_accuracy': max_accuracy,
        'wandb_id': wandb_id,
    }, checkpoint_path)


def train(args, opts):
    # use to store
    if opts.use_log:
        file_path = 'log/' + opts.wandb_name
        print(file_path)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        logfile = os.path.join(file_path, 'logging.log')
        sys.stdout = Logger(logfile)

    print_args(args)
    create_directory_if_not_exists(opts.new_checkpoint)

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')

    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        'prefetch_factor': (opts.num_cpus - 1) // 3,
        'persistent_workers': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_anglemodel(args)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    n_params = count_param_numbers(model)
    print(f"[INFO] Number of parameters: {n_params:,}")

    lr = args.learning_rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr,
                            weight_decay=args.weight_decay)
    lr_decay = args.lr_decay
    epoch_start = 0
    # max_accuracy corresponds to the loss between the target angles and the current predicted angles.
    max_accuracy = 0  # Used for storing the best model
    wandb_id = opts.wandb_run_id if opts.wandb_run_id is not None else wandb.util.generate_id()

    if opts.checkpoint:
        checkpoint_path = os.path.join(opts.checkpoint, opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth.tr")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=True)

            if opts.resume:
                lr = checkpoint['lr']
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                max_accuracy = checkpoint['max_accuracy']
                if 'wandb_id' in checkpoint and opts.wandb_run_id is None:
                    wandb_id = checkpoint['wandb_id']
        else:
            print("[WARN] Checkpoint path is empty. Starting from the beginning")
            opts.resume = False

    if not opts.eval_only:
        if opts.resume:
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                           project='forecast_polar_angle',
                           resume="allow",
                           settings=wandb.Settings(start_method='fork'))
        else:
            print(f"Run ID: {wandb_id}")
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                           name=opts.wandb_name,
                           project='forecast_polar_angle',
                           settings=wandb.Settings(start_method='fork'))
                wandb.config.update({"run_id": wandb_id})
                wandb.config.update(args)
                installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
                wandb.config.update({'installed_packages': installed_packages})

    checkpoint_path_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth.tr')
    checkpoint_path_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth.tr')

    for epoch in range(epoch_start, args.epochs):
        if opts.eval_only:
            evaluate(args, model, test_loader, device)
            exit()

        print(f"[INFO] epoch {epoch}")
        loss_names = ['3d_pose', 'total']
        losses = {name: AverageMeter() for name in loss_names}

        train_one_epoch(args, model, train_loader, optimizer, device, losses)

        accuracy, precision, recall, f1 = evaluate(args, model, test_loader, device)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, max_accuracy, wandb_id)
        save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, max_accuracy, wandb_id)

        if opts.use_wandb:
            wandb.log({
                'lr': lr,
                'train/loss_3d_pose': losses['3d_pose'].avg,
                'train/total': losses['total'].avg,
                'eval/accuracy': accuracy,
                'eval/max_accuracy': max_accuracy,
                'eval/precision': precision,
                'eval/recall': recall,
                'eval/f1': f1,
            }, step=epoch + 1)

        lr = decay_lr_exponentially(lr, lr_decay, optimizer)

    if opts.use_wandb:
        artifact = wandb.Artifact(f'model', type='model')
        artifact.add_file(checkpoint_path_latest)
        artifact.add_file(checkpoint_path_best)
        wandb.log_artifact(artifact)


def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    args = get_config(opts.config)

    train(args, opts)


if __name__ == '__main__':
    main()
