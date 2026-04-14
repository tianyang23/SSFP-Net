import argparse
import csv
import json
import torch
import datetime
import os
import random
import sys
import numpy as np
from tqdm import tqdm
from torch import nn, optim
import torch.nn.init as init
from itertools import zip_longest
from torch.optim import lr_scheduler
from ResUNet_3D import ResUNet3D
from graphical_regularization import compute_similarity_weights, laplacian_regularization
from dataset_xsd_fold import create_dataloaders
from losses_xsd import DiceLoss, TverskyLoss, CombinedLoss, FocalLoss, BoundaryLoss, HausdorffLoss, soft_dice_cldice
from cbDice_loss import SoftcbDiceLoss
from metrics import dice, accuracy, jaccard, sensitivity, clDice

# Logging setup.
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
RUN_TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(log_dir, f'train_{RUN_TIMESTAMP}.log')


class Tee:


    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


METRIC_KEYS = [
    'loss', 'dice', 'cldice', 'acc', 'iou', 'sen'
]


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_weights(net, init_type='xavier', seed=None):


    if seed is not None:
        set_seed(seed)
    for m in net.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            if init_type == 'xavier':
                init.xavier_normal_(m.weight)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'normal':
                init.normal_(m.weight, 0, 0.02)
            else:
                raise NotImplementedError(f"Initialization method {init_type} is not implemented")
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)


def evaluate_model(net, data_loader, criterion, device, desc='Evaluation'):


    net.eval()
    total_loss = 0.0
    dice_val = 0.0
    cldice_val = 0.0
    jaccard_val = 0.0
    sensitivity_val = 0.0
    accuracy_val = 0.0
    num_batches = 0

    with torch.no_grad():
        # Keep evaluation progress visible in the terminal.
        for imgs, true_masks in tqdm(data_loader, desc=desc):
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            masks_pred, _ = net(imgs)
            loss = criterion(masks_pred, true_masks)
            total_loss += loss.item()


            pred = torch.sigmoid(masks_pred)
            pred = (pred > 0.5).float()

            pred_np = pred.cpu().numpy()
            true_masks_np = true_masks.cpu().numpy()


            dice_val += dice(pred_np, true_masks_np)
            cldice_val += clDice(pred_np, true_masks_np)
            jaccard_val += jaccard(pred_np, true_masks_np)
            sensitivity_val += sensitivity(pred_np, true_masks_np)
            accuracy_val += accuracy(pred_np, true_masks_np)
            num_batches += 1


    if num_batches == 0:
        return {key: 0.0 for key in METRIC_KEYS}

    return {
        'loss': total_loss / num_batches,
        'dice': dice_val / num_batches,
        'cldice': cldice_val / num_batches,
        'acc': accuracy_val / num_batches,
        'iou': jaccard_val / num_batches,
        'sen': sensitivity_val / num_batches,
    }


def create_model_pair():


    net1 = ResUNet3D(num_input_channels=1, num_output_channels=1)
    net2 = ResUNet3D(num_input_channels=1, num_output_channels=1)
    initialize_weights(net1, init_type='xavier', seed=42)
    initialize_weights(net2, init_type='xavier', seed=123)
    net1.to(device)
    net2.to(device)
    return net1, net2


def train_cps(net1, net2, device, train_loader, unsuped_loader, val_loader, epochs=5, batch_size=1,
              lr1=0.01, lr2=0.001, run_name='default'):


    optimizer1 = optim.Adam(net1.parameters(), lr=lr1)
    optimizer2 = optim.SGD(net2.parameters(), lr=lr2, momentum=0.99, nesterov=True)
    scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=40, gamma=0.5)
    scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()
    soft_cbDice = SoftcbDiceLoss()
    best_dice = 0.0
    best_model_idx = 1
    best_model_path = ''
    best_epoch = 0
    best_val_metrics = {}


    start_timestamp = datetime.datetime.now().strftime('%Y %m %d-%H %M')
    print(f'Training started at {start_timestamp} | {run_name}')


    os.makedirs('cps_train_best', exist_ok=True)
    output_dir = f'cps_train_best/{run_name}_{start_timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):

        lambda_cps = 0.5 * np.exp(-5 * (1 - epoch / epochs) ** 2)
        print(f'\nStarting epoch {epoch + 1}/{epochs}, lambda_cps: {lambda_cps}')
        net1.train()
        net2.train()
        train_loss_sup1 = 0.0
        train_loss_sup2 = 0.0
        epoch_cps_loss = 0.0
        epoch_grl_loss = 0.0
        epoch_ske_loss = 0.0


        train_loop = tqdm(
            zip_longest(unsuped_loader, train_loader, fillvalue=None),
            total=len(unsuped_loader),
            desc=f'Training Epoch {epoch + 1}/{epochs}'
        )
        for data in train_loop:
            unsup_imgs, imgs_and_gts = data


            if imgs_and_gts is None:
                continue
            imgs, gts = imgs_and_gts

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            imgs = imgs.to(device=device, dtype=torch.float32)
            gts = gts.to(device=device, dtype=torch.float32)
            unsup_imgs = unsup_imgs.to(device=device, dtype=torch.float32)


            masks_pred_sup1, features_sup1 = net1(imgs)
            masks_pred_unsup1, features_unsup1 = net1(unsup_imgs)

            masks_pred_sup2, features_sup2 = net2(imgs)
            masks_pred_unsup2, features_unsup2 = net2(unsup_imgs)


            pred_l = torch.cat([masks_pred_sup1, masks_pred_unsup1], dim=0)
            pred_r = torch.cat([masks_pred_sup2, masks_pred_unsup2], dim=0)

            pseudo_probs_l = torch.sigmoid(pred_l)
            pseudo_probs_r = torch.sigmoid(pred_r)
            pseudo_labels_l = pseudo_probs_l
            pseudo_labels_r = pseudo_probs_r


            # Cross pseudo supervision couples the two branches.
            cps_loss = criterion(pred_l, pseudo_labels_r) + criterion(pred_r, pseudo_labels_l)
            cps_loss = cps_loss * lambda_cps


            loss_seg_sup1 = criterion(masks_pred_sup1, gts)
            loss_seg_sup2 = criterion(masks_pred_sup2, gts)
            seg_loss = loss_seg_sup1 + loss_seg_sup2 + cps_loss


            loss_ske_sup1 = soft_cbDice(masks_pred_sup1, gts)
            loss_ske_sup2 = soft_cbDice(masks_pred_sup2, gts)
            ske_loss = loss_ske_sup1 + loss_ske_sup2


            i_indices_sup1, j_indices_sup1, weights_sup1 = compute_similarity_weights(features_sup1)
            grl_loss_sup1 = laplacian_regularization(features_sup1, i_indices_sup1, j_indices_sup1, weights_sup1)

            i_indices_sup2, j_indices_sup2, weights_sup2 = compute_similarity_weights(features_sup2)
            grl_loss_sup2 = laplacian_regularization(features_sup2, i_indices_sup2, j_indices_sup2, weights_sup2)

            grl_loss = grl_loss_sup1 + grl_loss_sup2


            alpha = 0.8
            beta = 0.2
            loss = alpha * seg_loss + beta * grl_loss + ske_loss
            loss.backward()


            torch.nn.utils.clip_grad_norm_(net1.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(net2.parameters(), max_norm=5.0)

            optimizer1.step()
            optimizer2.step()


            train_loss_sup1 += loss_seg_sup1.item()
            train_loss_sup2 += loss_seg_sup2.item()
            epoch_cps_loss += cps_loss.item()
            epoch_grl_loss += grl_loss.item() if hasattr(grl_loss, 'item') else grl_loss
            epoch_ske_loss += ske_loss.item() if hasattr(ske_loss, 'item') else ske_loss

        scheduler1.step()
        scheduler2.step()
        current_lr1 = scheduler1.get_last_lr()[0]
        current_lr2 = scheduler2.get_last_lr()[0]

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Current learning rates: net1 = {current_lr1}, net2 = {current_lr2}')
        print(
            f'train_sup1 loss: {train_loss_sup1 / len(train_loader)}, '
            f'train_sup2 loss: {train_loss_sup2 / len(train_loader)}, '
            f'CPS loss: {epoch_cps_loss / len(train_loader)},\n '
            f'GRl loss: {epoch_grl_loss / len(train_loader)}, '
            f'SKE loss: {epoch_ske_loss / len(train_loader)}'
        )


        val_metrics_1 = evaluate_model(net1, val_loader, criterion, device, desc=f'Validation-Model1 {epoch + 1}/{epochs}')
        val_metrics_2 = evaluate_model(net2, val_loader, criterion, device, desc=f'Validation-Model2 {epoch + 1}/{epochs}')

        print(f"Validation loss1: {val_metrics_1['loss']:.6f}")
        print(f"Validation Dice1: {val_metrics_1['dice']:.6f}")
        print(f"Validation clDice1: {val_metrics_1['cldice']:.6f}")
        print(f"Validation Acc1: {val_metrics_1['acc']:.6f}")
        print(f"Validation IoU1: {val_metrics_1['iou']:.6f}")
        print(f"Validation Sen1: {val_metrics_1['sen']:.6f}")
        print('---------------------------')
        print(f"Validation loss2: {val_metrics_2['loss']:.6f}")
        print(f"Validation Dice2: {val_metrics_2['dice']:.6f}")
        print(f"Validation clDice2: {val_metrics_2['cldice']:.6f}")
        print(f"Validation Acc2: {val_metrics_2['acc']:.6f}")
        print(f"Validation IoU2: {val_metrics_2['iou']:.6f}")
        print(f"Validation Sen2: {val_metrics_2['sen']:.6f}")


        if val_metrics_1['dice'] > best_dice or val_metrics_2['dice'] > best_dice:
            if val_metrics_1['dice'] > val_metrics_2['dice']:
                best_dice = val_metrics_1['dice']
                best_model_idx = 1
                best_epoch = epoch + 1
                best_val_metrics = val_metrics_1.copy()
                best_model_path = f'{output_dir}/model1_best_dice_{best_dice:.6f}.pth'
                print(f'Saving model 1 at epoch {epoch + 1} with validation Dice Coefficient: {best_dice}')
                torch.save(net1.state_dict(), best_model_path)
            else:
                best_dice = val_metrics_2['dice']
                best_model_idx = 2
                best_epoch = epoch + 1
                best_val_metrics = val_metrics_2.copy()
                best_model_path = f'{output_dir}/model2_best_dice_{best_dice:.6f}.pth'
                print(f'Saving model 2 at epoch {epoch + 1} with validation Dice Coefficient: {best_dice}')
                torch.save(net2.state_dict(), best_model_path)

    print('Training finished!')
    return {
        'run_name': run_name,
        'output_dir': output_dir,
        'best_model_idx': best_model_idx,
        'best_model_path': best_model_path,
        'best_epoch': best_epoch,
        'best_val_metrics': best_val_metrics,
        'best_val_dice': best_dice,
    }


def load_best_model(best_model_idx, best_model_path):


    model = ResUNet3D(num_input_channels=1, num_output_channels=1)
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def save_json(data, path):

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_summary_csv(results, path):

    fieldnames = [
        'fold', 'best_model_idx', 'best_epoch', 'best_model_path',
        'val_loss', 'val_dice', 'val_cldice', 'val_acc', 'val_iou', 'val_sen',
        'test_loss', 'test_dice', 'test_cldice', 'test_acc', 'test_iou', 'test_sen'
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({
                'fold': item['fold'],
                'best_model_idx': item['best_model_idx'],
                'best_epoch': item['best_epoch'],
                'best_model_path': item['best_model_path'],
                'val_loss': item['best_val_metrics'].get('loss', 0.0),
                'val_dice': item['best_val_metrics'].get('dice', 0.0),
                'val_cldice': item['best_val_metrics'].get('cldice', 0.0),
                'val_acc': item['best_val_metrics'].get('acc', 0.0),
                'val_iou': item['best_val_metrics'].get('iou', 0.0),
                'val_sen': item['best_val_metrics'].get('sen', 0.0),
                'test_loss': item['test_metrics'].get('loss', 0.0),
                'test_dice': item['test_metrics'].get('dice', 0.0),
                'test_cldice': item['test_metrics'].get('cldice', 0.0),
                'test_acc': item['test_metrics'].get('acc', 0.0),
                'test_iou': item['test_metrics'].get('iou', 0.0),
                'test_sen': item['test_metrics'].get('sen', 0.0),
            })


def aggregate_fold_results(results):


    summary = {}
    for prefix in ['best_val_metrics', 'test_metrics']:
        prefix_name = 'val' if prefix == 'best_val_metrics' else 'test'
        summary[prefix_name] = {
            key: float(np.mean([item[prefix].get(key, 0.0) for item in results]))
            for key in METRIC_KEYS
        }
    return summary


def parse_args():


    parser = argparse.ArgumentParser(description='Train CPS model with 5-fold cross validation splits')
    parser.add_argument('--base_dir', type=str, default=os.environ.get('IRCADB_BASE_DIR', r'/home/gpuserver/zhz/Datasets-3/3Dircadb/fold'))
    parser.add_argument('--splits_dir', type=str, default=os.environ.get('IRCADB_SPLITS_DIR', os.path.join(os.path.dirname(__file__), '3Dircadb_5fold_splits_ssl')))
    parser.add_argument('--fold', type=int, default=int(os.environ.get('IRCADB_FOLD', 1)), choices=[1, 2, 3, 4, 5])
    parser.add_argument('--run_all_folds', action='store_true', help='Run fold1-fold5 automatically and summarize results.')
    parser.add_argument('--label_ratio', type=int, default=int(os.environ.get('IRCADB_LABEL_RATIO', 50)), choices=[20, 50])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--labeled_batch_size', type=int, default=16)
    parser.add_argument('--unlabeled_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--lr1', type=float, default=0.001)
    parser.add_argument('--lr2', type=float, default=0.001)
    parser.add_argument('--patch_d', type=int, default=16)
    parser.add_argument('--patch_h', type=int, default=16)
    parser.add_argument('--patch_w', type=int, default=16)
    parser.add_argument('--stride_d', type=int, default=8)
    parser.add_argument('--stride_h', type=int, default=8)
    parser.add_argument('--stride_w', type=int, default=8)
    parser.add_argument('--results_dir', type=str, default='cv_results')
    return parser.parse_args()


def build_loaders_for_fold(args, fold):

    # Build patch and stride tuples from CLI arguments.
    patch_size = (args.patch_d, args.patch_h, args.patch_w)
    stride = (args.stride_d, args.stride_h, args.stride_w)
    return create_dataloaders(
        base_dir=args.base_dir,
        splits_dir=args.splits_dir,
        fold=fold,
        label_ratio=args.label_ratio,
        patch_size=patch_size,
        stride=stride,
        labeled_batch_size=args.labeled_batch_size,
        unlabeled_batch_size=args.unlabeled_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
    )


def print_split_info(loaders, fold, label_ratio):


    print(f'Using fold {fold}, label ratio {label_ratio}%')
    print('Patients in labeled train set:')
    for patient in loaders['split_patients']['train_labeled']:
        print(patient)
    print('\nPatients in unlabeled train set:')
    for patient in loaders['split_patients']['train_unlabeled']:
        print(patient)
    print('\nPatients in validation set:')
    for patient in loaders['split_patients']['val']:
        print(patient)
    print('\nPatients in test set:')
    for patient in loaders['split_patients']['test']:
        print(patient)
    print()


def run_one_fold(args, fold):

    # Train one fold, reload the best checkpoint, and evaluate on the test set.
    loaders = build_loaders_for_fold(args, fold)
    print_split_info(loaders, fold, args.label_ratio)

    net1, net2 = create_model_pair()
    run_name = f'fold{fold}_label{args.label_ratio}'
    train_result = train_cps(
        net1=net1,
        net2=net2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr1=args.lr1,
        lr2=args.lr2,
        device=device,
        train_loader=loaders['train_labeled_dataloader'],
        unsuped_loader=loaders['train_unlabeled_dataloader'],
        val_loader=loaders['val_dataloader'],
        run_name=run_name,
    )

    criterion = nn.BCEWithLogitsLoss()
    best_model = load_best_model(train_result['best_model_idx'], train_result['best_model_path'])
    test_metrics = evaluate_model(best_model, loaders['test_dataloader'], criterion, device, desc=f'Test-Fold{fold}')

    fold_result = {
        'fold': fold,
        'label_ratio': args.label_ratio,
        'best_model_idx': train_result['best_model_idx'],
        'best_model_path': train_result['best_model_path'],
        'best_epoch': train_result['best_epoch'],
        'best_val_metrics': train_result['best_val_metrics'],
        'test_metrics': test_metrics,
        'split_patients': loaders['split_patients'],
    }

    os.makedirs(args.results_dir, exist_ok=True)
    fold_result_path = os.path.join(args.results_dir, f'fold{fold}_label{args.label_ratio}_result.json')
    save_json(fold_result, fold_result_path)
    print(f'Fold {fold} test Dice: {test_metrics["dice"]:.6f}')
    print(f'Fold {fold} result saved to: {fold_result_path}')
    return fold_result


def run_all_folds(args):


    results = []
    folds = [1, 2, 3, 4, 5]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_results_dir = os.path.join(args.results_dir, f'label{args.label_ratio}_{timestamp}')
    os.makedirs(run_results_dir, exist_ok=True)
    args.results_dir = run_results_dir

    for fold in folds:
        print(f'\n==================== Running fold {fold} ====================')
        fold_result = run_one_fold(args, fold)
        results.append(fold_result)

    aggregate = aggregate_fold_results(results)
    summary = {
        'label_ratio': args.label_ratio,
        'folds': folds,
        'results': results,
        'average': aggregate,
    }

    summary_json_path = os.path.join(run_results_dir, f'five_fold_summary_label{args.label_ratio}.json')
    summary_csv_path = os.path.join(run_results_dir, f'five_fold_summary_label{args.label_ratio}.csv')
    save_json(summary, summary_json_path)
    write_summary_csv(results, summary_csv_path)

    print('\n==================== 5-Fold Mean Results ====================')
    print('Validation mean metrics:')
    for key, value in aggregate['val'].items():
        print(f'  {key}: {value:.6f}')
    print('Test mean metrics:')
    for key, value in aggregate['test'].items():
        print(f'  {key}: {value:.6f}')
    print(f'JSON summary saved to: {summary_json_path}')
    print(f'CSV summary saved to: {summary_csv_path}')


def main():


    log_f = open(LOG_FILE, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    print(f'Log file: {LOG_FILE}')

    args = parse_args()
    if args.run_all_folds:
        run_all_folds(args)
    else:
        run_one_fold(args, args.fold)


if __name__ == '__main__':
    main()
