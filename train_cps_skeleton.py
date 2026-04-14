import torch
from datetime import datetime
import os
import random
import logging
import numpy as np
from tqdm import tqdm
from torch import nn, optim
import torch.nn.init as init
from itertools import zip_longest
from torch.optim import lr_scheduler
from ResUNet_3D import ResUNet3D
from graphical_regularization import compute_similarity_weights, laplacian_regularization
from dataset_xsd import val_dataloader, train_labeled_dataloader, train_unlabeled_dataloader
from losses_xsd import DiceLoss, TverskyLoss, CombinedLoss, FocalLoss, BoundaryLoss, HausdorffLoss, soft_dice_cldice
from cbDice_loss import SoftcbDiceLoss
from metrics import dice, accuracy, jaccard, sensitivity, clDice

"""Legacy CPS training entry point."""


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class DiceLoss(nn.Module):
#    def __init__(self, eps=1e-7):
#        super(DiceLoss, self).__init__()
#        self.eps = eps
#    def forward(self, output, target):
#        intersection = (output * target).sum(dim=(2, 3, 4))
#        union = output.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + self.eps
#        dice = 2 * intersection / union
#        return 1. - dice.mean()
# class CombinedLoss(nn.Module):
#    def __init__(self, alpha):
#        super(CombinedLoss, self).__init__()
#        self.alpha = alpha
#        self.dice_loss = DiceLoss()
#        self.bce_with_logits = nn.BCEWithLogitsLoss()
#    def forward(self, pred, target):
#        return self.alpha * self.bce_with_logits(pred, target) + (1 - self.alpha) * self.dice_loss(pred, target)


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


def dice_coefficient(predicted, true):
    smooth = 1.0
    intersection = (predicted * true).sum()
    return (2. * intersection + smooth) / (predicted.sum() + true.sum() + smooth)


def train_cps(net1, net2, device, train_loader, unsuped_loader, val_loader, epochs=5, batch_size=1, lr1=0.01, lr2=0.001):

    optimizer1 = optim.Adam(net1.parameters(), lr=lr1)
    # optimizer1 = optim.SGD(net1.parameters(), lr=lr1, momentum=0.9, nesterov=True)

    optimizer2 = optim.SGD(net2.parameters(), lr=lr2, momentum=0.99, nesterov=True)
    # optimizer2 = optim.Adam(net2.parameters(), lr=lr2)
    scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=40, gamma=0.5)
    scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()
    #soft_clDice = soft_dice_cldice()
    soft_cbDice = SoftcbDiceLoss()

    # criterion = CombinedLoss(alpha=0.9)
    #    criterion = DiceLoss()
    # criterion = TverskyLoss(alpha=0.1, beta=0.2)

    # criterion = FocalLoss(alpha=4.0, gamma=5.0)

    # criterion = BoundaryLoss(boundary_weight=1.0)

    # criterion = HausdorffLoss(distance_weight=1.0)

    best_dice = 0.0

    start_timestamp = datetime.now().strftime("%Y %m %d-%H %M")
    logger.info(f"Training started at {start_timestamp}")


    if not os.path.exists('cps_train_best'):
        os.makedirs('cps_train_best')


    output_dir = f'cps_train_best/{start_timestamp}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(epochs):

        # lambda_cps = min(0.1 + (epoch / epochs) * 0.1, 0.5)
        lambda_cps = 0.5 * np.exp(-5 * (1 - epoch / epochs) ** 2)
        # lambda_cps = 0.1
        
        for handler in logger.handlers:
            if hasattr(handler, "stream"):
                handler.stream.write("\n")
                handler.flush()
                
        logger.info(f"Starting epoch {epoch + 1}/{epochs}, lambda_cps: {lambda_cps}")
        
        net1.train()
        net2.train()
        train_loss_sup1 = 0
        train_loss_sup2 = 0
        epoch_cps_loss = 0
        epoch_grl_loss = 0
        epoch_ske_loss = 0

        train_loop = tqdm(zip_longest(unsuped_loader, train_loader, fillvalue=None), total=len(unsuped_loader),
                          desc=f"Training Epoch {epoch + 1}/{epochs}")
        for data in train_loop:
            unsup_imgs, imgs_and_gts = data

            if imgs_and_gts is None:

                continue
            imgs, gts = imgs_and_gts

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            imgs = imgs.to(device=device, dtype=torch.float32)
            gts = gts.to(device=device, dtype=torch.float32)  # Ground Truth
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
#            del i_indices_sup1, j_indices_sup1, weights_sup1


            i_indices_sup2, j_indices_sup2, weights_sup2 = compute_similarity_weights(features_sup2)
            grl_loss_sup2 = laplacian_regularization(features_sup2, i_indices_sup2, j_indices_sup2, weights_sup2)
#            del i_indices_sup2, j_indices_sup2, weights_sup2


#            adjacency_matrix_unsup1 = compute_similarity_weights(features_unsup1)
#            grl_loss_unsup2 = graphical_regularization(features_unsup2, adjacency_matrix_unsup1)
#            del adjacency_matrix_unsup1
#            adjacency_matrix_unsup2 = compute_similarity_weights(features_unsup2)
#            grl_loss_unsup1 = graphical_regularization(features_unsup1, adjacency_matrix_unsup2)
#            del adjacency_matrix_unsup2
#            cross_grl_loss = grl_loss_unsup1 + grl_loss_unsup2

            # grl_loss = grl_loss_sup1 + grl_loss_sup2 + cross_grl_loss
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
            epoch_grl_loss += grl_loss
            epoch_ske_loss += ske_loss

#            del imgs, gts, unsup_imgs, masks_pred_sup1, masks_pred_unsup1, masks_pred_sup2, masks_pred_unsup2
#            del features_sup1, features_unsup1, features_sup2, features_unsup2, pred_l, pred_r
#            torch.cuda.empty_cache()

        scheduler1.step()
        scheduler2.step()

        lr1 = scheduler1.get_last_lr()[0]
        lr2 = scheduler2.get_last_lr()[0]

        logger.info(f'Epoch {epoch + 1}/{epochs}')
        logger.info(f'Current learning rates: net1 = {lr1}, net2 = {lr2}')
        logger.info(f'train_sup1 loss: {train_loss_sup1 / len(train_loader)}, '
              f'train_sup2 loss: {train_loss_sup2 / len(train_loader)}, '
              f'CPS loss: {epoch_cps_loss / len(train_loader)},\n '
              f'GRl loss: {epoch_grl_loss / len(train_loader)}, '
              f'SKE loss: {epoch_ske_loss / len(train_loader)}')


        net1.eval()
        net2.eval()
        val_loss1 = 0
        val_loss2 = 0
        dice_val1 = 0
        dice_val2 = 0
        cldice_val1 = 0
        cldice_val2 = 0
        jaccard_val1 = 0
        jaccard_val2 = 0
        sensitivity_val1 = 0
        sensitivity_val2 = 0
        accuracy_val1 = 0
        accuracy_val2 = 0
        num_val_batches = 0


        val_loader_tqdm = tqdm(val_loader, desc=f"Validation {epoch + 1}/{epochs}")

        with torch.no_grad():
            for imgs, true_masks in tqdm(val_loader, desc=f"Validation {epoch + 1}/{epochs}"):
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred1, futures1 = net1(imgs)
                masks_pred2, futures2 = net2(imgs)


                # logger.info(f'masks_pred1 all zero: {(masks_pred1 == 0).all().item()}, all one: {(masks_pred1 == 1).all().item()}')
                # logger.info(f'masks_pred2 all zero: {(masks_pred2 == 0).all().item()}, all one: {(masks_pred2 == 1).all().item()}')

                loss1 = criterion(masks_pred1, true_masks)
                loss2 = criterion(masks_pred2, true_masks)

                val_loss1 += loss1.item()
                val_loss2 += loss2.item()

                pred1 = torch.sigmoid(masks_pred1)
                pred1 = (pred1 > 0.5).float()

                pred2 = torch.sigmoid(masks_pred2)
                pred2 = (pred2 > 0.5).float()

                pred1_np = pred1.cpu().numpy()
                pred2_np = pred2.cpu().numpy()
                true_masks_np = true_masks.cpu().numpy()

                # dice_val += dice_coefficient(pred, true_masks).item()


                dice_val1 += dice(pred1_np, true_masks_np)
                dice_val2 += dice(pred2_np, true_masks_np)

                cldice_val1 += clDice(pred1_np, true_masks_np)
                cldice_val2 += clDice(pred2_np, true_masks_np)

                jaccard_val1 += jaccard(pred1_np, true_masks_np)
                jaccard_val2 += jaccard(pred2_np, true_masks_np)

                sensitivity_val1 += sensitivity(pred1_np, true_masks_np)
                sensitivity_val2 += sensitivity(pred2_np, true_masks_np)

                accuracy_val1 += accuracy(pred1_np, true_masks_np)
                accuracy_val2 += accuracy(pred2_np, true_masks_np)
                num_val_batches += 1

        dice_val1 /= num_val_batches
        dice_val2 /= num_val_batches
        cldice_val1 /= num_val_batches
        cldice_val2 /= num_val_batches
        jaccard_val1 /= num_val_batches
        jaccard_val2 /= num_val_batches
        sensitivity_val1 /= num_val_batches
        sensitivity_val2 /= num_val_batches
        accuracy_val1 /= num_val_batches
        accuracy_val2 /= num_val_batches

        avg_val_loss1 = val_loss1 / num_val_batches
        avg_val_loss2 = val_loss2 / num_val_batches

        logger.info(f'Validation loss1: {avg_val_loss1:.6f}')
        logger.info(f'Validation Dice1: {dice_val1:.6f}')
        logger.info(f'Validation clDice1: {cldice_val1:.6f}')
        logger.info(f'Validation Acc1: {accuracy_val1:.6f}')
        logger.info(f'Validation IoU1: {jaccard_val1:.6f}')
        logger.info(f'Validation Sen1: {sensitivity_val1:.6f}')
        logger.info('---------------------------')
        logger.info(f'Validation loss2: {avg_val_loss2:.6f}')
        logger.info(f'Validation Dice2: {dice_val2:.6f}')
        logger.info(f'Validation clDice2: {cldice_val2:.6f}')
        logger.info(f'Validation Acc2: {accuracy_val2:.6f}')
        logger.info(f'Validation IoU2: {jaccard_val2:.6f}')
        logger.info(f'Validation Sen2: {sensitivity_val2:.6f}')

        if dice_val1 > best_dice or dice_val2 > best_dice:
            if dice_val1 > dice_val2:
                best_dice = dice_val1
                best_dice_filename = f'{output_dir}/model1_best_dice_{best_dice:.6f}.pth'
                logger.info(f'Saving model 1 at epoch {epoch + 1} with validation Dice Coefficient: {best_dice}')
                torch.save(net1.state_dict(), best_dice_filename)
            else:
                best_dice = dice_val2
                best_dice_filename = f'{output_dir}/model2_best_dice_{best_dice:.6f}.pth'
                logger.info(f'Saving model 2 at epoch {epoch + 1} with validation Dice Coefficient: {best_dice}')
                torch.save(net2.state_dict(), best_dice_filename)

    logger.info('Training finished!')


net1 = ResUNet3D(num_input_channels=1, num_output_channels=1)
net2 = ResUNet3D(num_input_channels=1, num_output_channels=1)


initialize_weights(net1, init_type='xavier', seed=42)
initialize_weights(net2, init_type='xavier', seed=123)

net1.to(device)
net2.to(device)

train_cps(
    net1=net1,
    net2=net2,
    epochs=100,
    batch_size=4,
    lr1=0.001,  # net1
    lr2=0.001,  # net2
    device=device,
    train_loader=train_labeled_dataloader,
    unsuped_loader=train_unlabeled_dataloader,
    val_loader=val_dataloader
)
