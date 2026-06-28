# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import cv2
import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from time import time

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable as V

from networks.FHNet import Instruct_FHNet
from framework4 import MyFrame
from loss import dice_bce_loss
from data1 import ImageFolder
import Constants9

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

# ==============================
# TTAFrame for LAVT (with text input)
# ==============================
class TTAFrame():
    def __init__(self, net_class):
        self.net = net_class().cuda()
        # Do NOT use DataParallel unless your training used it

    def load(self, path):
        state_dict = torch.load(path)
        self.net.load_state_dict(state_dict)

    def test_one_img_from_path(self, path, text, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * 4
        if batchsize >= 4:
            return self.test_one_img_from_path_4(path, text)

    def test_one_img_from_path_4(self, path, text):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.resize(img, (1024, 1024))
        img1 = img
        img2 = img1[::-1]
        img3 = img1[:, ::-1]
        img4 = img2[:, ::-1]

        def preprocess(im):
            im = im.transpose(2, 0, 1)
            im = np.array(im, np.float32) / 255.0 * 3.2 - 1.6
            return torch.from_numpy(im).unsqueeze(0)

        img1 = preprocess(img1)
        img2 = preprocess(img2)
        img3 = preprocess(img3)
        img4 = preprocess(img4)

        imgs = torch.cat([img1, img2, img3, img4], dim=0).cuda()
        text_tensor = torch.tensor(text).unsqueeze(0).repeat(4, 1).cuda()

        with torch.no_grad():
            masks = self.net(imgs, text_tensor).squeeze(1).cpu().numpy()

        mask2 = masks[0] + masks[1][::-1] + masks[2][:, ::-1] + masks[3][::-1, ::-1]
        return mask2

def accuracy(pred_mask, label):
    pred_mask = pred_mask.astype(np.uint8)
    TP = FN = TN = FP = 0
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred_mask[i][j] == 1:
                    FP += 1
                else:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP + 1e-6)
    sen = TP / (TP + FN + 1e-6)
    iou = TP / (TP + FN + FP + 1e-6)
    pre = TP / (TP + FP + 1e-6)
    f1 = (2 * pre * sen) / (pre + sen + 1e-6)
    return acc, sen, iou, pre, f1

def validate_model(weight_path, image_dir, mask_dir, json_file, threshold=3.6, disc=20):
    solver = TTAFrame(Instruct_FHNet)
    solver.load(weight_path)

    with open(json_file, 'r') as f:
        instruction = json.load(f)

    image_names = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    iou_list = []

    for name in image_names:
        # Get text instruction
        text = None
        for item in instruction:
            if name in item:
                text = item[name]
                break
        if text is None:
            continue

        img_path = os.path.join(image_dir, name)
        if not os.path.exists(img_path):
            continue

        try:
            mask = solver.test_one_img_from_path(img_path, text)
        except Exception as e:
            print(f"Skip {name}: {e}")
            continue

        # Post-process: thresholding
        mask[mask > threshold] = 255
        mask[mask <= threshold] = 0

        # Load GT
        gt_name = name.split('_')[0] + '_labels.jpg'
        gt_path = os.path.join(mask_dir, gt_name)
        if not os.path.exists(gt_path):
            gt_path = os.path.join(mask_dir, name)
        if not os.path.exists(gt_path):
            continue

        gt_img = cv2.imread(gt_path)
        if gt_img is None:
            continue
        ground_truth = gt_img[:, :, 1]  # green channel

        # Resize prediction to GT size
        mask = cv2.resize(mask, (ground_truth.shape[1], ground_truth.shape[0]))

        # Binarize
        pred_binary = np.zeros_like(mask)
        pred_binary[mask > disc] = 1
        gt_binary = np.zeros_like(ground_truth)
        gt_binary[ground_truth > 0] = 1

        _, _, iou, _, _ = accuracy(pred_binary, gt_binary)
        iou_list.append(iou)

    mean_iou = np.mean(iou_list) if iou_list else 0.0
    print(f"Validation IoU: {mean_iou:.4f}")
    return mean_iou

# ==============================
# Paths
# ==============================
VAL_JSON_FILE = '/data/yangzhigang/RCFSNet/dataset/paris_build/test_CAPTIONS_parisBuild_road.json'
VAL_IMAGE_DIR = './dataset/paris_build/test/images/'
VAL_MASK_DIR = './dataset/paris_build/test/road_mask'

def CE_Net_Train():
    NAME = 'Instruct_FHNet' + Constants9.ROOT.split('/')[-1]
    print(NAME)

    solver = MyFrame(Instruct_FHNet, dice_bce_loss, 2e-4)
    solver.load('./weights/Instruct_FHNetparis_build_temp.th')
    batchsize = torch.cuda.device_count() * Constants9.BATCHSIZE_PER_CARD

    dataset = ImageFolder(
        root_path=Constants9.ROOT,
        datasets='parisbuild_text',
        jsoncap_path=Constants9.JSONcap_path,
        jsoncaplen_path=Constants9.JSONcaplen_path
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()

    no_optim = 0
    total_epoch = Constants9.TOTAL_EPOCH
    train_epoch_best_loss = Constants9.INITAL_EPOCH_LOSS
    best_val_iou = 0.0
    start_epoch=203

    for epoch in range(start_epoch, total_epoch + 1):
        solver.net.train()
        train_epoch_loss = 0
        for img, mask, text, caplen in data_loader:
            solver.set_input(img, mask, text, caplen)
            train_loss, pred = solver.optimize()
            train_epoch_loss += train_loss

        train_epoch_loss /= len(data_loader)

        # ============ Validation ============
        val_iou = 0.0
        if epoch >= 285:  # or start later, e.g., epoch > 10
            temp_weight = f'./weights/{NAME}_temp.th'
            solver.save(temp_weight)
            val_iou = validate_model(
                temp_weight,
                VAL_IMAGE_DIR,
                VAL_MASK_DIR,
                VAL_JSON_FILE,
                threshold=3.6,
                disc=20
            )
            os.remove(temp_weight)  # optional: clean up

        # ============ Logging ============
        print('********', file=mylog)
        print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
        print('train_loss:', train_epoch_loss, file=mylog)
        print('val_iou (TTA):', val_iou, file=mylog)
        print('SHAPE:', Constants9.Image_size, file=mylog)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('val_iou (TTA):', val_iou)
        print('SHAPE:', Constants9.Image_size)

        # ============ Save best IoU model ============
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            solver.save('./weights/' + NAME + '_best_iou.th')
            print(f"Saved best IoU model with IoU = {best_val_iou:.4f}")

        # ============ Early stopping & LR ============
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('./weights/' + NAME + '.th')

        if no_optim > Constants9.NUM_EARLY_STOP:
            print('early stop at %d epoch' % epoch, file=mylog)
            break
        if no_optim > Constants9.NUM_UPDATE_LR:
            if solver.old_lr < 5e-7:
                break
            solver.load('./weights/' + NAME + '.th')
            solver.update_lr(5.0, factor=True, mylog=mylog)

        mylog.flush()

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()

if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)
    CE_Net_Train()