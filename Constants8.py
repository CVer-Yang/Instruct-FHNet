import os
import shutil

# ########修改输入的图像的大小
# Image_size = (448, 448)
# Image_size = (512, 512)
#Image_size = (1024, 1024)
Image_size = (1024, 1024)
# ########模仿他的数据的放置方式，文件夹的命名
# ROOT = './dataset/DRIVE'
# ROOT = './dataset/ROAD'
ROOT = './dataset/Masroad_build'
JSONcap_path ='training/train_CAPTIONS_Mas_1_cap_per_img_1_min_word_freq.json'
JSONcaplen_path = 'training/train_CAPLENS_Mas_1_cap_per_img_1_min_word_freq.json'
# #######batch_size大小，根据训练的情况进行
BATCHSIZE_PER_CARD = 2
TOTAL_EPOCH = 400
INITAL_EPOCH_LOSS = 10000
NUM_EARLY_STOP = 8
NUM_UPDATE_LR = 5
BINARY_CLASS = 1
