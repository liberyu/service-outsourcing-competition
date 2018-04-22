# -*- coding: utf-8 -*-
#from load_data import loadDataGeneral
from load_data2 import loadDataGeneral2
import numpy as np
import pandas as pd
import nibabel as nib
import os


"""
可在此基础上做单张图片的测试,必须要有label,求出dice值
而且target_floder 文件夹下边必须建好三个文件夹,pred,image,label
"""

def IoU(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def value_dice(csv_path,pred_path,label_path):
    df = pd.read_csv(csv_path)      
    ious, dices = [], []
    for i,item in df.iterrows():
        pred_img_path = os.path.join(pred_path,item[0][:-7]+'-pred.nii.gz')
        label_img_path = os.path.join(label_path,item[0])
        
        gt = nib.load(label_img_path).get_data()>0.5
        pr = nib.load(pred_img_path).get_data()>0.5
        iou = IoU(gt, pr)
        dice = Dice(gt, pr)
        ious.append(iou)
        dices.append(dice)
        print(pred_img_path)
        print(" {}th".format(i))
        print('iou:{}\t dice:{}'.format(iou,dice))
    print 'Mean IoU:'
    print np.array(ious).mean()

    print 'Mean Dice:'
    print np.array(dices).mean()       
    

if __name__ == '__main__':

    csv_path = './test.csv'
    #pred_path = '/home/yanyu/competition/data/test_rotate/pred'
    #label_path = '/home/yanyu/competition/data/test_rotate/label'
    pred_path = './vali_data/pred'
    label_path = './vali_data/label'
    value_dice(csv_path, pred_path, label_path)
    

 
