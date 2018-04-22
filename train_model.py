# -*- coding: utf-8 -*-
from load_data2 import loadDataGeneral2
from opt_model import densu_net,half_dense_u1_1,build_model1_3
from build_model import build_model
from half_dense_u import half_dense_u1_4
from Dilateddensenet import  half_dense_u1_2
import pandas as pd
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras
from keras import backend as K
 
import csv
import numpy as np

'''
需要做一个新的vali_csv和修改vali_path
将Record_dice做更新
'''

class Record_dice(keras.callbacks.Callback):
    def __init__(self,vali_img,vali_msk,train_img,train_msk,vali_per_epoch):
        # keras.callbacks.Callback.__init__(self)
        self.vali_img = vali_img
        self.vali_msk = vali_msk  
        self.dice_list = []
        self.iou_list = []
        self.train_dice = []
        self.train_iou = []
        self.train_img = train_img
        self.train_msk = train_msk
        self.vali_per_epoch = vali_per_epoch
    def on_epoch_end(self, epoch, logs=None):
        #print("this epoch is:",epoch)
        pred = model.predict(self.train_img, batch_size=1)[..., 1]
        pr = pred > 0.5
        gt = self.train_msk[..., 1] > 0.5
        iou = IoU(y_true=gt, y_pred=pr)
        dice = Dice(y_true=gt , y_pred=pr)
        self.train_dice.append(dice)
        self.train_iou.append(iou)
        print("\n training:\t dice:{:.5f} \t iou:{:.5f}".format(dice,iou))            
        if epoch !=0 and epoch % self.vali_per_epoch == 0:
            print("this epoch is:",epoch)
            pred = model.predict(self.vali_img, batch_size=1)[..., 1]
            pr = pred > 0.5
            gt = self.vali_msk[..., 1] > 0.5
            iou = IoU(y_true=gt, y_pred=pr)
            dice = Dice(y_true=gt , y_pred=pr)
            self.iou_list.append(iou)
            self.dice_list.append(dice)
            print("validation:\t dice:{:.5f} \t iou:{:.5f}".format(dice,iou))

    def vis_losss(self):
        def make_log_csv(list_epoch,save_name,iou_list,dice_list):
            
            epoch_list = []
            for n in list_epoch:
                epoch_list.append(str(n)+"epoch")            
            
            csvfile = file(save_name,'wb')
            writer = csv.writer(csvfile)
            writer.writerow(['epoch','iou','dice'])
            for l,m,n in zip(epoch_list,iou_list,dice_list):
                writer.writerow([l,m,n])
            csvfile.close()
            
        list1 = (np.arange(len(self.iou_list))+1)*self.vali_per_epoch
        make_log_csv(list_epoch=list1, save_name="vali_log.csv", iou_list=self.iou_list, dice_list=self.dice_list)
        list2 = (np.arange(len(self.train_dice))+1)
        make_log_csv(list2, "training_log.csv", self.train_iou, self.train_dice)    #同时将训练的dice和iou记录在表格中
 
        plt.figure(1)
        plt.plot(list1,self.iou_list,label='vali_iou')
        plt.plot(list1,self.dice_list,label='vali_dice')
        plt.xlabel('epochs')
        plt.ylabel('vali-accuracy')
        plt.legend()
        plt.title('The validation process')   
        plt.savefig("record_dice.jpg")
        plt.show()

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
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    train_csv_path = '/home/yanyu/train/Demo/crop.csv'
    vali_csv_path = '/home/yanyu/train/Demo/vali.csv'
    train_data_floder = '/home/yanyu/train/Demo/crop'
    vali_data_floder = '/home/yanyu/train/Demo/crop'
    # Path to the folder with images. Images will be read from path + path_from_csv


    df = pd.read_csv(train_csv_path)
    df2 = pd.read_csv(vali_csv_path)

    # Shuffle rows in dataframe. Random state is set for reproducibility.
    df = df.sample(frac=1, random_state=23)

    # Load training data
    append_coords = False
    X, y = loadDataGeneral2(df, train_data_floder, append_coords)
    vali_img, vali_msk = loadDataGeneral2(df2,vali_data_floder, append_coords)

    continue_train = False

    if continue_train:
        model = load_model('6th_model.070.hdf5')
        ini_epoch = 70
    else:
        ###############################################################################
    # Build model
        inp_shape = X[0].shape
        model = build_model1_3(inp_shape)
        #model = half_dense_u1_1(inp_shape)
        #model = densu_net(inp_shape)

        model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=[dice_coef])

    # Visualize model
        plot_model(model, 'model.png', show_shapes=True)

        model.summary()
        ini_epoch = 0

    ##########################################################################################
 #
    checkpointer = ModelCheckpoint('9th_model.{epoch:03d}.hdf5', period=5)

    record_dice = Record_dice(vali_img, vali_msk,train_img=X,train_msk=y,vali_per_epoch=5)          
                

    model.fit(X, y, batch_size=1, initial_epoch=ini_epoch,  epochs=200, callbacks=[checkpointer,record_dice])
    
    record_dice.vis_losss()


