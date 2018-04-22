# -*- coding: utf-8 -*-
#from load_data import loadDataGeneral
from load_data2 import loadDataGeneral2
import numpy as np
import pandas as pd
import nibabel as nib
from keras.models import load_model
import time
from scipy.misc import imresize
from skimage.color import hsv2rgb, rgb2hsv, gray2rgb
from skimage import io, exposure

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


if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    start_time = time.time()
    
    csv_path = '/home/yanyu/competition/data/crop_preprocess_data/vali.csv'
    target_floder = '/home/yanyu/competition/data/crop_preprocess_data/data'


    df = pd.read_csv(csv_path)      

    # Load test data
    append_coords = False
   # X_old, y_old = loadDataGeneral(df,target_floder, path, append_coords)
    X,y = loadDataGeneral2(df, target_floder, append_coords)

    n_test = X.shape[0]             # n_test get the number of tested picture
    inpShape = X.shape[1:]          # inpShape get the shape of input i.e. a tuple record X.shape

    # Load model
    #model_name = '../pre_model/8th_model.110.hdf5' # Model should be trained with the same `append_coords`
    #model = load_model(model_name)
    #model_name2 = '../pre_model/8th_drop_model.080.hdf5'
    #model = load_model(model_name2)
    model = load_model(filepath='build_model1_3.185.hdf5')

                                                        #array[..., 1] ==> got the specified part along the last axis 
    pred = model.predict(X, batch_size=1)[..., 1]       # pred.shape -->(4,128,128,64)
    #pred2 = model2.predict(X, batch_size=1)[..., 1]
    
    #pred = (pred1*0.5+pred2*0.5)

    # Compute scores and visualize
    ious = np.zeros(n_test)
    dices = np.zeros(n_test)
    for i in range(n_test):
        gt = y[i, :, :, :, 1] > 0.5 # ground truth binary mask
        pr = pred[i] > 0.5 # binary prediction
        ifsave = False
        # Save 3D images with binary masks if needed
        if ifsave:
            tImg = nib.load(target_floder+ '/image/' + df.ix[i].path)
            nib.save(nib.Nifti1Image(255 * pr.astype('float'), affine=tImg.get_affine()), target_floder +'pred/' + df.ix[i].path+'-pred.nii.gz')
            #nib.save(nib.Nifti1Image(255 * gt.astype('float'), affine=tImg.get_affine()), target_floder +'pred/' + df.ix[i].path + '-gt.nii.gz')
        # Compute scores
        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)
        print df.ix[i]['path'][-21:-7], ious[i], dices[i]




    print 'Mean IoU:'
    print ious.mean()

    print 'Mean Dice:'
    print dices.mean()
    print('耗时：{}'.format(time.time()-start_time))
