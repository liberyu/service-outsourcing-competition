import numpy as np
import nibabel as nib
import pandas as pd
from keras.models import load_model
import os
import time
import csv

def squeeze_dim(img):
    if img.ndim == 4: 
        if img.shape[0] == 1:
            img = np.squeeze(img,axis=0)
        if img.shape[1] == 1:
            img = np.squeeze(img,axis=1)
        if img.shape[2] == 1:
            img = np.squeeze(img,axis=2)
        if img.shape[3] == 1:
            img = np.squeeze(img,axis=3)
    else:
        pass
    return img
def crop(img, a , b, c):
    '''
    有一个缺点就是 a,b,c的值只能是偶数,即便是设置为奇数,返回的shape也是(a-1,b-1,c-1)之类的偶数形状
    '''
    img_shape = img.shape
    img_a = img_shape[0]
    img_b = img_shape[1]
    img_c = img_shape[2]
    img_res = img[img_a/2-a/2:img_a/2+a/2, img_b/2-b/2:img_b/2+b/2, img_c/2-c/2:img_c/2+c/2]
    return img_res
def decrop(in_img,shell):
    """
    shell 是一个全零的壳子矩阵,用来装预测的图形
    in_img 是预测得到的被剪切图像
    """
    in_img_shape = in_img.shape
    shell_shape = shell.shape
    a = in_img_shape[0]
    b = in_img_shape[1]
    c = in_img_shape[2]
    img_a = shell_shape[0]
    img_b = shell_shape[1]
    img_c = shell_shape[2]
    shell[img_a/2-a/2:img_a/2+a/2, img_b/2-b/2:img_b/2+b/2, img_c/2-c/2:img_c/2+c/2] = in_img
    return shell
def make_csv(floder= 'sample/image',save_name='./test.csv'):
    """
    get all filename in floder to make a csv_file
    input:
        floder is a path(type = string)
    """
    list_file = os.listdir(floder)
    csvfile = file(save_name,'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['path'])
    list_file.sort()
    for file_name in list_file:
        writer.writerow([file_name])
    csvfile.close()
def loadDataGeneral(df,target_floder):
    """
    这个是将待测图片整体读入,然后记录shape 其中做了整体的归一化
    This function loads data stored in nifti format. Data should already be of
    appropriate shape.

    Inputs:
    - df: Pandas dataframe with two columns: image filenames and ground truth filenames.
    - path: Path to folder containing filenames from df.
    - append_coords: Whether to append coordinate channels or not.
    Returns:
    - X: Array of 3D images with 1 or 4 channels depending on `append_coords`.
    - y: Array of 3D masks with 1 channel.
    """
   
    X, shape_list = [], []
    for i, item in df.iterrows():

        img_path = os.path.join(target_floder,item[0])

        nii_img = nib.load(img_path)
        img = nii_img.get_data()
        
        img = squeeze_dim(img)
        shape = img.shape
        img = crop(img,96,96,80)

        img = np.array(img, dtype=np.float64)
        brain = img > 0
        #img -= img[brain].mean()
        #img /= img[brain].std()        
        img -= img.mean()
        img /= img.std()
        X.append(img)
        shape_list.append(shape)

    X = np.array(X)
    #X -= -5.134725821872465e-18
    #X /= 0.9999999999999917
    print(X.mean())
    
    print('std:{}'.format(X.std()))
    #X -= X.mean()   
    #X /= X.std()
    
    X = np.expand_dims(X, -1)  # X.shape  --> (4,128,128,64,1)
    

    print '### Dataset loaded'
    print '\t{}'.format(target_floder)
    print '\t{}\t'.format(X.shape)
    print '\tX:{:.1f}-{:.1f}\t'.format(X.min(), X.max())
    return X, shape_list
def prediction( target_floder):
    
    """
    载入的模型名称是my_model.hdf5
    用以针对一个文件夹下边的文件进行整体的预测,并保存预测图像到与target_floder文件夹相同子目录pred文件夹下
    这个做的对原图(未经过裁剪的图像)进行的推断,得到的是与原图相匹配的pred文件
    """
    start_time = time.time()
    make_csv(target_floder)
    
    df = pd.read_csv('test.csv') 
    X,shape_list= loadDataGeneral(df,target_floder)
    n_test = X.shape[0]   
    model_name = '1th_model.085.hdf5' # Model should be trained with the same `append_coords`
    model = load_model(model_name)
    pred = model.predict(X, batch_size=1)[..., 1]  
    save_pred_floder = target_floder[:target_floder.rfind("/")]+'/pred/'
    if not os.path.exists(save_pred_floder):
        os.makedirs(save_pred_floder)
    for i in range(n_test):

        pr = pred[i] > 0.5 # binary prediction
        shell = np.zeros(shape_list[i])
        pr = decrop(pr,shell)

        tImg = nib.load(os.path.join(target_floder , df.ix[i].path))
        nib.save(nib.Nifti1Image(2 * pr.astype('float'), affine=tImg.get_affine()),
                 save_pred_floder + df.ix[i].path[:-7]+'-pred.nii.gz')
    duration = time.time() - start_time
    return duration

if __name__ == "__main__":
    
    #target_floder = '/home/yanyu/competition/data/test_rotate/image'  #最后不能够出现'/'否则生成pred文件夹会出错
    target_floder = './vali_data/image'
    duration = prediction(target_floder)
    print('{}sec'.format(duration))