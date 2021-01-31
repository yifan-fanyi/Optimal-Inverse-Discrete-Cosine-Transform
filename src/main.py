# 2020.09.07
# finalized 2020.09.26
# @yifan
#
# use DCT/PCA as foreword kernel
# use pinv / Linear Regression find the optimal inverse transformation kernel
# write kerenel to txt file, manually copy the kernels to following array in files:
#   fore_K in <jfdctflt.c> 
#   inv_K in <jidctflt.c>
# use <-dct float> mode when running the jpg codec
#   ./cjpeg -dct float -quality "$N" -outfile "$i"_"$N".jpg  "$i".bmp
#   ./djpeg -dct float -bmp -outfile "$i"_"$N".bmp  "$i"_"$N".jpg

import numpy as np
from sklearn.decomposition import PCA
import copy
from sklearn.linear_model import LinearRegression
from scipy.fftpack import dct, idct
import warnings
warnings.filterwarnings("ignore")

from utli import *
from evaluate import PSNR, SSIM

win = 8
isPCA = False
dataset = 'Kodak'
color='YUV'

root = '../'
myDataset = {'Kodak':{'path':'data/Kodak/', 'size':(512, 768)},
           'DIV2K':{'path':'data/DIV2K/', 'size':(512, 512)},
            'Duck':{'path':'data/Duck/', 'size':(1024, 1792)},
            'NB':{'path':'data/NormalizedBrodatz/', 'size':(640,640)}}

def T(X, Kernels=None, train=True, isPCA=True):
    X = Shrink(X, {'win':win})
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 3, 8, 8)
    if train == True:
        if isPCA == True:
            pca = PCA(n_components=X.shape[-1]).fit(X)
            Kernels = pca.components_
        else:
            tX = dct(X, axis=4, norm='ortho')
            tX = dct(tX, axis=5, norm='ortho')
            Kernels = np.dot(np.linalg.pinv(tX.reshape(-1,64)), X.reshape(-1,64))
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 3, 64)
    X = np.dot(X, np.transpose(Kernels))
    return X, Kernels

def iT(X, Kernels):
    X = np.dot(X, Kernels)
    X = invShrink(X, {'win':win})
    return X

def opt_K(dataset, X, tX, N):
    X = Shrink(X, {'win':win})
    X = X.reshape(-1, win*win)
    tX = tX.reshape(-1, tX.shape[-1])
    K = np.dot(np.linalg.pinv(tX), X)
    write_to_file(dataset+'_opt_inv@N='+str(N)+'.txt', np.transpose(K).astype('float32'))
    return np.transpose(K)

def main(dataset, Qf, ct, num_train=10):
    X = load_image(root+myDataset[dataset]['path'], color=color, count=ct, size=myDataset[dataset]['size'])

    print('Input shape: ', X.shape)
    tX, K = T(X-128, Kernels=None, train=True, isPCA=isPCA)
    #write_to_file('fore_dct.txt', K.astype('float32'))
    
    print('Transfrom shape: ', tX.shape)
    tX = dQ(Q(tX, Qf), Qf)
    print('raw')
    iX = iT(tX, Kernels=K)
    print('PSNR: ', PSNR(X[:num_train], iX[:num_train]+128)) 
    print('PSNR: ', PSNR(X[num_train:], iX[num_train:]+128))  

    print('opt')
    K_opt = opt_K(dataset, copy.deepcopy(X[:num_train])-128, copy.deepcopy(tX[:num_train]), Qf)
    iX = iT(tX, Kernels=np.transpose(K_opt))
    print('PSNR: ', PSNR(X[:num_train], iX[:num_train]+128)) 
    print('PSNR: ', PSNR(X[num_train:], iX[num_train:]+128)) 
    return K, K_opt
    
if __name__ == "__main__":
    for i in range(30, 100, 20):
        main('Kodak', Qf=i, ct=10, num_train=10)
    