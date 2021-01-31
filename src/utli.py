# 2020.09.07
# finalized 2020.09.26
# @yifan
#
# use DCT as foreword kernel
# use Linear Regression find the optimal inverse kernel
# utli functions

import numpy as np 
import pickle
import cv2
import os
from skimage.util import view_as_windows
from scipy.fftpack import dct, idct

def to_bmp(src, dist):
    name = os.listdir(src)
    ct =0
    name.sort()
    for n in name:
        try:
            x = cv2.imread(src+n)
            cv2.imwrite(dist+str(ct)+'.bmp', x)
            ct+=1
        except:
            print(" Missing file", n)
            continue

def load_by_channel(path, color='R', count=10, size=None):
    name = os.listdir(path)
    name.sort()
    raw, ct = [], 0
    if color == 'Y' or color == 'B':
        channel = 0
    if color == 'U' or color == 'G':
        channel = 1
    if color == 'V' or color == 'R':
        channel = 2
    for n in name:
        x = cv2.imread(path+n)
        try:
            x.shape
        except:
            continue
        print(path+n)
        if color == 'Y' or color == 'U' or color == 'V':
            x = cv2.cvtColor(x, cv2.COLOR_BGR2YCR_CB)
        x = x[:, :, channel]
        if x.shape[0] > x.shape[1]:
            x = np.transpose(x)
        x = cv2.resize(x, (size[1], size[0]))
        raw.append(x.reshape(x.shape[0], x.shape[1], 1))
        ct+=1
        if ct == count:
            break
    return np.array(raw)

def load_image(path, count=10, color='BGR', size=None):
    name = os.listdir(path)
    name.sort()
    raw, ct = [], 0
    for n in name:
        x = cv2.imread(path+n)
        try:
            x.shape
        except:
            continue
        if color == 'YUV':
            x = cv2.cvtColor(x, cv2.COLOR_BGR2YUV)
        if x.shape[0] > x.shape[1]:
            r = np.transpose(x[:,:,0]).reshape(x.shape[1],x.shape[0],1)
            g = np.transpose(x[:,:,1]).reshape(x.shape[1],x.shape[0],1)
            b = np.transpose(x[:,:,2]).reshape(x.shape[1],x.shape[0],1)
            x = np.concatenate((r,g,b), axis=-1)
        
        raw.append(x)#cv2.resize(x, (size[1], size[0])))
        ct+=1
        if ct == count:
            break
    return np.array(raw)

def Shrink(X, shrinkArg):
    win = shrinkArg['win']
    X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

def invShrink(X, invshrinkArg):
    win = invshrinkArg['win']
    S = X.shape
    X = X.reshape(S[0], S[1], S[2], -1, 1, win, win, 1)
    X = np.moveaxis(X, 5, 2)
    X = np.moveaxis(X, 6, 4)
    return X.reshape(S[0], win*S[1], win*S[2], -1)

def JPEG_Quant(X, N=50, deQ=False):
    JPEG_Q = np.array([16, 11, 10, 16, 24, 40, 51, 61,
                      12, 12, 14, 19, 26, 58, 60, 55,
                      14, 13, 16, 24, 40, 57, 69, 56,
                      14, 17, 22, 29, 51, 87, 80, 62,
                      18, 22, 37, 56, 68, 109, 103, 77,
                      24, 35, 55, 64, 81, 104, 113, 92,
                      49, 64, 78, 87, 103, 121, 120, 101,
                      72, 92, 95, 98, 112, 100, 103, 99], dtype='float64')
    if N > 50:   
        newQ = (100. - N) / 50. * JPEG_Q.copy()
    elif N < 50:
        newQ = 50. / N * JPEG_Q.copy()
    else:
        newQ = JPEG_Q.copy()
    if deQ == False:
        X /= newQ.reshape(64)
    else:
        X *= newQ.reshape(64)
    return np.round(X)

def Q(X, N=50, jpg=True):
    if jpg == True:
        return JPEG_Quant(X, N=N, deQ=False)
    return np.round(X/N)

def dQ(X, N, jpg=True):
    if jpg == True:
        return JPEG_Quant(X, N=N, deQ=True)
    return X*N

def write_to_file(name, data):
    with open('../kernel/'+name, 'w') as f:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                f.write(str(data[i,j]))
                f.write(', ')








