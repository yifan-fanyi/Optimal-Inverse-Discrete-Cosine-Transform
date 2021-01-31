# 2020.09.18
# finalized 2020.09.29
# @yifan
#
# plot and evl functions

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cv2
import os             
import copy
import pickle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from utli import *
from evaluate import PSNR, SSIM
from main import main

root = '../'
myDataset = {'Kodak':{'path':'data/Kodak/', 'size':(512, 768)},
           'DIV2K':{'path':'data/DIV2K/', 'size':(512, 512)},
            'Duck':{'path':'data/Duck/', 'size':(1024, 1792)},
            'NB':{'path':'data/NormalizedBrodatz/', 'size':(640,640)}}

# compute PSNR, SSIM in one folder
def cal(pathraw, pathres, start, end, name='tmp', write=False, QF_s=10, QF_e=100, QF_step=10, verbose=True):
    Nl, psnr, ssim = [], [], []
    for N in range(QF_s,QF_e,QF_step):
        ps, ss = [], []
        print('   Start @N='+str(N))
        for i in range(start, end):
            try:
                ref = cv2.imread(pathraw+str(i)+'.bmp')
                dec = cv2.imread(pathres+'decode/'+str(i)+'_'+str(N)+'.bmp')
                ps.append(PSNR(ref, dec))
                ss.append(SSIM(ref, dec))
                if verbose == True:
                    print('     <INFO> image '+str(i)+' @N='+str(N)+' PSNR='+str(ps[-1])+' SSIM='+str(ss[-1]))
            except:
                #print('     <WARNING> missing image '+str(i)+' @N='+str(N))
                continue
        psnr.append(np.mean(ps))
        ssim.append(np.mean(ss))
        Nl.append(N)
        print('<Summary> PSNR='+str(psnr[-1])+' SSIM='+str(ssim[-1])+' @N='+str(Nl[-1]))
    if write == True:
        with open('../fig/'+name+'.pkl', 'wb') as f:
            pickle.dump({'N':Nl, 'PSNR':psnr, 'SSIM':ssim}, f)
            print('<INFO> Write to file: '+'../fig/'+name+'.pkl')
    return Nl, psnr, ssim

def load(dataset='kodak'):
    with open('../fig/'+dataset+'.pkl', 'rb') as f:
        d = pickle.load(f)
    return np.array(d['N']), np.array(d['Nt']), np.array(d['PSNR']), np.array(d['PSNRt']), np.array(d['SSIM']), np.array(d['SSIMt'])

# opt K computed at Qf=k, images encoded at same Qf
# data[0] is for raw jpeg
def N_diag(N, data):
    N_r, data_r = [], []
    for i in range(1, N.shape[1]):
        N_r.append(N[i+1,i])
        data_r.append(data[i+1,i])
    return N_r, data_r

def jpeg(N, data):
    return N[0], data[0]

# plt opt K computed at Qf=k, images encoded at same Qf
def plt_NvN(p_s_n_r=True, train=True):
    N, Nt, psnr, psnrt, ssim, ssimt = load('kodak')
    if p_s_n_r == False:
        psnr = ssim
        psnrt = ssimt
    figure(num=None, figsize=(11, 8), dpi=200, facecolor='w', edgecolor='k')
    if train == True:
        x, y = jpeg(N, psnr)
        plt.plot(x[3:-1], y[3:-1], label='JPEG (Kodak train)', color='b', alpha=0.3)
        x, y = N_diag(N, psnr)
        plt.plot(x[1:-1], y[1:-1], label='(ours) (Kodak train)', color='r')
    else:
        x, y = jpeg(Nt, psnrt)
        plt.plot(x[2:], y[2:], label='JPEG', color='b', alpha=0.3)

        x1, y1 = N_diag(Nt, psnrt)
        plt.plot(x1[1:], y1[1:], label='Ours', color='r')
        print(x1, y1-y[1:])
    plt.xlabel('Quality Factor')
    if p_s_n_r == False:
        plt.ylabel('SSIM')
    else:
        plt.ylabel('PSNR (dB)')
    plt.legend(prop={'size': 12})
    plt.show()
    
    figure(num=None, figsize=(11, 8), dpi=200, facecolor='w', edgecolor='k')
    N, Nt, psnr, psnrt, ssim, ssimt = load('div2k')
    if p_s_n_r == False:
        psnr = ssim
        psnrt = ssimt
    if train == True:
        plt.plot(N[0][1:-1], psnr[0][1:-1], label='JPEG (DIV2K train)',color='m', alpha=0.3)
        plt.plot(N[1][1:-1], psnr[1][1:-1], label='(ours) (DIV2K train)',color='g')
    else:   
        plt.plot(N[0][1:], psnrt[0][1:], label='JPEG',color='g', alpha=0.3)
        plt.plot(N[1][1:], psnrt[1][1:], label='Ours',color='m')
        print(N[0][1:], psnrt[1][1:]-psnrt[0][1:])
    plt.xlabel('Quality Factor')
    if p_s_n_r == False:
        plt.ylabel('SSIM')
    else:
        plt.ylabel('PSNR (dB)')
    plt.legend(prop={'size': 12})
    plt.show()

# save images with PSNR larger than <th>
def plt_image_block(dataset, win, th=0.8, folder='kodak_opt_inv@N=50', s=10, num=23):
    for i in range(s, num):
        try:
            x = cv2.imread('../data/'+dataset+'/'+str(i)+'.bmp')
            y = cv2.imread('../result/'+dataset+'/'+folder+'/decode/'+str(i)+'_70.bmp')
            z = cv2.imread('../result/'+dataset+'/jpeg_raw/decode/'+str(i)+'_70.bmp')
            print(i, '----------------',PSNR(x,y) - PSNR(x,z))
            if abs(PSNR(x,z) - PSNR(x,y))>0.1:
                x = view_as_windows(x, (win,win,3), (win,win,3))
                x = x.reshape(-1, win, win, 3)
                y = view_as_windows(y, (win,win,3), (win,win,3)).reshape(-1, win, win, 3)
                z = view_as_windows(z, (win,win,3), (win,win,3)).reshape(-1, win, win, 3)
                for j in range(x.shape[0]):
                    c = PSNR(x[j],y[j]) - PSNR(x[j],z[j])
                    if  c > th:
                        print(c)
                        cv2.imwrite('../result/tmp/'+str(i)+'_'+str(j)+'_opt'+str(c)+'.png', y[j])
                        cv2.imwrite('../result/tmp/'+str(i)+'_'+str(j)+'_jpg.png', z[j])
                        cv2.imwrite('../result/tmp/'+str(i)+'_'+str(j)+'_raw.png', x[j])
        except:
            continue

# draw correlation matrix between kerenl computed at different Qf
def draw_dist_png():
    with open('../fig/corr.pkl', 'rb') as f:
        Z = pickle.load(f)
    fig = plt.figure(figsize=(16,8))
    ax = fig.gca(projection='3d')
    X, Y = [], []
    for i in range(5, 100, 10):
        X.append(i)
        Y.append(i)
    X, Y = np.meshgrid(np.array(X), np.array(Y))
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)


    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('Quality Factor')
    ax.set_ylabel('Quality Factor')
    ax.set_zlabel('L2 Distance')
    plt.show()

# one Qf for images encoded at all Qf
def plt_N_all(name='div2k_kodak@N=80', p_s_n_r=0):
    figure(num=None, figsize=(16, 12), dpi=200, facecolor='w', edgecolor='k')
    N = [50, 60, 70, 80, 90]
    N1 = [50, 70, 90] 
    psnr = [33.814829041420815, 34.72220158638222, 37.28080544370594]
    b = [33.88332856858257, 34.27231624600438, 34.850441813588006, 35.703367093531334, 37.336899920746575]
    c = [0.9100433641797571, 0.9296664839477011, 0.9615681240089557]
    cc = [0.9114377837974508, 0.9210319443138413, 0.9313069519843874, 0.9455445405903505, 0.9625424889039493]
    if p_s_n_r == False:
        plt.plot(N1, psnr, label='JPEG (DIV2K)',color='m', alpha=0.3)
        plt.plot(N, b, label='ours (DIV2K)',color='g')
    else:
        plt.plot(N1, c, label='JPEG (DIV2K)',color='m', alpha=0.3)
        plt.plot(N, cc, label='ours (DIV2K)',color='g')
    plt.xlabel('Quality Factor')
    if p_s_n_r == False:
        plt.ylabel('SSIM')
    else:
        plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.show()

def run(dataset, opt):
    Nl, psnr, ssim = cal('../data/'+dataset+'/', '../result/'+dataset+'/'+opt+'/', start=0, end=100, name=opt+'_train', write=0, QF_s=10, QF_e=91, QF_step=20)  
    Nlt, psnrt, ssimt = cal('../data/'+dataset+'/', '../result/'+dataset+'/'+opt+'/', start=100, end=785, name=opt+'_test', write=0, QF_s=90, QF_e=91, QF_step=20)  
    with open('../fig/'+dataset+'_'+opt+'tmp.pkl', 'wb') as f:
        pickle.dump({'N':0, 'PSNR':0, 'SSIM':0, 'Nt':Nlt, 'PSNRt':psnrt, 'SSIMt':ssimt}, f)
    return 0, 0, 0, Nlt, psnrt, ssimt

def multi_run(dataset = 'DIV2K'):
    Nl, psnr, ssim, Nlt, psnrt, ssimt = [], [], [], [], [], []
    a,b,c,d,e,f = run(dataset, opt='jpeg_raw')
    Nl.append(a)
    psnr.append(b)
    ssim.append(c)
    Nlt.append(d)
    psnrt.append(e)
    ssimt.append(f)
    for i in range(50, 51, 10):
        print(i,'----------------------------------')
        a,b,c,d,e,f = run(dataset, opt='div2k')
        Nl.append(a)
        psnr.append(b)
        ssim.append(c)
        Nlt.append(d)
        psnrt.append(e)
        ssimt.append(f)
    with open('../fig/div2k.pkl', 'wb') as f:
        pickle.dump({'N':Nl, 'PSNR':psnr, 'SSIM':ssim, 'Nt':Nlt, 'PSNRt':psnrt, 'SSIMt':ssimt}, f)

if __name__ == "__main__":
    #plt_NvN(p_s_n_r=0, train=0)
    #plt_NvN(p_s_n_r=0, train=1)
    #plt_NvN(p_s_n_r=1, train=0)
    #plt_NvN(p_s_n_r=1, train=1)

    #plt_N_all(p_s_n_r=0)
    #plt_N_all(p_s_n_r=1)

    #draw_corr('Kodak')

    #plt_image_block('DIV2K', 32, th=0.6, folder='@N=70', s=100, num=200)

    #a,b,c,d,e,f = run('DIV2K', opt='@N=50')
    a = 90
    cal(pathraw="/Users/alex/Desktop/proj/compression/data/Kodak/Kodak/",
     pathres="/Users/alex/Documents/GitHub/Optimal_Inverse/result/Kodak/jpeg_raw/",
      start=10, end=24, name='tmp', write=False, QF_s=a, QF_e=a+2, QF_step=10, verbose=0)
    cal(pathraw="/Users/alex/Desktop/proj/compression/data/Kodak/Kodak/",
     pathres="/Users/alex/Documents/GitHub/Optimal_Inverse/result/Kodak/@N="+str(a)+"/",
      start=10, end=24, name='tmp', write=False, QF_s=a, QF_e=a+2, QF_step=10, verbose=0)

