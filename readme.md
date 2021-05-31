# A Machine Learning Approach to Optimal Inverse Discrete Cosine Transform (IDCT) Design

#### &emsp; @Yifan Wang
##### &emsp; yifanwang0916@outlook.com 
##### &emsp; September, 2020 
##### &emsp; Paper Available at: https://arxiv.org/abs/2102.00502

***
### `Method`:
&emsp; Estimate the quantization error in lossy compression and find an optimal inverse kernel using linear regression rather than original IDCT kernel.
### `Result`:
&emsp; Our learnt optimal inverse kernels have a low dependency on training images. The kernel trained at certain quality factor `N` can be used to decode images encoded by a quality factor near to it. It can achieve `0.1-0.2dB` PSNR increment on DIV2K dataset @N=50 and more than `0.3dB` PSNR increment on Normalized Brodatz textures from MBT dataset @N=90. 

&emsp; More details can be found in our paper.

***
### `Usage`:
&emsp; Use `./src/main.py` to train the optimal inverse kernel on training images. Then copy the kernel value in `./kernel/` folder to `inv_K` in `./src/jpeg-6b-our/jidctflt.c`. 

&emsp; Use `./src/run.sh` to encode and decode all provided images. 

&emsp; `./src/plot.py` is used to run the evaluation result and plot the figures used in this paper. 

#####   `Code provided are not optimized. `



