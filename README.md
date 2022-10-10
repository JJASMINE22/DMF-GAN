## Deep MultiScale Fusion GAN --PyTorch 
---

## 目录
1. [所需环境 Environment](#所需环境)
2. [模型结构 Structure](#模型结构)
3. [注意事项 Cautions](#注意事项)
4. [训练步骤 How2train](#训练步骤)
5. [生成效果 generate](#生成效果)
6. [参考资料 Reference](#参考资料)

## 所需环境
1. Python3.7
2. Pytorch>=1.10.1+cu113  
3. Torchvision>=0.11.2+cu113
4. Numpy==1.19.5
5. Pillow==8.2.0
6. CUDA 11.0+
7. Cudnn 8.0.4+

## 模型结构  
大体结构如下：  
![image](https://github.com/JJASMINE22/DMF-GAN/blob/main/samples/model.png)  

## 注意事项  
** 该网络使用birds数据集，并结合文本描述，各批量中的文本长度可能不同(由最长的决定) **  
1. 使用bool mask替代pack_padded、pad_packed等排除文本掩码影响的方法,
加入MultiHeadAttn，增强文本特征
2.  加入BN层，防止特征偏移；更新仿射层的归一化方法
3.  文本中加入部分结束符，特殊字符
3.  在DFGAN的基础上，生成大、小双尺度图像特征，且分步优化模型 
4.  加入权重正则化操作，防止过拟合
5.  不激活输出，并使用Hinge Loss解决边界像素值不收敛的问题

## 训练步骤
运行train.py即可开始训练。  

## 生成效果  
训练次数不够  
sample1  
![image](https://github.com/JJASMINE22/DMF-GAN/blob/main/samples/latter/sample1.jpg)  

sample2  
![image](https://github.com/JJASMINE22/DMF-GAN/blob/main/samples/latter/sample2.jpg)  

## 参考资料  
https://arxiv.org/abs/1911.11897  
https://arxiv.org/abs/2008.05865  
