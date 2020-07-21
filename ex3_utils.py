import sys
from typing import List

import numpy as np
from numpy import linalg as LA

import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

def isgray(img):
    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False
 
def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
  
    #convert to grayscale if it is RGB 

    if (not isgray(im1)):
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if (not isgray(im2)):
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
 
    # derivate
 
    Ix,Iy = np.gradient(im2)

    # calculate It 

    It = im2 - im1

    # u,v to restore the optical flow
    uv = []
    pts = []

    w = int (win_size)
    
    for i in range(w, im1.shape[0]-w,step_size):
        for j in range(w, im1.shape[1]-w,step_size):
            ix = Ix[i-w:i+w+1, j-w:j+w+1].ravel()
            iy = Iy[i-w:i+w+1, j-w:j+w+1].ravel()
            it = It[i-w:i+w+1, j-w:j+w+1].ravel()
            B = -1*it.reshape(it.shape[0],1)
            A = np.full((len(ix),2),0)
            A[:,0] = ix
            A[:,1] = iy
            if(i==w and j==w):print("A",A)

            # check if A^T*A can be inverted 
            At= np.transpose(A)
            AtA =  At.dot(A)
            if(i==w and j==w):print("AtA",AtA)
            eigenvalues, _ = LA.eig(AtA)
            eigenvalues = np.sort(eigenvalues)
            lambda1 = eigenvalues[-1]
            lambda2 = eigenvalues[-2]
            
            if(lambda2>1 and ((lambda1/lambda2)<100)):
                Atb= At.dot(B)
                ATAreverse = LA.inv(AtA)
                delta = ATAreverse.dot(Atb)
                pts.append([j,i])   
                uv.append(delta)
                
    uv = np.asarray(uv) 
    pts = np.asarray(pts)    

    print("uv mean",np.mean(uv[0]))
    print("uv mean",np.mean(uv[1]))
   

    return pts,uv;       
                
           
    
    


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    
    gaussList = gaussianPyr(img,levels)
    laplaceList = []

    kernel = np.array([[ 1.0278445 ,  4.10018648,  6.49510362,  4.10018648,  1.0278445 ],
       [ 4.10018648, 16.35610171, 25.90969361, 16.35610171,  4.10018648],
       [ 6.49510362, 25.90969361, 41.0435344 , 25.90969361,  6.49510362],
       [ 4.10018648, 16.35610171, 25.90969361, 16.35610171,  4.10018648],
       [ 1.0278445 ,  4.10018648,  6.49510362,  4.10018648,  1.0278445 ]])   
    for i in range(len(gaussList)-1):
        smaller =  gaussList[i+1]   
        exp = gaussExpand(smaller,kernel)
        a = gaussList[i].shape ==exp.shape  
        if (not a): exp = exp[:-1, :-1]
        newLevel = gaussList[i] - exp
        laplaceList.append(newLevel)
        # plt.imshow(newLevel,"gray")
        # plt.show()
        
    laplaceList.append(gaussList[-1])
    return laplaceList
    

def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
   
    kernel = np.array([[ 1.0278445 ,  4.10018648,  6.49510362,  4.10018648,  1.0278445 ],
       [ 4.10018648, 16.35610171, 25.90969361, 16.35610171,  4.10018648],
       [ 6.49510362, 25.90969361, 41.0435344 , 25.90969361,  6.49510362],
       [ 4.10018648, 16.35610171, 25.90969361, 16.35610171,  4.10018648],
       [ 1.0278445 ,  4.10018648,  6.49510362,  4.10018648,  1.0278445 ]])    

    laplaceList = lap_pyr[::-1] 

    result = laplaceList[0]

    size = len(laplaceList)

    for i in range (size-1):
        exp = gaussExpand(result,kernel)
        a = laplaceList[i+1].shape ==exp.shape  
        if (not a): exp = exp[:-1, :-1]

        result =  laplaceList[i+1] + exp

    return result


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    Listofimages = []
    Listofimages.append(img)
    kernel=cv2.getGaussianKernel(5,sigma = 1.1)
    for i in range(1, levels):
        newlevel = cv2.filter2D(Listofimages[-1], -1, kernel)  
        newlevel = newlevel[:: 2 , :: 2]
        Listofimages.append(newlevel)
    return Listofimages
    


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    if (len(img.shape)==2):
        w = img.shape[0]
        h = img.shape[1]
        newLevel = np.full((2*w,2*h),0,dtype=img.dtype) 
        newLevel = newLevel.astype(np.float)
        newLevel[::2,::2] = img       
    if (len(img.shape)==3): 
        w,h,z = img.shape
        newLevel = np.full((2*w,2*h,z),0,dtype=img.dtype)
        newLevel = newLevel.astype(np.float)
        newLevel[::2,::2] = img
   
    gs_k = (gs_k*4)/gs_k.sum() #make sure it is 4
    newLevel =cv2.filter2D(newLevel, -1,gs_k,borderType=cv2.BORDER_DEFAULT)
        

    return newLevel



def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
   
    naiveBlend = img_1*mask +img_2*(1-mask)

    maskList = gaussianPyr(mask, levels)
    firstImgList = laplaceianReduce(img_1, levels)
    SecondImgList = laplaceianReduce(img_2, levels)

    Ls = []

    for i in range (levels):

        Ls.append(firstImgList[i]*maskList[i]+ SecondImgList[i]*(1-maskList[i])) 

    return naiveBlend,laplaceianExpand(Ls)



    