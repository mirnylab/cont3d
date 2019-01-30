import numpy as np 
import pandas as pd 
import cooltools.lib.numutils


def leave_triu(mat):
    """Leave only upper triangular elements. 
    Upper triangular are defined as x[i,j,k]
    where i < j < k
    
    Strict less is because we want to leave only 3-point interactions
    (one may argue that we may want to leave i <= j < k and i < j <=k too
    But I chose to be careful and leave stricty i<j<k)"""
    M = mat.shape[0]
    ar = np.arange(M)
    triu_mask = (ar[None,None,:] > ar[None,:,None]) * (ar[None,:,None] > ar[:,None,None])  # who knew this is possible
    matc = np.array(mat)
    matc[~triu_mask] = 0 
    return matc



def add_transpose(mat):
    """
    Reverse of the leave_triu operation: add all 6 transposes to the matrix
    """
    perms = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
    return sum([np.transpose(mat, axes=i) for i in perms])



def flatten(mat3d):
    """
    Projects the 3D matrix to be a 2d matrix 
    """
    return np.sum(mat3d, axis=0) + np.sum(mat3d, axis=1) + np.sum(mat3d, axis=2)
    
def balance_simple(mat3d, iters=10):
    """
    Basic iterative iterative correction 
    Flattens the matrix, does IC, applies IC weights to 3D matrix 
    Rinses and repeats, because it will overcorrect as with ICE. 
    
    """
    mmean = mat3d.mean()
    for _ in range(iters):
        mat2d = flatten(mat3d)
        corrected, bias,_ =  cooltools.lib.numutils.iterative_correction_symmetric(mat2d)        
        mat3d = mat3d / (bias[:,None,None] * bias[None,:,None] * bias[None,None,:])
    mat3d = mat3d * (mmean / mat3d.mean())
    report = {}
    return mat3d, bias, report

def generateNorm(mat2d):
    a = mat2d[:,:,None] * mat2d[None,:,:]
    b = leave_triu(a)
    c = add_transpose(b)
    return c 


