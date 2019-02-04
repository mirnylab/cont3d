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
    triu_mask = (ar[None,None,:] > ar[None,:,None]) * (ar[None,:,None] > ar[:,None,None])  # who knew this is even possible
    matc = np.array(mat)  # copy
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
    Basic iterative iterative correction  (IIC) 
    Of course we had to invent it specifically for the 3C data. 
    
    Flattens the matrix, does IC, applies IC weights to 3D matrix 
    Rinses and repeats, because it will overcorrect at first as with ICE. 
    
    """
    mmean = mat3d.mean()
    for _ in range(iters):
        mat2d = flatten(mat3d)
        corrected, bias,_ =  cooltools.lib.numutils.iterative_correction_symmetric(mat2d)        
        mat3d = mat3d / (bias[:,None,None] * bias[None,:,None] * bias[None,None,:])
    mat3d = mat3d * (mmean / mat3d.mean())  # restore the same mean value 
    report = {}
    return mat3d, bias, report

"""

-------------------- Working on norms for the 3-point Hi-C data------- 

What we measure by 3-point Hi-C is P(A,B,C) - probability that 3 points A,B,C are all in contact 

What we need to do is to construct a "prior" for the 3-point Hi-C from the 2-point Hi-C. 

We know that it has to have the units of the second power of 2-point Hi-C. There is an argument for that in terms of units:
contact probability with the cutoff radius of R is proportional to R^3. Contact probability for 3 points 
would be proportional to the R^6: if we have a sphere around point A of radius R, point B would be there with probability ~R^3, and 
point C would be there with the same porobability, so both of them would be there with probability ~R^6. 

We also think it should be symmetric in A,B,C, but can it? We consider one nonsymmetric measure: normConsecutive
It takes into account that if we have 3 points along the genome: A,B,C, then P(A,B) is independent from P(B,C) 
While P(A,B) and P(A,C) are dependent: if B is very close to C, then P(A,B) is correlated with P(A,C).
Indeed, if A and B are in contact, then A and B+1 are very likely in contact.  
This dependence is actually what is causing a lot of trouble. 

We also have other norms. We attempted a brute-force norm of (P(A,B) * P(B,C) * P(A,C)) ^ (2/3). It is like 
we would want to multiply all the probabilities together, but we end up with the wrong power then, so let's just 
take it to the power of (2/3) and restore the needed second power of P(X,Y). 

We also considered norms that rely on multiplying two probabilities together, and are symmetric. 
There are two of them now. One is P(A,B) * P(B,C) + P(A,B) * P(A,C) + P(B,C) * P(A,C) 
Another is similar, but takes the max of 3, not the sum. 

"""


def normConsecutive(mat2d):
    """
    A norm inspired by the fact that 
    """
    a = mat2d[:,:,None] * mat2d[None,:,:]
    b = leave_triu(a)
    c = add_transpose(b)
    return c 
def normTriple(mat2d):
    a = (mat2d[:,:,None] * mat2d[None,:,:] * mat2d[:,None,:]) ** (2/3)
    b = leave_triu(a)
    c = add_transpose(b)
    return c 

def normPairsSum(mat2d):
    a = ( mat2d[None,:,:] * mat2d[:,None,:] + mat2d[:,:,None] * mat2d[:,None,:]+mat2d[:,:,None] * mat2d[None,:,:] ) 
    b = leave_triu(a)
    c = add_transpose(b)
    return c 
def normPairsMax(mat2d):
    a = np.maximum( mat2d[None,:,:] * mat2d[:,None,:],  mat2d[:,:,None] * mat2d[:,None,:],mat2d[:,:,None] * mat2d[None,:,:] ) 
    b = leave_triu(a)
    c = add_transpose(b)
    return c 

normDict = {"AB_BC":normConsecutive, "AB_BC_AC_23":normTriple,
         "ABBC_ABAC_BCAC_sum":normPairsSum, "ABBC_ABAC_BCAC_max":normPairsMax}    


