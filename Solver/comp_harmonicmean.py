import numpy as np

def comp_harmonicmean(Avg,Vec):
    
    invVec = 1.0/Vec  #elementwise inverse of vector for harmonic mean
    
    invVec[np.isneginf(invVec)] = 0.0
    invVec[np.isposinf(invVec)] = 0.0

    #print('Vector',Vec)
    #print('Inverted Vector',invVec)
    
    invVec = Avg @ invVec        # calculating algebraic mean of inverse
    HarVec = 1.0/invVec          #elementwise inverse of vector for harmonic mean 

    HarVec[np.isneginf(HarVec)] = 0.0
    HarVec[np.isposinf(HarVec)] = 0.0
    
    #print('Algebraic', Avg @ Vec)
    #print('Harmonic', HarVec)
    #print('Subtract',Avg @ Vec - HarVec)  
    #print('Norm',np.linalg.norm(Avg @ Vec - HarVec))
    return HarVec;