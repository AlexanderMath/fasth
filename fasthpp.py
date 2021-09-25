import torch 
import time 
import numpy as np 
torch.manual_seed(42)

# Define problem 
d  = 64
bs = 8
V  = torch.zeros((d, d)).normal_(0, 1)

# Normalize V so we don't need to divide by norm. 
def normalize(V):   
	d	  = V.shape[0]
	norms   = torch.norm(V, 2, dim=1)
	V[:,:]  = V / norms.view(d, 1)
	return norms
normalize(V.T)

# Naive approach used for test case. 
I = torch.eye(d)
def H(v): 
  v = v.reshape(d,1)
  return I - 2 * v @ v.T #/ (v.T @ v)
def Q(V):  # O(d^4)
  M = I 
  for i in range(d):
    M = M @ H(V[:, i:i+1])
  return M

# New algorithm with O(d/t + log2(t)) operations. 
def fasthpp(V, X, stop_recursion=3): 
  """
    V: matrix that represent weights of householder matrices (d, d)
    X: rectangular matrix (d, bs) to compute H(V) @ X
    stop_recursion: integer that controls how many merge iterations before recursion stops. 
    		    if None recursion continues until base case. 
  """
  d = V.shape[0]

  Y_ = V.clone().T
  W_ = -2*Y_.clone()

  # Only works for powers of two. 
  assert (d & (d-1)) == 0 and d != 0, "d should be power of two. You can just pad the matrix. " 

  # Step 1: compute (Y, W)s by merging! 
  k = 1
  for i, c in enumerate(range(int(np.log2(d)))):  
    k_2 = k 
    k  *= 2

    m1_ = Y_.view(d//k_2, k_2, d)[0::2] @ torch.transpose(W_.view(d//k_2, k_2, d)[1::2], 1, 2)
    m2_ = torch.transpose(W_.view(d//k_2, k_2, d)[0::2], 1, 2) @ m1_

    W_ = W_.view(d//k_2, k_2, d)
    W_[1::2] += torch.transpose(m2_, 1, 2)
    W_ = W_.view(d, d)

    if stop_recursion is not None and c == stop_recursion: break

  # Step 2: 
  if stop_recursion is None:   return X + W_.T @ (Y_ @ X) 
  else: 
    # For each (W,Y) pair multiply with 
    for i in range(d // k-1, -1, -1 ):
      X = X + W_[i*k: (i+1)*k].T @ (Y_[i*k: (i+1)*k]  @ X )
    return X 


# Test for (d, bs). 
X     = torch.zeros((d, bs)).normal_(0, 1)
prod  = Q(V) @ X  # naive approach 
prod_ = fasthpp(V, X, stop_recursion=4)
print( "Max absolute error %f for X.shape=%s"%( (prod-prod_).abs().max(), str(X.shape)))
assert torch.allclose(prod, prod_, atol=10**(-5))

# Test for (d, d, bs)
X     = torch.zeros((d, d, bs)).normal_(0, 1)
prod  = Q(V) @ X  # naive approach 
prod_ = fasthpp(V, X, stop_recursion=4) 
print( "Max absolute error %f for X.shape=%s"%( (prod-prod_).abs().max(), str(X.shape)))
assert torch.allclose(prod, prod_, atol=10**(-5))

print("Passed!")

