import torch
import time
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load
import numpy as np 

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.manual_seed(42)


@torch.jit.script
def sequential_mult(V, X):  
	for row in range(V.shape[0]-1, -1, -1): 
		X =  X - 2 * V[row:row+1, :].t() @ (V[row:row+1, :] @ X)
	return X

# ------ Load c++ and cuda code.  ---------
t0 = time.time()
print("Compiling our c++/cuda code, this usually takes 1-2 min. ")
algo = load(name="fasth", sources=["../fasth.cpp", "../fasth_cuda.cu"])
print("Finished compilation, took: %.4fs"%(time.time()-t0))

def algo_compute_dec(V, m):
	d = V.shape[0]
	assert d % m == 0, "The CUDA implementation assumes m=%i divides d=%i which, for current parameters, is not true.  "%(d, m)
	
	Y = torch.clone(V) 
	algo.compute_dec(V, Y, m)
	return Y 

def algo_backwards(V, Y, output, grad_output, norms, m):
	d = V.shape[0]
	assert d % m == 0, "The CUDA implementation assumes m=%i divides d=%i which, for current parameters, is not true.  "%(d, m)

	gradV = torch.zeros_like(V)
	algo.backward(V, gradV, Y, output, grad_output, norms, m)
	return grad_output, gradV

def algo_mult(V, X, Y, m):
	d = V.shape[0]
	assert d % m == 0, "The CUDA implementation assumes m=%i divides d=%i which, for current parameters, is not true.  "%(d, m)

	result = X.clone()
	algo.mult(V, result, Y, m)
	return result

def algo_inv_mult(V, X, Y, m):  
	d = V.shape[0]
	assert d % m == 0, "The CUDA implementation assumes m=%i divides d=%i which, for current parameters, is not true.  "%(d, m)

	algo.inv_mult(V, X, Y, m)
	return X
# -----------------------------------------

# Helper function 
def normalize(V):   
	d	  = V.shape[0]
	norms   = torch.norm(V, 2, dim=1)
	V[:,:]  = V / norms.view(d, 1)
	return norms


def experiment_seq(d, bs, G): 
	V 		= torch.zeros((d, d)). normal_(-1, 1) 
	X 		= torch.zeros((d, bs)).normal_(0, 1)
	V.requires_grad_(True)
	torch.cuda.synchronize()

	# Start timing of forward and backwards pass. 
	t0 = time.time()
	Y = sequential_mult(V, X)
	torch.autograd.backward( Y, G )
	torch.cuda.synchronize()
	return time.time() - t0

def run_seq(d, bs, repeats): 
	times = [] 
	G = torch.zeros((d, bs)).normal_(0, 1)  

	for i in range(repeats + 1):
		t = experiment_seq(d, bs, G)
		if i > 0: times.append(t)

	return np.array(times)


def exp(d, m, bs, G) : 
	V 		= torch.zeros((d, d)). normal_(0, 1)
	X 		= torch.zeros((d, bs)).normal_(0, 1)
	Bs 		= torch.zeros((d // m, d, bs))
	torch.cuda.synchronize()

	# Start timing of forward and backwards pass. 
	t0 = time.time()

	# Includes the time of normalization and cloning. 
	# Time difference is 0.0131 to 0.0146. 
	# Could be done faster if cloning / normalization was handled in CUDA code. 
	W 			= V.clone() 
	output 		= X.clone()
	norms 		= normalize(V)  
	Y = algo_compute_dec(V, m)
	algo_inv_mult(V, X, Y, m)
	algo_backwards(V, W, output, G, norms, m)

	return time.time() - t0


def run_svd(d, bs, repeats, m=None): 

	times = []

	if m is None: 
		if d <= 128: 	m = 16 
		else:		 	m = 32

	G = torch.zeros((d, bs)).normal_(0, 1) 

	for i in range(repeats + 1):
		t = exp(d, m, bs, G)

		if i > 0: times.append(t)

	return np.array(times) 


if __name__ == "__main__": 

	for m in [4, 8, 16, 32]:
		print(np.sum(run_svd(128, 32, 10, 32), 0))
