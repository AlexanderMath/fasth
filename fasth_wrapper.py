"""
	See 'README.md' for details. 
"""
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.cpp_extension import load
import time

torch.manual_seed(42)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Helper function 
def normalize(V):   
	d	  = V.shape[0]
	norms   = torch.norm(V, 2, dim=1)
	V[:,:]  = V / norms.view(d, 1)
	return norms

# ---------- Naive implementation ----------
@torch.jit.script
def sequential_mult(V, X):  
	for row in range(V.shape[0]-1, -1, -1): 
		X =  X - 2 * V[row:row+1, :].t() @ (V[row:row+1, :] @ X)
	return X

@torch.jit.script
def sequential_inv_mult(V, X):	
	for row in range(V.shape[0]): 
		X =  X - 2 * V[row:row+1, :].t() @ (V[row:row+1, :] @ X)
	return X
# -----------------------------------------


# ------ Load c++ and cuda code.  ---------
t0 = time.time()
print("Compiling our c++/cuda code, this usually takes 1-2 min. ")
algo = load(name="fasth", sources=["fasth.cpp", "fasth_cuda.cu"])
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


# ---------- PyTorch wrapper --------------
class HouseProd(torch.autograd.Function):

	m = 28 

	@staticmethod
	def forward(ctx, input, V):
		input = input.clone()

		V				   = V.clone()
		ctx.norms		   = normalize(V) 

		ctx.Y = algo_compute_dec(V, HouseProd.m)
		ctx.V = V
		algo_inv_mult	   (V, input, ctx.Y, HouseProd.m)

		ctx.output = input.clone()

		return input

	@staticmethod
	def backward(ctx, grad_output):
		V			   = ctx.V
		Y			   = ctx.Y
		output		  = ctx.output
		norms		   = ctx.norms

		return algo_backwards(V, Y, output, grad_output, norms, HouseProd.m)


class Orthogonal(nn.Module):

	def __init__(self, d, m=28, strategy = "fast"): 
		super(Orthogonal, self).__init__()
		self.d		  = d

		if strategy == "fast": assert d % m == 0, "The CUDA implementation assumes m=%i divides d=%i which, for current parameters, is not true.  "%(d, m)
	
		if not strategy in ["fast", "sequential"]: 
			raise NotImplementedError("The only implemented strategies are 'fast' and 'sequential'. ")

		self.strategy = strategy

		self.U = torch.nn.Parameter(torch.zeros((d, d)).normal_(0, 0.05))

		HouseProd.m = m

	def forward(self, X):

		if self.strategy == "fast": 
			X = HouseProd.apply(X, self.U)
		elif self.strategy == "sequential": 
			X = sequential_mult(self.U, X)
		else: raise NotImplementedError("The only implemented strategies are 'fast' and 'sequential'. ")

		return X

	def inverse(self, X): 
		if self.strategy == "fast": 
			X = HouseProd.apply(X, torch.flip(self.U, dims=[0]))
		elif self.strategy == "sequential": 
			X = sequential_mult(troch.flip(self.U, dims=[0]), X)
		else: raise NotImplementedError("The only implemented strategies are 'fast' and 'sequential'. ")

		return X

	def lgdet(self, X): 	return 0


# -----------------------------------------
class OrthNet(torch.nn.Module): 

	def __init__(self, d, m=32, strategy="fast"): 
		super(OrthNet, self).__init__()
		self.d		  = d

		if not strategy in ["fast", "sequential"]: 
			raise NotImplementedError("The only implemented strategies are 'fast' and 'sequential'. ")

		self.strategy = strategy

		self.o1 = Orthogonal(d, m, strategy)
		self.o2 = Orthogonal(d, m, strategy)
		self.o3 = Orthogonal(d, m, strategy)

	def forward(self, X):
		X = self.o1(X)
		X = torch.nn.functional.relu(X)
		X = self.o2(X)
		X = torch.nn.functional.relu(X)
		X = self.o3(X)
		return X 

# -----------------------------------------

class LinearSVD(torch.nn.Module): 
	def __init__(self, d, m=32): 
		super(LinearSVD, self).__init__()
		self.d		  = d

		self.U = Orthogonal(d, m)
		self.D = torch.empty(d, 1).uniform_(0.99, 1.01)
		self.V = Orthogonal(d, m)

	def forward(self, X):
		X = self.U(X)
		X = self.D * X 
		X = self.V(X)
		return X 

# -----------------------------------------


