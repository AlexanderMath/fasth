"""
	Handles experiments with matrix exponential and Cayley map. 

"""
import torch
import time
import matplotlib.pyplot as plt
from expm32 import expm32, differential 
from trivializations import cayley_map
import numpy as np 

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.manual_seed(42)

# -------------------------------------------------------------------- #
def experiment_exp(d, bs, G, func): 
	V 		= torch.zeros((d,  d)). normal_(0, 1) 
	X 		= torch.zeros((bs, d)).normal_(0, 1)

	V 		= V - V.t()  # make skew symmetric. 
	V.requires_grad_(True)
	torch.cuda.synchronize()

	# Start timing of forward and backwards pass. 
	t0 = time.time()
	y = X @ func(V) 
	torch.autograd.backward( y, G )
	torch.cuda.synchronize()
	
	return time.time() - t0

def _run(d, bs, repeats, func): 
	times = [] 

	G = torch.zeros((bs, d)).normal_(0, 1)  

	for i in range(repeats + 1):
		t = experiment_exp(d, bs, G, func)

		if i > 0: times.append(t)

	return np.array(times)
 # -------------------------------------------------------------------- #
def run_cay(d, bs, repeats): return _run(d, bs, repeats, cayley_map)
def run_exp(d, bs, repeats): return _run(d, bs, repeats, expm32)
# -------------------------------------------------------------------- #

if __name__ == "__main__": 
	run_exp(1024*4, 32, 10)
	run_cay(1024*4, 32, 10)
