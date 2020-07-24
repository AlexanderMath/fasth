"""
	See 'README.md' for details. 

    Expected output from 'python test_case.py': 

        Compiling our c++/cuda code, this usually takes 1-2 min. 
        Finished compilation, took: 0.0701s

        ---------- Assertions ----------
        Multiplication:			PASSED!
        Inverse Multiplication: 	PASSED!
        Reconstruction: 		PASSED!

        ---------- Check Orthogonal Inverse ----------
        Multiplication:			PASSED!

        ---------- Time Algorithms ----------
        Printing time of both algorithms as [mean +- std]. 
        [100 / 100]	Seq: 	[0.066833 +- 0.002570] 	Ours: [0.001502 +- 0.000026] 	 
        Speed-up: 	0.066833 / 0.001502 = 44.50 times faster

        ---------- Time Neural Network ----------
        Taking time of a single gradient step for a
        Neural Network with orthogonal matrices of size 
        [512, 512, 512]. 
        Sequential: 	0.3743
        Our approach: 	0.0212
        Speed-up: 	17.6459 times faster


"""
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.cpp_extension import load
import time
from fasth_wrapper import *

torch.manual_seed(42)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Define problem size
d		    = 512
m		    = 32
batch_size  = 32


# Initialize problem. 
V	   = torch.zeros((d,d)).normal_(0, 1).cuda()
X	   = torch.zeros((d,batch_size)).normal_(0, 1).cuda()
norms  = normalize(V)

# 1. Demonstrate that both algorithms computes the same thing. 
print("\n---------- Assertions ----------")

# Compute multiplication with sequential algorithm. 
mult_seq	= sequential_mult	(V, X)

# Compute multiplication with our algorithm. 
Y		    = algo_compute_dec(V, m)
mult_algo   = algo_mult(V, X, Y, m)

# Check that both algorithms computed the same. 
assert torch.allclose(mult_seq, mult_algo, atol=10**(-5))
print("Multiplication:\t\t\tPASSED!")


# Compute inverse multiplication with sequential algorithm. 
imult_seq	= sequential_inv_mult(V, mult_seq)

# Compute inverse multiplication with our algorithm. 
imult_algo   = algo_inv_mult(V, mult_algo, Y, m)

# Check that both algorithms computed the same. 
assert torch.allclose(imult_seq, imult_algo, atol=10**(-5))
print("Inverse Multiplication: \tPASSED!")


# Multiplication and inverse multiplication is identity. 
# Check that both algorithms also succesfully computed identity
assert torch.allclose(imult_algo, X, atol=10**(-5))
assert torch.allclose(imult_seq,  X, atol=10**(-5))
print("Reconstruction: \t\tPASSED!")


# 1. Check invertibility of orthogonal matrix. 
print("\n---------- Check Orthogonal Inverse ----------")
orth = Orthogonal(d, 32, "fast")

X   = torch.zeros((d, 32)).normal_(0, 1)
enc = orth.forward(X)
rec = orth.inverse(enc)

assert torch.allclose(rec, X, atol=10**(-5))
assert not torch.allclose(enc, X, atol=10**(-5))
print("Multiplication:\t\t\tPASSED!")


# 2. Take time of both algorithms. 
print("\n---------- Time Algorithms ----------")
print("Printing time of both algorithms as [mean +- std]. ")
time_seq = []
time_our = [] 
for i in range(100): 

	# Initialize a new problem each time. 
	V	   = torch.zeros((d,d)).normal_(0, 1).cuda()
	X	   = torch.zeros((d,batch_size)).normal_(0, 1).cuda()
	norms   = normalize(V)
	torch.cuda.synchronize()

	t0 = time.time() 
	sequential_mult(V, X)
	sequential_inv_mult(V, X)
	torch.cuda.synchronize()
	t1 = time.time()

	Y		   = algo_compute_dec(V, m)
	mult_algo   = algo_mult(V, X, Y, m)
	mult_algo   = algo_inv_mult(V, X, Y, m)
	torch.cuda.synchronize()
	t2 = time.time()

	time_seq.append(t1-t0)
	time_our.append(t2-t1)

	print("\r[%i / %i]\tSeq: \t[%.6f +- %.6f] \tOurs: [%.6f +- %.6f] \t "%( 
			i+1, 100,
			np.mean(time_seq), np.std(time_seq), 
			np.mean(time_our), np.std(time_our)), 
			end="", flush=True)

print("")
print("Speed-up: \t%.6f / %.6f = %.2f times faster"%( 
			np.mean(time_seq) , np.mean(time_our), 
			np.mean(time_seq)/np.mean(time_our)))


# 3. Compare FastH in a Neural Network. 
print("\n---------- Time Neural Network ----------")
print("Taking time of a single gradient step for a")
print("Neural Network with orthogonal matrices of size ")
print("[%i, %i, %i]. "%(d, d, d))
seq_net = OrthNet(d, 32, "sequential")
our_net = OrthNet(d, 32, "fast")


X = torch.zeros((d, 32)).normal_(0, 1)
Y = torch.zeros((d, 32)).normal_(0, 1)

# warmup round. 
pred        = seq_net(X)
error       = torch.mean( (pred - Y)**2)
error.backward()
torch.cuda.synchronize()


t0          = time.time()
pred        = seq_net(X)
error       = torch.mean( (pred - Y)**2)
error.backward()
torch.cuda.synchronize()
t1          = time.time()
time_seq    = t1-t0
print("Sequential: \t%.4f"%(t1-t0))

t0          = time.time()
pred        = our_net(X)
error       = torch.mean( (pred - Y)**2)
error.backward()
torch.cuda.synchronize()
t1          = time.time()
time_ours   = t1-t0

print("Our approach: \t%.4f"%(t1-t0))
print("Speed-up: \t%.4f times faster"%(time_seq/time_ours))


