"""
    Produces small version of Figure 3 from https://arxiv.org/pdf/2009.13977

    Differences.

    - To allow fast experiment we have repeats=10 and d<1024+1. To run full experiment change lines 25+26 and 32+33. 
      Without these changes the standard deviations are a bit larger. You will also need to change xlim in plotting code 'plot.py'. 

    - The plot in the article uses sequential/parallel algorithm from https://github.com/zhangjiong724/
      Their code is not documented at all, it took us a week to get the code running on our machine. 
      To allow our code to be easily run, we chose not to use their code in the public version of our experiment. 
      Instead, we use a PyTorch implementation of the sequential algorithm and do not include the parallel algorithm.  
      This may cause some differences in the timing of the sequential algorithm relative to the article. 

"""

import numpy as np
import torch
import sys
import os 

from run_svd import run_svd, run_seq
from run_exp import run_exp, run_cay

#repeats = 100 
repeats = 10
bs   	= 32 

data = np.zeros((48, 5, repeats)) 
print("| %-10s | %-10s | %-10s | %-10s | %-10s |"%("dimension", "FastH", "Exp", "Cayley", "Sequential"))

#for i, d in enumerate(range(64, 64*48+1, 64)): 
for i, d in enumerate(range(64, 512+1, 64)): 
	svd = run_svd(d, bs, repeats)
	exp = run_exp(d, bs, repeats)
	cay = run_cay(d, bs, repeats)
	if d <= 512: seq = run_seq(d, bs, repeats)

	data[i, 0, :] = svd
	data[i, 1, :] = exp
	data[i, 2, :] = cay
	if d <= 512: data[i, 3, :] = seq # this takes too long for d>512. 

	print("| %-10i | %-10f | %-10f | %-10f | %-10f |"%(d, data[i, 0, :].mean(), data[i, 1, :].mean(), data[i, 2, :].mean(), data[i, 3, :].mean()))
	np.savez("data", data)
	
import plot  
plot.plot()
