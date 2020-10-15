# from https://github.com/Lezcano/expRNN/blob/master/trivializations.py
import torch

from expm32 import expm32, differential

def cayley_map(X):
	n = X.size(0)
	Id = torch.eye(n, dtype=X.dtype, device=X.device)
	return torch.solve(Id - X, Id + X)[0]

class expm_class(torch.autograd.Function):
	@staticmethod
	def forward(ctx, A):
		ctx.save_for_backward(A)
		return expm32(A)

	@staticmethod
	def backward(ctx, G):
		(A,) = ctx.saved_tensors

		#print(A.t().shape, G.shape)
		
		y = differential(expm32, A.t(), G)

		return y

expm = expm_class.apply
