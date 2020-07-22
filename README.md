# FastH
Code accompanying article <a href="https://invertibleworkshop.github.io/accepted_papers/pdfs/10.pdf">Fast Orthogonal Parameterization with Householder Matrices</a> accepted for publication at the ICML 2020 Workshop on <a target="_blank" href="https://invertibleworkshop.github.io/">Invertible Neural Networks and Normalizing Flows</a>. 

<img src="plot.png" width="800px" height="200px" />

# Requirements 
Run 
```
pip install -r requirements.txt
```
Check installation by running test cases. 
```
python test_case.py
```

See <a target="_blank" href="test_case.py">test_case.py</a> for expected output.


# Minimal Working Example 

```
import torch
from fasth import Orthogonal 

class OrthNet(torch.nn.Module): 

	def __init__(self, d, m=32): 
		super(OrthNet, self).__init__()
		self.d		  = d

		self.o1 = Orthogonal(d, m)
		self.o2 = Orthogonal(d, m)
		self.o3 = Orthogonal(d, m)

	def forward(self, X):
		X = self.o1(X)
		X = torch.nn.functional.relu(X)
		X = self.o2(X)
		X = torch.nn.functional.relu(X)
		X = self.o3(X)
		return X 

bs = 32
d  = 512
onet = OrthNet(d=d)
onet.forward(torch.zeros(d, bs))
```

# Bibtex
If you use this code, please cite 
```
@inproceedings{fasth,
    title={{F}aster {O}rthogonal {P}arameterization with {H}ouseholder {M}atrices},
    author={Mathiasen, Alexander and Hvilsh{\o}j, Frederik and J{\o}rgensen, Jakob R{\o}dsgaard and Nasery, Anshul and Mottin, Davide},
    booktitle={ICML Workshop on Invertible Neural Networks and Normalizing Flows},
    year={2020}
}
```

