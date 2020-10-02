# FastH
Code accompanying article <a href="https://arxiv.org/abs/2009.13977">What if Neural Networks had SVDs?</a> accepted for spotlight presentation at NeurIPS 2020. 

<p align="center">
<img src="plot.png" width="400px" height="200px" >
</p>

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
    title={What If Neural Networks had SVDs?,
    author={Mathiasen, Alexander and Hvilsh{\o}j, Frederik and J{\o}rgensen, Jakob R{\o}dsgaard and Nasery, Anshul and Mottin, Davide},
    booktitle={NeurIPS},
    year={2020}
}
```
A previous version of the <a href="https://invertibleworkshop.github.io/accepted_papers/pdfs/10.pdf" target="_blank">article</a> was presented at the ICML workshop on <a target="_blank" href="https://invertibleworkshop.github.io/">Invertible Neural Networks and Normalizing Flows</a>. This does not constitute a dual submission because the workshop does not qualify as an archival peer reviewed venue.
