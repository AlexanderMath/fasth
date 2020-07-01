/* Inspired by https://pytorch.org/tutorials/advanced/cpp_extension.html */
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <stdio.h> 

// ---------------------------------------------------------------------------------- //
void mult_cuda(             at::Tensor V, at::Tensor X, at::Tensor Y, int m);
void inv_mult_cuda(         at::Tensor V, at::Tensor X, at::Tensor Y, int m);
void compute_dec_cuda(      at::Tensor V, at::Tensor Y, int m);
void backward_cuda(         at::Tensor V, at::Tensor gradV, at::Tensor W, at::Tensor output, at::Tensor grad_output, at::Tensor norms, int m);
// ---------------------------------------------------------------------------------- //
void compute_dec(at::Tensor V, at::Tensor Y, int m)                 { compute_dec_cuda(V, Y, m); }
void mult       (at::Tensor V, at::Tensor X, at::Tensor Y, int m)   { mult_cuda(V, X, Y, m); }
void inv_mult   (at::Tensor V, at::Tensor X, at::Tensor Y, int m)   { inv_mult_cuda(V, X, Y, m); }

void backward(  at::Tensor V, at::Tensor gradV, at::Tensor W, 
                at::Tensor output, at::Tensor grad_output,
                at::Tensor norms, int m) {
    backward_cuda(V, gradV, W, output, grad_output, norms, m);
}
// ---------------------------------------------------------------------------------- //
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mult",                 &mult,              "");
  m.def("inv_mult",             &inv_mult,          "");
  m.def("compute_dec",          &compute_dec,       "");
  m.def("backward",             &backward,          "");
}
