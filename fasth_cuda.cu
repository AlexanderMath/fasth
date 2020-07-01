/* 
    Inspired by https://pytorch.org/tutorials/advanced/cpp_extension.html 

    Implements our algorithm in CUDA/C++.
    This file is not meant to be read/understand. 
    
 */
#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h> 
#include <math.h>

// DIFFERENT CONSANTS.
// OPTIMIZING THESE MIGHT IMROVE PEFROMANCE FURTHER. 

#define TILE_WIDTH_17_1 2
#define THREADS_pr_TILE_17_1 64
#define TILE_WIDTH_17_2 16
#define THREADS_pr_TILE_17_2 2

#define TILE_WIDTH_20_1 2
#define THREADS_pr_TILE_20_1 64
#define TILE_WIDTH_20_2 16
#define THREADS_pr_TILE_20_2 2

// --------------------------------- HELPER FUNCTIONS ---------------------------------------------------- //
__global__ void
clone(at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> from, float *to, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) to[i * cols + j] = from[i][j];
}
void clone_matrix(at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> from, float *to, int rows, int cols) {
    int BLOCK_WIDTH = 16;

    unsigned int grid_rows = rows / BLOCK_WIDTH;
    if (rows % BLOCK_WIDTH) grid_rows++;
    unsigned int grid_cols = cols / BLOCK_WIDTH;
    if (cols % BLOCK_WIDTH) grid_cols++;
    dim3 grid(grid_cols, grid_rows);
    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);

    clone << < grid, block >> > (from, to, rows, cols);
}
// -------------------------------------- COMPUTE WY DECOMPOSITION ----------------------------------------------- //
__global__
void compute_dec_cuda_kernel(
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> Va,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> Wa,
        const int m, int d) {

    int j = blockIdx.x;

    extern __shared__ float mm1_a[];

    for (int row = 1; row < m; row++) {
        for (int l = threadIdx.y; l < row; l += blockDim.y) {
            mm1_a[l] = 0;
        }
        __syncthreads();
        for (int l = threadIdx.y; l < row; l += blockDim.y) {
            float sum = 0;
            for (int h = threadIdx.x; h < d; h += blockDim.x) {
                sum += Wa[j * m + row][h] * Va[j * m + l][h];
            }
            atomicAdd(&mm1_a[l], sum);
        }
        __syncthreads();

        for (int h = threadIdx.x; h < d; h += blockDim.x) {
            float sum = 0;
            for (int l = threadIdx.y; l < row; l += blockDim.y) {
                sum += mm1_a[l] * Wa[j * m + l][h];
            }
            atomicAdd(&Wa[j * m + row][h], -2 * sum);
        }
        __syncthreads();
    }
}

void compute_dec_cuda(at::Tensor V, at::Tensor W, int m) {
    int d = V.size(0);
    auto Wa = W.packed_accessor32<float, 2, at::RestrictPtrTraits>();
    auto Va = V.packed_accessor32<float, 2, at::RestrictPtrTraits>();

    dim3 grid(d / m);
    dim3 block(min(d, 128), min(m, 8));
    compute_dec_cuda_kernel << < grid, block, m >> > (Va, Wa, m, d);
}
// -------------------------------- COMPUTE INVERSE MULT ----------------------------------------------------- //
__global__
void inv_mult_cuda_kernel_a(
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> V,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> X,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> W,
        float *mm1_,
        int m, int d, int bs, int j) {

    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    __shared__ float Ws[TILE_WIDTH_20_1][TILE_WIDTH_20_1][THREADS_pr_TILE_20_1];
    __shared__ float Xs[TILE_WIDTH_20_1][TILE_WIDTH_20_1][THREADS_pr_TILE_20_1];

    int jm = j * m;

    if (l < m && h < bs ) {
        mm1_[l * bs + h] = 0;
    }

    float sum = 0;

    for (int i = threadIdx.z; i < ceilf(d / (float) TILE_WIDTH_20_1); i += blockDim.z) {
        if (l < m && (i * TILE_WIDTH_20_1 + tx) < d) {
            Ws[ty][tx][tz] = W[jm + l][i * TILE_WIDTH_20_1 + tx];
        } else {
            Ws[ty][tx][tz] = 0;
        }
        if (h < bs && (i * TILE_WIDTH_20_1 + ty) < d) {
            Xs[ty][tx][tz] = X[i * TILE_WIDTH_20_1 + ty][h];
        } else {
            Xs[ty][tx][tz] = 0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH_20_1; k++) {
            sum += Ws[ty][k][tz] * Xs[k][tx][tz];
        }
        __syncthreads();
    }

    if (l < m && h < bs)
        atomicAdd(&mm1_[l * bs + h], sum);
}

__global__
void inv_mult_cuda_kernel_b(
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> V,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> X,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> W,
        float *mm1_,
        int m, int d, int bs, int j) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    __shared__ float Vs[TILE_WIDTH_20_2][TILE_WIDTH_20_2][THREADS_pr_TILE_20_2];
    __shared__ float mm1_s[TILE_WIDTH_20_2][TILE_WIDTH_20_2][THREADS_pr_TILE_20_2];

    float sum = 0;
    for (int i = threadIdx.z; i < ceilf(m / (float) TILE_WIDTH_20_2); i += blockDim.z) {
        if (k < d && (i * TILE_WIDTH_20_2 + tx) < m) {
            Vs[ty][tx][tz] = V[j * m + i * TILE_WIDTH_20_2 + tx][k];
        } else {
            Vs[ty][tx][tz] = 0;
        }
        if (h < bs && (i * TILE_WIDTH_20_2 + ty) < m) {
            mm1_s[ty][tx][tz] = mm1_[(i * TILE_WIDTH_20_2 + ty) * bs + h];
        } else {
            mm1_s[ty][tx][tz] = 0;
        }
        __syncthreads();

        for (int l = 0; l < TILE_WIDTH_20_2; l++) {
            sum += Vs[ty][l][tz] * mm1_s[l][tx][tz];
        }
        __syncthreads();
    }
    if (k < d && h < bs)
        atomicAdd(&X[k][h], -2 * sum);
}

void inv_mult_cuda(at::Tensor V, at::Tensor X, at::Tensor Y, int m) {

    int bs = X.size(1);
    int d = X.size(0);

    auto Wa = Y.packed_accessor32<float, 2, at::RestrictPtrTraits>();
    auto Va = V.packed_accessor32<float, 2, at::RestrictPtrTraits>();
    auto Xa = X.packed_accessor32<float, 2, at::RestrictPtrTraits>();

    float *mm1_a;
    cudaMalloc(&mm1_a, sizeof(float) * m * bs);

    for (int j = 0; j < d / m; j++) {

        unsigned int grid_rows = m / TILE_WIDTH_20_1;
        if (m % TILE_WIDTH_20_1) grid_rows++;
        unsigned int grid_cols = bs / TILE_WIDTH_20_1;
        if (bs % TILE_WIDTH_20_1) grid_cols++;
        dim3 grid_1(grid_cols, grid_rows);
        dim3 block_1(TILE_WIDTH_20_1, TILE_WIDTH_20_1, THREADS_pr_TILE_20_1);

        inv_mult_cuda_kernel_a << < grid_1, block_1 >> > (Va, Xa, Wa, mm1_a, m, d, bs, j);

        grid_rows = d / TILE_WIDTH_20_2;
        if (d % TILE_WIDTH_20_2) grid_rows++;
        grid_cols = bs / TILE_WIDTH_20_2;
        if (bs % TILE_WIDTH_20_2) grid_cols++;
        dim3 grid_2(grid_cols, grid_rows);
        dim3 block_2(TILE_WIDTH_20_2, TILE_WIDTH_20_2, THREADS_pr_TILE_20_2);

        inv_mult_cuda_kernel_b << < grid_2, block_2 >> > (Va, Xa, Wa, mm1_a, m, d, bs, j);
    }
    cudaFree(mm1_a);
}

// ------------------------------------ COMPUTE MULT ------------------------------------------------- //
__global__ void mult_kernel_a(
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> V_acc,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> W_acc,
        float *M1_acc,
        float *M2_acc,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_output_acc,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output_acc,
        int d, int m, int batch_size, int j) {

    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    M1_acc[l * batch_size + h] = 0;
    M2_acc[l * batch_size + h] = 0;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    __shared__ float V_acc_s[TILE_WIDTH_17_1][TILE_WIDTH_17_1][THREADS_pr_TILE_17_1];
    __shared__ float grad_output_acc_s[TILE_WIDTH_17_1][TILE_WIDTH_17_1][THREADS_pr_TILE_17_1];
    __shared__ float output_acc_s[TILE_WIDTH_17_1][TILE_WIDTH_17_1][THREADS_pr_TILE_17_1];

    float sum_grad = 0;
    float sum_output = 0;
    for (int i = threadIdx.z; i < ceilf(d / (float) TILE_WIDTH_17_1); i += blockDim.z) {
        if (l < m && (i * TILE_WIDTH_17_1 + tx) < d) {
            V_acc_s[ty][tx][tz] = V_acc[j * m + l][i * TILE_WIDTH_17_1 + tx];
        } else {
            V_acc_s[ty][tx][tz] = 0;
        }
        if (h < batch_size && (i * TILE_WIDTH_17_1 + ty) < d) {
            grad_output_acc_s[ty][tx][tz] = grad_output_acc[i * TILE_WIDTH_17_1 + ty][h];
            output_acc_s[ty][tx][tz] = output_acc[i * TILE_WIDTH_17_1 + ty][h];
        } else {
            grad_output_acc_s[ty][tx][tz] = 0;
            output_acc_s[ty][tx][tz] = 0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH_17_1; k++) {
            sum_grad += V_acc_s[ty][k][tz] * grad_output_acc_s[k][tx][tz];
            sum_output += V_acc_s[ty][k][tz] * output_acc_s[k][tx][tz];
        }
        __syncthreads();
    }
    if (l < m && h < batch_size) {
        atomicAdd(&M1_acc[l * batch_size + h], sum_grad);
        atomicAdd(&M2_acc[l * batch_size + h], sum_output);
    }
}


__global__ void mult_kernel_b(
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> V_acc,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> W_acc,
        float *M1_acc,
        float *M2_acc,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_output_acc,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output_acc,
        int d, int m, int batch_size, int j) {


    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    __shared__ float W_acc_s[TILE_WIDTH_17_2][TILE_WIDTH_17_2][THREADS_pr_TILE_17_2];
    __shared__ float M1_acc_s[TILE_WIDTH_17_2][TILE_WIDTH_17_2][THREADS_pr_TILE_17_2];
    __shared__ float M2_acc_s[TILE_WIDTH_17_2][TILE_WIDTH_17_2][THREADS_pr_TILE_17_2];


    float sum_grad = 0;
    float sum_output = 0;

    for (int i = threadIdx.z; i < ceilf(m / (float) TILE_WIDTH_17_2); i += blockDim.z) {
        if (k < d && (i * TILE_WIDTH_17_2 + tx) < m) {
            W_acc_s[ty][tx][tz] = W_acc[j * m + i * TILE_WIDTH_17_2 + tx][k];
        } else {
            W_acc_s[ty][tx][tz] = 0;
        }
        if (h < batch_size && (i * TILE_WIDTH_17_2 + ty) < m) {
            M1_acc_s[ty][tx][tz] = M1_acc[(i * TILE_WIDTH_17_2 + ty) * batch_size + h];
            M2_acc_s[ty][tx][tz] = M2_acc[(i * TILE_WIDTH_17_2 + ty) * batch_size + h];
        } else {
            M1_acc_s[ty][tx][tz] = 0;
            M2_acc_s[ty][tx][tz] = 0;
        }
        __syncthreads();

        for (int l = 0; l < TILE_WIDTH_17_2; l++) {
            sum_grad -= W_acc_s[ty][l][tz] * M1_acc_s[l][tx][tz];
            sum_output -= W_acc_s[ty][l][tz] * M2_acc_s[l][tx][tz];
        }
        __syncthreads();
    }
    if (k < d && h < batch_size) {
        atomicAdd(&grad_output_acc[k][h], sum_grad);
        atomicAdd(&output_acc[k][h], sum_output);
    }

}

__global__ void compute_bucket_j(float *input_list,
                                    float *grad_output_list,
                                    float *V,
                                    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> gradV,
                                    int d, int m, int batch_size, float *norms) {
    int j = blockIdx.x;

    float *input = &input_list[j * d * batch_size];
    float *grad_output = &grad_output_list[j * d * batch_size];

    extern __shared__ float shared_array[];

    float *alpha = shared_array;
    float *beta = &alpha[batch_size];
    float *c = &beta[batch_size];

#pragma unroll 8
    for (int i = 0; i < m; i++) {
        int indx = (j + 1) * m - i - 1;
        float normnorm2 = 2 / (norms[indx] * norms[indx]);
        for (int k = (threadIdx.x * blockDim.y + threadIdx.y); k < d; k += (blockDim.x * blockDim.y)) {
            V[indx * d + k] = norms[indx] * V[indx * d + k];
            gradV[indx][k] = 0;
        }
        for (int l = (threadIdx.x * blockDim.y + threadIdx.y); l < batch_size; l += (blockDim.y * blockDim.x)) {
            alpha[l] = float(0);
            beta[l] = float(0);
            c[l] = 0;
        }
        __syncthreads();
        for (int l = threadIdx.y; l < batch_size; l += blockDim.y) {
            float local_c = 0;
            for (int k = threadIdx.x; k < d; k += blockDim.x) {
                local_c += V[indx * d + k] * input[k * batch_size + l];
            }
            atomicAdd(&c[l], local_c * normnorm2);
        }
        __syncthreads();
        for (int l = threadIdx.y; l < batch_size; l += blockDim.y) {

            float local_alpha = 0;
            float local_beta = 0;
            for (int k = threadIdx.x; k < d; k += blockDim.x) {
                input[k * batch_size + l] -= V[indx * d + k] * c[l];
                local_alpha += V[indx * d + k] * input[k * batch_size + l];
                local_beta += V[indx * d + k] * grad_output[k * batch_size + l];
            }
            atomicAdd(&alpha[l], local_alpha * normnorm2);
            atomicAdd(&beta[l], local_beta * normnorm2);
        }

        __syncthreads();
        for (int k = threadIdx.x; k < d; k += blockDim.x) {
            float sum = 0;
            for (int h = threadIdx.y; h < batch_size; h += blockDim.y) {
                grad_output[k * batch_size + h] -= V[indx * d + k] * beta[h];
                sum += -grad_output[k * batch_size + h] * alpha[h] - input[k * batch_size + h] * beta[h];
            }
            atomicAdd(&gradV[indx][k], sum);
        }
    }
}




void backward_cuda( at::Tensor V, at::Tensor gradV, at::Tensor W, at::Tensor output,
                    at::Tensor grad_output, at::Tensor norms_, int m) {

    float *norms = norms_.data_ptr<float>();
    W = at::mul(W, 2.); 

    int d = V.size(0);
    int batch_size = output.size(1);


    auto V_acc = V.packed_accessor32<float, 2, at::RestrictPtrTraits>();
    auto gradV_acc = gradV.packed_accessor32<float, 2, at::RestrictPtrTraits>();
    auto W_acc = W.packed_accessor32<float, 2, at::RestrictPtrTraits>();
    auto output_acc = output.packed_accessor32<float, 2, at::RestrictPtrTraits>();
    auto grad_output_acc = grad_output.packed_accessor32<float, 2, at::RestrictPtrTraits>();

    float *M1_acc;
    cudaMalloc(&M1_acc, sizeof(float) * m * batch_size);
    float *M2_acc;
    cudaMalloc(&M2_acc, sizeof(float) * m * batch_size);

    float *V_clone_acc;
    cudaMalloc(&V_clone_acc, sizeof(float) * d * d);
    clone_matrix(V_acc, V_clone_acc, d, d);

    int num_of_buckets = d / m;

    float *output_list;
    cudaMalloc(&output_list, sizeof(float) * d * batch_size * num_of_buckets);


    float *grad_output_list;
    cudaMalloc(&grad_output_list, sizeof(float) * d * batch_size * num_of_buckets);


    int grid_rows = m / TILE_WIDTH_17_1;
    if (m % TILE_WIDTH_17_1) grid_rows++;
    int grid_cols = batch_size / TILE_WIDTH_17_1;
    if (batch_size % TILE_WIDTH_17_1) grid_cols++;
    dim3 grid_1(grid_cols, grid_rows);
    dim3 block_1(TILE_WIDTH_17_1, TILE_WIDTH_17_1, THREADS_pr_TILE_17_1);

    grid_rows = d / TILE_WIDTH_17_2;
    if (d % TILE_WIDTH_17_2) grid_rows++;
    grid_cols = batch_size / TILE_WIDTH_17_2;
    if (batch_size % TILE_WIDTH_17_2) grid_cols++;
    dim3 grid_2(grid_cols, grid_rows);
    dim3 block_2(TILE_WIDTH_17_2, TILE_WIDTH_17_2, THREADS_pr_TILE_17_2);

    /* ---------- STEP 1 ---------- */
    for (int j = int(d / m - 1); j > -1; j--) {
        float *output_j = &output_list[j * d * batch_size];
        float *grad_output_j = &grad_output_list[j * d * batch_size];
        clone_matrix(output_acc, output_j, d, batch_size);
        clone_matrix(grad_output_acc, grad_output_j, d, batch_size);

        mult_kernel_a << < grid_1, block_1 >> >
                                      (V_acc, W_acc, M1_acc, M2_acc, grad_output_acc, output_acc, d, m, batch_size, j);

        mult_kernel_b << < grid_2, block_2 >> >
                                      (V_acc, W_acc, M1_acc, M2_acc, grad_output_acc, output_acc, d, m, batch_size, j);
    }


    /* ---------- STEP 2 ---------- */
    int share_memory_size = 3 * sizeof(float) * batch_size;
    dim3 block(512 / min(512, batch_size), min(512, batch_size));

    compute_bucket_j<< < num_of_buckets, block, share_memory_size >> > (
            output_list, grad_output_list,
                    V_clone_acc, gradV_acc,
                    d, m, batch_size, norms);

    cudaFree(output_list);
    cudaFree(grad_output_list);
    cudaFree(M1_acc);
    cudaFree(M2_acc);
    cudaFree(V_clone_acc);
}


// ------------------------------------- BACKPROPAGATION ------------------------------------------------ //
// code is written to compute gradients of inv_mult which might be a bit confusing.
// this was done to simplify our implementation. 

__global__
void mult_cuda_kernel_a( 
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> V,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> X,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> W,
        float *mm1_,
        int m, int d, int bs, int j) {
    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    __shared__ float Vs[TILE_WIDTH_20_1][TILE_WIDTH_20_1][THREADS_pr_TILE_20_1];
    __shared__ float Xs[TILE_WIDTH_20_1][TILE_WIDTH_20_1][THREADS_pr_TILE_20_1];


    float sum = 0;

    if (l < m && h < bs)
        mm1_[l * bs + h] = 0;

    for (int i = threadIdx.z; i < ceilf(d / (float) TILE_WIDTH_20_1); i += blockDim.z) {
        if (l < m && (i * TILE_WIDTH_20_1 + tx) < d) {
            Vs[ty][tx][tz] = V[j * m + l][i * TILE_WIDTH_20_1 + tx];
        } else {
            Vs[ty][tx][tz] = 0;
        }
        if (h < bs && (i * TILE_WIDTH_20_1 + ty) < d) {
            Xs[ty][tx][tz] = X[i * TILE_WIDTH_20_1 + ty][h];
        } else {
            Xs[ty][tx][tz] = 0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH_20_1; k++) { sum += Vs[ty][k][tz] * Xs[k][tx][tz];
        }
        __syncthreads();
    }
    if (l < m && h < bs)
        atomicAdd(&mm1_[l * bs + h], sum);
}

__global__
void mult_cuda_kernel_b( 
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> V,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> X,
        at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> W,
        float *mm1_,
        int m, int d, int bs, int j) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    __shared__ float Ws[TILE_WIDTH_20_2][TILE_WIDTH_20_2][THREADS_pr_TILE_20_2];
    __shared__ float mm1_s[TILE_WIDTH_20_2][TILE_WIDTH_20_2][THREADS_pr_TILE_20_2];

    float sum = 0;
    for (int i = threadIdx.z; i < ceilf(m / (float) TILE_WIDTH_20_2); i += blockDim.z) {
        if (k < d && (i * TILE_WIDTH_20_2 + tx) < m) {
            Ws[ty][tx][tz] = W[j * m + i * TILE_WIDTH_20_2 + tx][k];
        } else {
            Ws[ty][tx][tz] = 0;
        }
        if (h < bs && (i * TILE_WIDTH_20_2 + ty) < m) {
            mm1_s[ty][tx][tz] = mm1_[(i * TILE_WIDTH_20_2 + ty) * bs + h];
        } else {
            mm1_s[ty][tx][tz] = 0;
        }
        __syncthreads();

        for (int l = 0; l < TILE_WIDTH_20_2; l++) {
            sum += Ws[ty][l][tz] * mm1_s[l][tx][tz];
        }
        __syncthreads();
    }
    if (k < d && h < bs)
        atomicAdd(&X[k][h], -2 * sum);
}

void mult_cuda(at::Tensor V, at::Tensor X, at::Tensor Y, int m) {
    int bs = X.size(1);
    int d = X.size(0);

    auto Wa = Y.packed_accessor32<float, 2, at::RestrictPtrTraits>();
    auto Va = V.packed_accessor32<float, 2, at::RestrictPtrTraits>();
    auto Xa = X.packed_accessor32<float, 2, at::RestrictPtrTraits>();
    float *mm1_a;
    cudaMalloc(&mm1_a, sizeof(float) * m * bs);

    for (int j = d / m - 1; j > -1; j--) {

        unsigned int grid_rows = m / TILE_WIDTH_20_1;
        if (m % TILE_WIDTH_20_1) grid_rows++;
        unsigned int grid_cols = bs / TILE_WIDTH_20_1;
        if (bs % TILE_WIDTH_20_1) grid_cols++;
        dim3 grid_1(grid_cols, grid_rows);
        dim3 block_1(TILE_WIDTH_20_1, TILE_WIDTH_20_1, THREADS_pr_TILE_20_1);
        mult_cuda_kernel_a << < grid_1, block_1 >> > (Va, Xa, Wa, mm1_a, m, d, bs, j);

        grid_rows = d / TILE_WIDTH_20_2;
        if (d % TILE_WIDTH_20_2) grid_rows++;
        grid_cols = bs / TILE_WIDTH_20_2;
        if (bs % TILE_WIDTH_20_2) grid_cols++;
        dim3 grid_2(grid_cols, grid_rows);
        dim3 block_2(TILE_WIDTH_20_2, TILE_WIDTH_20_2, THREADS_pr_TILE_20_2);
        mult_cuda_kernel_b << < grid_2, block_2 >> > (Va, Xa, Wa, mm1_a, m, d, bs, j);
    }
}

