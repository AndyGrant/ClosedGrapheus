#include "affine_sparse.h"
// clang-format off
__global__ void operations::affine_sparse_kernel(
    const float*        __restrict__ mat,
    const size_t*       __restrict__ inp_col_indices,
    const size_t                     inp_col_max_entries,
    const float*        __restrict__ bia,
          float*        __restrict__ res,
    const size_t                     m,
    const size_t                     n,
    const size_t                     lda,
    const size_t                     ldc){

    // clang-format on
    // compute which output value we are looking at
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // skip out of bounds
    if (col >= n || row >= m)
        return;

    // get the offset at which we look into our sparse input
    int offset = col * (inp_col_max_entries + 1);
    // check how many values we are going to read
    int count = inp_col_indices[offset];

    // track the sum
    float sum = bia[row];

    // start at offset + 1 (offset contains the amount of values to read)
    for (int i = offset + 1; i < offset + 1 + count; i++) {

        // get the sparse index (set row of the input)
        auto b_row = inp_col_indices[i];
        // get the corresponding weight
        auto wgt = mat[MATRIX_INDEX(lda, row, b_row)];

        sum += wgt;
    }
    res[MATRIX_INDEX(ldc, row, col)] = sum;
};


// clang-format off
__global__ void operations::affine_sparse_qat_kernel(
    const float*        __restrict__ mat,
    const size_t*       __restrict__ inp_col_indices,
    const size_t                     inp_col_max_entries,
    const float*        __restrict__ bia,
          float*        __restrict__ res,
    const size_t                     m,
    const size_t                     n,
    const size_t                     lda,
    const size_t                     ldc,
    const float                      quant_scalar) {

    // clang-format on
    // compute which output value we are looking at
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // skip out of bounds
    if (col >= n || row >= m)
        return;

    // get the offset at which we look into our sparse input
    int offset = col * (inp_col_max_entries + 1);
    // check how many values we are going to read
    int count = inp_col_indices[offset];

    // track the sum
    float sum = bia[row];

    // start at offset + 1 (offset contains the amount of values to read)
    for (int i = offset + 1; i < offset + 1 + count; i++) {

        // get the sparse index (set row of the input)
        auto b_row = inp_col_indices[i];
        // get the corresponding weight
        auto wgt = mat[MATRIX_INDEX(lda, row, b_row)];

        sum += (int) (wgt * quant_scalar) / quant_scalar;
    }
    res[MATRIX_INDEX(ldc, row, col)] = sum;
};