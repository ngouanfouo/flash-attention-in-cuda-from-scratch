"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = a[i] * scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = expf(a[i]);
    }
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float m = -INFINITY;
        int base = r * cols;
        for (int c = 0; c < cols; ++c) {
            float v = matrix[base + c];
            if (v > m) m = v;
        }
        out[r] = m;
    }
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int base = r * cols;

    float local = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        local += matrix[base + c];
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[r] = sdata[0];
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // column of C
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row of C
    if (i < m && j < n) {
        float acc = 0.0f;
        for (int l = 0; l < k; ++l) {
            acc += a[i * k + l] * b[l * n + j];
        }
        c[i * n + j] = acc;
    }
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;  // column in input
    int r = blockIdx.y * blockDim.y + threadIdx.y;  // row in input
    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

# Step 9 - qk_scores
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // key index
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // query index
    if (i < seq_len && j < seq_len) {
        const float* q_row = q + i * head_dim;
        const float* k_row = k + j * head_dim;
        float dot = dot_product(q_row, k_row, head_dim);
        float scale = rsqrtf((float)head_dim);
        scores[i * seq_len + j] = dot * scale;
    }
}

# Step 10 - softmax_rows
__global__ void softmax_rows(float* matrix, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int base = r * cols;

    // Phase 1: per-row maximum
    float local_max = -INFINITY;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = matrix[base + c];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Phase 2: sum of exp(x - max)
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        local_sum += expf(matrix[base + c] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // Phase 3: normalize
    float inv_sum = 1.0f / row_sum;
    for (int c = tid; c < cols; c += blockDim.x) {
        matrix[base + c] = expf(matrix[base + c] - row_max) * inv_sum;
    }
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;  // head dimension index
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // sequence index
    if (i < seq_len && d < head_dim) {
        float acc = 0.0f;
        const float* p_row = p + i * seq_len;
        for (int j = 0; j < seq_len; ++j) {
            acc += p_row[j] * v[j * head_dim + d];
        }
        out[i * head_dim + d] = acc;
    }
}

# Step 12 - naive_attention
void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {
    // Allocate scratch for the (seq_len x seq_len) score / probability matrix
    float* d_scores = nullptr;
    size_t scores_bytes = (size_t)seq_len * (size_t)seq_len * sizeof(float);
    cudaMalloc(&d_scores, scores_bytes);

    // 1) scores = Q K^T / sqrt(head_dim)
    {
        dim3 block(16, 16);
        dim3 grid((seq_len + block.x - 1) / block.x,
                  (seq_len + block.y - 1) / block.y);
        qk_scores<<<grid, block>>>(d_q, d_k, d_scores, seq_len, head_dim);
    }

    // 2) softmax_rows in place, one block per row
    {
        int blockSize = 128;
        while (blockSize < seq_len && blockSize < 1024) blockSize <<= 1;
        if (blockSize > 1024) blockSize = 1024;
        size_t smem = (size_t)blockSize * sizeof(float);
        softmax_rows<<<seq_len, blockSize, smem>>>(d_scores, seq_len, seq_len);
    }

    // 3) out = P V
    {
        dim3 block(16, 16);
        dim3 grid((head_dim + block.x - 1) / block.x,
                  (seq_len  + block.y - 1) / block.y);
        pv_matmul<<<grid, block>>>(d_scores, d_v, d_out, seq_len, head_dim);
    }

    cudaFree(d_scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor (not yet solved)
# TODO: implement

# Step 15 - update_running_sum (not yet solved)
# TODO: implement

# Step 16 - rescale_output (not yet solved)
# TODO: implement

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

