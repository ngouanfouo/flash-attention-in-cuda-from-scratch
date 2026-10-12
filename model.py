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

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    return expf(old_max - new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    return old_sum * correction + block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    for (int d = 0; d < head_dim; ++d) {
        out_row[d] *= correction;
    }
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    int total = tile_rows * tile_cols;
    for (int i = thread_id; i < total; i += num_threads) {
        int r = i / tile_cols;
        int c = i % tile_cols;
        int gr = src_row_start + r;
        int gc = src_col_start + c;
        float val = 0.0f;
        if (gr < src_rows && gc < src_cols) {
            val = src[gr * src_cols + gc];
        }
        shared_dst[i] = val;
    }
}

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    int total = tile_q * tile_k;
    for (int idx = thread_id; idx < total; idx += num_threads) {
        int i = idx / tile_k;
        int j = idx % tile_k;
        const float* q_row = q_tile + i * head_dim;
        const float* k_row = k_tile + j * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            acc += q_row[d] * k_row[d];
        }
        s_tile[i * tile_k + j] = acc * scale;
    }
}

# Step 19 - tile_rowmax
__device__ void tile_rowmax(const float* s_tile, float* row_max_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float m = -INFINITY;
        const float* row = s_tile + r * tile_k;
        for (int c = 0; c < tile_k; ++c) {
            float v = row[c];
            if (v > m) m = v;
        }
        row_max_out[r] = m;
    }
}

# Step 20 - tile_exp
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    int total = tile_q * tile_k;
    for (int idx = thread_id; idx < total; idx += num_threads) {
        int r = idx / tile_k;
        s_tile[idx] = expf(s_tile[idx] - row_max[r]);
    }
}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float s = 0.0f;
        const float* row = p_tile + r * tile_k;
        for (int c = 0; c < tile_k; ++c) {
            s += row[c];
        }
        row_sum_out[r] = s;
    }
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile, const float* v_tile, float* out_acc,
                              int tile_q, int tile_k, int head_dim,
                              int thread_id, int num_threads) {
    int total = tile_q * head_dim;
    for (int idx = thread_id; idx < total; idx += num_threads) {
        int r = idx / head_dim;
        int d = idx % head_dim;
        const float* p_row = p_tile + r * tile_k;
        float acc = 0.0f;
        for (int c = 0; c < tile_k; ++c) {
            acc += p_row[c] * v_tile[c * head_dim + d];
        }
        out_acc[idx] += acc;
    }
}

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    extern __shared__ float smem[];

    // Shared-memory partition:
    //   q_smem [tile_q * head_dim]
    //   k_smem [tile_k * head_dim]
    //   v_smem [tile_k * head_dim]
    //   s_smem [tile_q * tile_k]
    //   m_tile [tile_q]  (tile row max, then reused as per-row alpha)
    //   l_tile [tile_q]  (tile row sum)
    //   m_run  [tile_q]  (running max)
    //   l_run  [tile_q]  (running sum)
    //   o_acc  [tile_q * head_dim]
    float* q_smem = smem;
    float* k_smem = q_smem + tile_q * head_dim;
    float* v_smem = k_smem + tile_k * head_dim;
    float* s_smem = v_smem + tile_k * head_dim;
    float* m_tile = s_smem + tile_q * tile_k;
    float* l_tile = m_tile + tile_q;
    float* m_run  = l_tile + tile_q;
    float* l_run  = m_run + tile_q;
    float* o_acc  = l_run + tile_q;

    int tid = threadIdx.x;
    int nt  = blockDim.x;
    int q_row_start = blockIdx.x * tile_q;

    // --- Load the Q tile once ---
    load_tile(q, q_smem, q_row_start, 0, seq_len, head_dim,
              tile_q, head_dim, tid, nt);
    __syncthreads();

    // --- Initialize running state ---
    for (int r = tid; r < tile_q; r += nt) {
        m_run[r] = -INFINITY;
        l_run[r] = 0.0f;
    }
    for (int i = tid; i < tile_q * head_dim; i += nt) {
        o_acc[i] = 0.0f;
    }
    __syncthreads();

    // --- Stream over K/V tiles ---
    for (int k_start = 0; k_start < seq_len; k_start += tile_k) {
        load_tile(k, k_smem, k_start, 0, seq_len, head_dim,
                  tile_k, head_dim, tid, nt);
        load_tile(v, v_smem, k_start, 0, seq_len, head_dim,
                  tile_k, head_dim, tid, nt);
        __syncthreads();

        // S_tile = scale * Q_tile * K_tile^T
        tile_scores(q_smem, k_smem, s_smem, tile_q, tile_k, head_dim, scale, tid, nt);
        __syncthreads();

        // Mask keys past seq_len so they drop out of the softmax
        for (int i = tid; i < tile_q * tile_k; i += nt) {
            int j = i % tile_k;
            if (k_start + j >= seq_len) {
                s_smem[i] = -INFINITY;
            }
        }
        __syncthreads();

        // Per-tile row max
        tile_rowmax(s_smem, m_tile, tile_q, tile_k, tid, nt);
        __syncthreads();

        // Update running max; reuse m_tile to stash alpha = exp(m_old - m_new)
        for (int r = tid; r < tile_q; r += nt) {
            float m_new = fmaxf(m_run[r], m_tile[r]);
            float alpha = correction_factor(m_run[r], m_new);
            m_tile[r] = alpha;
            m_run[r]  = m_new;
        }
        __syncthreads();

        // Rescale running output and running sum into the new max frame
        for (int i = tid; i < tile_q * head_dim; i += nt) {
            int r = i / head_dim;
            o_acc[i] *= m_tile[r];
        }
        for (int r = tid; r < tile_q; r += nt) {
            l_run[r] *= m_tile[r];
        }
        __syncthreads();

        // P_tile = exp(S - m_run), in place
        tile_exp(s_smem, m_run, tile_q, tile_k, tid, nt);
        __syncthreads();

        // Per-tile row sum, added into the running denominator
        tile_rowsum(s_smem, l_tile, tile_q, tile_k, tid, nt);
        __syncthreads();

        for (int r = tid; r < tile_q; r += nt) {
            l_run[r] += l_tile[r];
        }
        __syncthreads();

        // o_acc += P_tile * V_tile
        accumulate_pv(s_smem, v_smem, o_acc, tile_q, tile_k, head_dim, tid, nt);
        __syncthreads();
    }

    // --- Normalize and store ---
    for (int i = tid; i < tile_q * head_dim; i += nt) {
        int r = i / head_dim;
        int d = i % head_dim;
        int gr = q_row_start + r;
        if (gr < seq_len) {
            float l = l_run[r];
            out[gr * head_dim + d] = (l > 0.0f) ? (o_acc[i] / l) : 0.0f;
        }
    }
}

# Step 24 - flash_attention_launcher
void flash_attention_launcher(const float* d_q, const float* d_k, const float* d_v,
                              float* d_out, int seq_len, int head_dim,
                              int tile_q, int tile_k) {
    float scale = 1.0f / sqrtf((float)head_dim);

    int blocks = (seq_len + tile_q - 1) / tile_q;

    int threads = 128;

    size_t smem_floats = (size_t)(2 * tile_q * head_dim
                                + 2 * tile_k * head_dim
                                + tile_q * tile_k
                                + 4 * tile_q);
    size_t smem_bytes = smem_floats * sizeof(float);

    flash_attention_kernel<<<blocks, threads, smem_bytes>>>(
        d_q, d_k, d_v, d_out, seq_len, head_dim,
        tile_q, tile_k, scale);
}

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

