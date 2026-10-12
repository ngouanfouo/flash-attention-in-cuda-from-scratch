# Flash Attention in CUDA from Scratch

> A step-by-step, ground-up implementation of tiled, IO-aware Flash Attention in CUDA, from basic GPU primitives to a fused, causal, online-softmax attention kernel.

![CUDA](https://img.shields.io/badge/CUDA-GPU%20Programming-76B900?logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## Table of Contents

- [Introduction](#introduction)
- [Why Flash Attention?](#why-flash-attention)
- [Key Concepts](#key-concepts)
- [Features](#features)
- [Project Roadmap](#project-roadmap)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Learning Outcomes](#learning-outcomes)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## Introduction

Attention is the core operation behind modern Transformer models, but the standard implementation is limited by **memory bandwidth**, not compute. It materializes the full `N x N` score matrix in GPU global memory (HBM), reads and writes it several times, and its memory use grows quadratically with sequence length.

**Flash Attention** solves this by restructuring the computation. Instead of building the full attention matrix, it processes `Q`, `K` and `V` in small **tiles** that fit in fast on-chip shared memory, and uses an **online softmax** to produce exact results without ever storing the full score matrix. The result is the same output as standard attention, with far less memory traffic and linear memory use in sequence length.

This project teaches and implements that idea **from scratch in CUDA**. It begins with elementary GPU kernels (vector addition, reductions, matrix multiplication), builds a naive attention baseline, works through the online softmax math, and ends with a fused Flash Attention kernel plus a **causal** variant used in autoregressive models such as GPT-style language models.

## Why Flash Attention?

| Aspect | Standard Attention | Flash Attention |
|---|---|---|
| Score matrix storage | Full `N x N` in HBM | Never materialized |
| Memory complexity | `O(N^2)` | `O(N)` |
| HBM reads/writes | Many passes over the score matrix | Minimal, tile-based |
| Kernel launches | Separate kernels (matmul, softmax, matmul) | Single fused kernel |
| Numerical result | Exact | Exact |

## Key Concepts

- **Tiling:** splitting `Q`, `K`, `V` into blocks that fit in shared memory so data is loaded from HBM once and reused.
- **IO-awareness:** designing the algorithm around the cost of moving data between memory levels, not just the FLOP count.
- **Online softmax:** computing softmax incrementally using a running row maximum and running sum, with a correction factor to rescale earlier partial results.
- **Kernel fusion:** merging `QK^T`, softmax and `PV` into one kernel to avoid round trips to global memory.
- **Causal masking:** preventing each token from attending to future tokens, which is required for autoregressive generation.

## Features

- Progressive, bottom-up curriculum of 26 small, testable steps
- Naive attention baseline for correctness comparison
- Online softmax building blocks (running max, correction factor, running sum, output rescaling)
- Tile-level primitives (load, score, row max, exp, row sum, accumulate)
- Fused Flash Attention CUDA kernel and host-side launcher
- Causal-masked Flash Attention kernel for autoregressive models

## Project Roadmap

The project is organized into six stages. Each step builds on the previous ones.

### Stage 1: GPU Primitives

- [x] **1.** `vector_add`
- [x] **2.** `scale_array`
- [x] **3.** `elementwise_exp`
- [x] **4.** `row_max`
- [x] **5.** `row_sum`
- [x] **6.** `dot_product`

### Stage 2: Linear Algebra Building Blocks

- [x] **7.** `matmul`
- [x] **8.** `transpose`

### Stage 3: Naive Attention Baseline

- [x] **9.** `qk_scores`
- [x] **10.** `softmax_rows`
- [x] **11.** `pv_matmul`
- [x] **12.** `naive_attention`

### Stage 4: Online Softmax

- [x] **13.** `online_max`
- [x] **14.** `correction_factor`
- [x] **15.** `update_running_sum`
- [x] **16.** `rescale_output`

### Stage 5: Tiled Flash Attention

- [x] **17.** `load_tile`
- [x] **18.** `tile_scores`
- [x] **19.** `tile_rowmax`
- [x] **20.** `tile_exp`
- [x] **21.** `tile_rowsum`
- [x] **22.** `accumulate_pv`
- [x] **23.** `flash_attention_kernel`
- [x] **24.** `flash_attention_launcher`

### Stage 6: Causal Attention

- [x] **25.** `causal_mask`
- [x] **26.** `flash_attention_causal_kernel`

## Prerequisites

- An **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** (including `nvcc`) installed and on your `PATH`
- **Python 3.8+**
- A working NVIDIA driver (verify with `nvidia-smi`)

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Verify your environment

```bash
nvcc --version
nvidia-smi
```

### 3. Run the project

```bash
python scaffold.py
```

### 4. Work through the steps

Follow the roadmap in order. Each step is a small kernel or function that you implement and verify before moving to the next. Compare your outputs against the naive attention baseline (Step 12) to confirm that the fused kernels return the same results.

## Project Structure

```text
.
├── scaffold.py      # Entry point: sets up and runs the project steps
├── README.md        # Project documentation
└── ...              # Kernels and supporting files for each step
```

> Adjust this tree to match your final repository layout.

## Learning Outcomes

By completing this project you will be able to:

1. Write and launch CUDA kernels, and reason about threads, blocks and grids.
2. Use shared memory and tiling to reduce global memory traffic.
3. Implement parallel reductions (max, sum, dot product).
4. Explain why standard attention is memory-bound.
5. Derive and implement the online softmax recurrence.
6. Fuse multiple operations into a single efficient kernel.
7. Extend an attention kernel with causal masking for autoregressive models.

## Contributing

Contributions, bug reports and suggestions are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

**Tiayo Durel**

## Acknowledgments

- Built on [Deep-ML](https://www.deep-ml.com/).
- Based on the ideas from *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* by Tri Dao et al.
- Online softmax formulation from *Online normalizer calculation for softmax* by Milakov and Gimelshein.
