// Hand-written CUDA kernels for the native Cohere ASR engine.
//
// Conventions:
//   - Output pointer is ALWAYS the first parameter (matching the launch helpers).
//   - All kernels `extern "C"`, storage dtype = __half, accumulation = float.
//   - sm_61+ target (P104-100). NO tensor-core / wmma intrinsics.
//   - These kernels are bandwidth/launch-bound elementwise + fusion ops.

#include <cuda_fp16.h>

// ============================================================================
// ReLU in-place
// ============================================================================

extern "C" __global__ void relu_inplace_f16(__half* __restrict__ x, int numel) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int base = t * 2;
    if (base >= numel) return;
    if (base + 1 < numel) {
        __half2 v = *reinterpret_cast<__half2*>(&x[base]);
        __half2 zero = __float2half2_rn(0.0f);
        *reinterpret_cast<__half2*>(&x[base]) = __hmax2(v, zero);
    } else {
        float v = __half2float(x[base]);
        x[base] = __float2half(v > 0.0f ? v : 0.0f);
    }
}

// ============================================================================
// LayerNorm: output ptr first, then inputs
// ============================================================================

extern "C" __global__ void layer_norm_f16(
    __half* __restrict__ y,        // [rows, dim] — output
    const __half* __restrict__ x,  // [rows, dim]
    const __half* __restrict__ w,  // [dim]
    const __half* __restrict__ b,  // [dim]
    int rows, int dim,
    float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const __half* xr = x + (size_t)row * dim;
    __half* yr = y + (size_t)row * dim;

    float local_sum = 0.0f, local_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(xr[i]);
        local_sum += v;
        local_sq += v * v;
    }
    smem[tid] = local_sum;
    smem[tid + blockDim.x] = local_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            smem[tid + blockDim.x] += smem[tid + blockDim.x + s];
        }
        __syncthreads();
    }
    float mean = smem[0] / dim;
    float var = smem[blockDim.x] / dim - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < dim; i += blockDim.x) {
        float v = (__half2float(xr[i]) - mean) * inv_std;
        yr[i] = __float2half(v * __half2float(w[i]) + __half2float(b[i]));
    }
}

// ============================================================================
// SiLU + bias: output first, then x, then bias
// ============================================================================

extern "C" __global__ void silu_bias_f16(
    __half* __restrict__ y,             // [rows, cols] — output
    const __half* __restrict__ x,       // [rows, cols]
    const __half* __restrict__ bias,    // [cols]
    int numel, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int c = idx % cols;
    float v = __half2float(x[idx]) + __half2float(bias[c]);
    float s = 1.0f / (1.0f + expf(-v));
    y[idx] = __float2half(v * s);
}

// ============================================================================
// Bias add in-place
// ============================================================================

extern "C" __global__ void add_bias_inplace_f16(
    __half* __restrict__ x,
    const __half* __restrict__ bias,
    int numel, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int c = idx % cols;
    x[idx] = __float2half(__half2float(x[idx]) + __half2float(bias[c]));
}

// ============================================================================
// Bias + residual: output first, then inputs
// ============================================================================

extern "C" __global__ void bias_residual_f16(
    __half* __restrict__ y,              // [rows, cols] — output
    const __half* __restrict__ out,      // [rows, cols]
    const __half* __restrict__ bias,     // [cols]
    const __half* __restrict__ residual, // [rows, cols]
    int numel, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int c = idx % cols;
    y[idx] = __float2half(
        __half2float(out[idx]) + __half2float(bias[c]) + __half2float(residual[idx])
    );
}

// ============================================================================
// Softmax over last dim: output first, then input
// ============================================================================

extern "C" __global__ void softmax_last_dim_f16(
    __half* __restrict__ y,        // [a, b, dim] — output
    const __half* __restrict__ x,  // [a, b, dim]
    int rows_ab, int dim
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows_ab) return;
    int tid = threadIdx.x;
    const __half* xr = x + (size_t)row * dim;
    __half* yr = y + (size_t)row * dim;

    // Pass 1: max
    float local_max = -__int_as_float(0x7f800000);
    for (int i = tid; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, __half2float(xr[i]));
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float mx = smem[0];

    // Pass 2: exp + sum
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float e = expf(__half2float(xr[i]) - mx);
        smem[blockDim.x + i] = e;
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];

    // Pass 3: divide (recompute exp)
    for (int i = tid; i < dim; i += blockDim.x) {
        float e = expf(__half2float(xr[i]) - mx);
        yr[i] = __float2half(e * inv_sum);
    }
}

// ============================================================================
// Relative-position shift (matches launch order: bd, out, heads, q_len, pos_len)
// ============================================================================

extern "C" __global__ void rel_shift_rank3_f16(
    const __half* __restrict__ bd,
    __half* __restrict__ out,
    int heads, int q_len, int pos_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * q_len * pos_len;
    if (idx >= total) return;

    int j = idx % pos_len;
    int rem = idx / pos_len;
    int i_dst = rem % q_len;
    int h = rem / q_len;

    int flat2 = i_dst * pos_len + j;
    int Rp = flat2 / q_len;
    int C = flat2 % q_len;
    int R = Rp + 1;
    int flat = R * q_len + C;
    int i_src = flat / (pos_len + 1);
    int slot = flat % (pos_len + 1);

    if (slot == 0) {
        out[idx] = __float2half(0.0f);
    } else {
        int k = slot - 1;
        out[idx] = bd[(h * q_len + i_src) * pos_len + k];
    }
}

// ============================================================================
// Scale in-place
// ============================================================================

extern "C" __global__ void scale_inplace_f16(
    __half* __restrict__ x,
    int numel,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    x[idx] = __float2half(__half2float(x[idx]) * scale);
}

// ============================================================================
// Elementwise add: output first, then inputs
// ============================================================================

extern "C" __global__ void add_f16(
    __half* __restrict__ y,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    y[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
}

// ============================================================================
// GLU + depthwise conv k9 + SiLU, fused: output first, then inputs
// ============================================================================

// Pass 1: GLU gate — gated[t, c] = (x[t,c]+bias[c]) * sigmoid(x[t,C+c]+bias[C+c]).
// Materializes [tokens, C] so the conv pass reads C elems/neighbor (not 2C) and
// avoids recomputing the sigmoid 9× per output.
extern "C" __global__ void glu_gate_f16(
    __half* __restrict__ gated,       // [tokens, C]
    const __half* __restrict__ x,     // [tokens, 2*C]
    const __half* __restrict__ bias,  // [2*C]
    int tokens, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * C;
    if (idx >= total) return;
    int t = idx / C;
    int ch = idx % C;
    int C2 = C * 2;
    float left = __half2float(x[(size_t)t * C2 + ch]) + __half2float(bias[ch]);
    float right = __half2float(x[(size_t)t * C2 + C + ch]) + __half2float(bias[C + ch]);
    float g = left * (1.0f / (1.0f + expf(-right)));
    gated[idx] = __float2half(g);
}

// Pass 2: depthwise conv k9 + SiLU on the precomputed gated buffer.
extern "C" __global__ void dw_conv_silu_f16(
    __half* __restrict__ y,               // [tokens, C]
    const __half* __restrict__ gated,     // [tokens, C]
    const __half* __restrict__ cdw_params,// [C, 10]
    int tokens, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * C;
    if (idx >= total) return;
    int t = idx / C;
    int ch = idx % C;

    float acc = __half2float(cdw_params[(size_t)ch * 10 + 9]); // bias
    for (int k = 0; k < 9; k++) {
        int src_t = t + k - 4;
        if (src_t >= 0 && src_t < tokens) {
            acc += __half2float(gated[(size_t)src_t * C + ch])
                 * __half2float(cdw_params[(size_t)ch * 10 + k]);
        }
    }
    float silu = acc * (1.0f / (1.0f + expf(-acc)));
    y[idx] = __float2half(silu);
}

// Original single-pass GLU+depthwise (kept for parity reference; not used in hot path).
extern "C" __global__ void glu_depthwise_conv_f16(
    __half* __restrict__ y,
    const __half* __restrict__ x,
    const __half* __restrict__ bias,
    const __half* __restrict__ cdw_params,
    int tokens, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * C;
    if (idx >= total) return;
    int t = idx / C;
    int ch = idx % C;

    int C2 = C * 2;
    float left_val = __half2float(x[(size_t)t * C2 + ch]) + __half2float(bias[ch]);
    float right_val = __half2float(x[(size_t)t * C2 + C + ch]) + __half2float(bias[C + ch]);
    float gated_center = left_val * (1.0f / (1.0f + expf(-right_val)));

    float acc = __half2float(cdw_params[(size_t)ch * 10 + 9]);

    for (int k = 0; k < 9; k++) {
        int src_t = t + k - 4;
        if (src_t >= 0 && src_t < tokens) {
            float l = __half2float(x[(size_t)src_t * C2 + ch]) + __half2float(bias[ch]);
            float r = __half2float(x[(size_t)src_t * C2 + C + ch]) + __half2float(bias[C + ch]);
            float g = l * (1.0f / (1.0f + expf(-r)));
            acc += g * __half2float(cdw_params[(size_t)ch * 10 + k]);
        }
    }

    float silu = acc * (1.0f / (1.0f + expf(-acc)));
    y[idx] = __float2half(silu);
}

// ============================================================================
// Stage 3.3 — fused data-layout kernels (eliminate CPU round-trips)
// ============================================================================

// Fused qkv split + head reshape + pos bias: produces 4 outputs from qkv.
// qkv: [tokens, 3*D] where D = heads * head_dim
// pos_bias_u, pos_bias_v: [D]
// q_u, q_v, k, v: [heads, tokens, head_dim]
// One thread per element of the output; each thread writes all 4 outputs.
extern "C" __global__ void split_qkv_heads_bias_f16(
    const __half* __restrict__ qkv,         // [tokens, 3*D]
    const __half* __restrict__ pos_bias_u,  // [D]
    const __half* __restrict__ pos_bias_v,  // [D]
    __half* __restrict__ q_u,               // [heads, tokens, head_dim]
    __half* __restrict__ q_v,               // [heads, tokens, head_dim]
    __half* __restrict__ k,                 // [heads, tokens, head_dim]
    __half* __restrict__ v,                 // [heads, tokens, head_dim]
    int tokens, int heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * tokens * head_dim;
    if (idx >= total) return;

    int D = heads * head_dim;
    int h = idx / (tokens * head_dim);
    int rem = idx % (tokens * head_dim);
    int t = rem / head_dim;
    int d = rem % head_dim;

    // Source index in qkv flat array
    size_t q_base = (size_t)t * 3 * D + (size_t)h * head_dim + d;
    size_t k_base = q_base + D;
    size_t v_base = q_base + 2 * D;

    float bias_u = __half2float(pos_bias_u[h * head_dim + d]);
    float bias_v = __half2float(pos_bias_v[h * head_dim + d]);

    q_u[idx] = __float2half(__half2float(qkv[q_base]) + bias_u);
    q_v[idx] = __float2half(__half2float(qkv[q_base]) + bias_v);
    k[idx] = qkv[k_base];
    v[idx] = qkv[v_base];
}

// Merge heads: [heads, tokens, head_dim] → [tokens, D] where D = heads * head_dim.
// out[t, h*head_dim + d] = inp[h, t, d]
extern "C" __global__ void merge_heads_f16(
    const __half* __restrict__ inp,  // [heads, tokens, head_dim]
    __half* __restrict__ out,        // [tokens, D]
    int tokens, int heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * tokens * head_dim;
    if (idx >= total) return;

    int D = heads * head_dim;
    int h = idx / (tokens * head_dim);
    int rem = idx % (tokens * head_dim);
    int t = rem / head_dim;
    int d = rem % head_dim;

    out[(size_t)t * D + (size_t)h * head_dim + d] = inp[idx];
}

// Reshape [tokens, D] → [heads, tokens, head_dim] where D = heads * head_dim.
// out[h, t, d] = inp[t, h*head_dim + d]
extern "C" __global__ void split_to_heads_f16(
    const __half* __restrict__ inp,  // [tokens, D]
    __half* __restrict__ out,        // [heads, tokens, head_dim]
    int tokens, int heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * tokens * head_dim;
    if (idx >= total) return;

    int D = heads * head_dim;
    int h = idx / (tokens * head_dim);
    int rem = idx % (tokens * head_dim);
    int t = rem / head_dim;
    int d = rem % head_dim;

    out[idx] = inp[(size_t)t * D + (size_t)h * head_dim + d];
}

// ============================================================================
// Stage 4 — decoder kernels
// ============================================================================

// Fused scale + causal mask + softmax over last dim of [heads, tokens, tokens].
// Upper triangle (j > i) is set to -inf before softmax. Scale is applied first.
// Expects input to be [heads, tokens, tokens] where tokens = q_len = k_len.
extern "C" __global__ void causal_softmax_f16(
    __half* __restrict__ y,        // [heads, tokens, tokens] — output
    const __half* __restrict__ x,  // [heads, tokens, tokens]
    int heads, int tokens,
    float scale
) {
    extern __shared__ float smem[];
    int row_idx = blockIdx.x;  // row = (head, q_pos)
    int total_rows = heads * tokens;
    if (row_idx >= total_rows) return;

    int h = row_idx / tokens;
    int q = row_idx % tokens;
    int tid = threadIdx.x;

    const __half* xr = x + (size_t)row_idx * tokens;
    __half* yr = y + (size_t)row_idx * tokens;

    // Pass 1: max (with causal mask — only consider j <= q)
    float local_max = -__int_as_float(0x7f800000);
    for (int j = tid; j <= q; j += blockDim.x) {
        float v = __half2float(xr[j]) * scale;
        local_max = fmaxf(local_max, v);
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float mx = smem[0];

    // Pass 2: exp + sum (causal — only j <= q)
    float local_sum = 0.0f;
    for (int j = tid; j <= q; j += blockDim.x) {
        float e = expf(__half2float(xr[j]) * scale - mx);
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];

    // Pass 3: write (causal — j <= q gets softmax, j > q gets 0)
    for (int j = tid; j < tokens; j += blockDim.x) {
        if (j <= q) {
            float e = expf(__half2float(xr[j]) * scale - mx);
            yr[j] = __float2half(e * inv_sum);
        } else {
            yr[j] = __float2half(0.0f);
        }
    }
}

// ReLU + bias broadcast: y[r, c] = relu(x[r, c] + bias[c]).
extern "C" __global__ void relu_bias_f16(
    __half* __restrict__ y,         // [rows, cols] — output
    const __half* __restrict__ x,   // [rows, cols]
    const __half* __restrict__ bias,// [cols]
    int numel, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int c = idx % cols;
    float v = __half2float(x[idx]) + __half2float(bias[c]);
    y[idx] = __float2half(v > 0.0f ? v : 0.0f);
}

// ============================================================================
// Stage 4.1 — KV cache kernels (incremental decoding)
// ============================================================================

// Scatter-write one token's KV into a pre-allocated cache buffer.
// inp: [heads, head_dim] (the new token's K or V)
// cache: [heads, max_seq, head_dim] — writes at row `pos`.
extern "C" __global__ void scatter_kv_f16(
    __half* __restrict__ cache,        // [heads, max_seq, head_dim]
    const __half* __restrict__ inp,    // [heads, head_dim]
    int pos, int max_seq, int heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * head_dim;
    if (idx >= total) return;
    int h = idx / head_dim;
    int i = idx % head_dim;
    cache[(size_t)h * max_seq * head_dim + (size_t)pos * head_dim + i] = inp[idx];
}

// Merge heads for a single token: [heads, 1, head_dim] → [head_dim*heads] contiguous.
// out[h*head_dim + i] = inp[h * stride + i]  (stride >= head_dim, handles cache buffers).
extern "C" __global__ void merge_heads_single_f16(
    const __half* __restrict__ inp,  // [heads, stride_seq, head_dim], 1 token at row 0
    __half* __restrict__ out,        // [heads * head_dim]
    int stride_seq, int heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * head_dim;
    if (idx >= total) return;
    int h = idx / head_dim;
    int i = idx % head_dim;
    out[(size_t)h * head_dim + i] = inp[(size_t)h * stride_seq * head_dim + i];
}

// Embedding gather: copy row `id` from a [vocab, dim] table → out[dim].
// Used for token + position embedding lookups (single token).
extern "C" __global__ void embed_gather_f16(
    __half* __restrict__ out,        // [dim]
    const __half* __restrict__ table,// [vocab, dim]
    int id, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    out[idx] = table[(size_t)id * dim + idx];
}

// Batch embedding: out[s, d] = token_emb[token_ids[s], d] + pos_emb[s, d].
// One thread per (s, d). Replaces per-token CPU gather + D2H of the whole
// embedding table on the prefill path.
extern "C" __global__ void embed_gather_batch_f16(
    __half* __restrict__ out,             // [seq, dim]
    const __half* __restrict__ token_emb, // [vocab, dim]
    const __half* __restrict__ pos_emb,   // [max_pos, dim] (pos 0..seq-1 used)
    const int* __restrict__ token_ids,    // [seq]
    int seq, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * dim;
    if (idx >= total) return;
    int d = idx % dim;
    int s = idx / dim;
    int tok = token_ids[s];
    out[idx] = __float2half(
        __half2float(token_emb[(size_t)tok * dim + d]) + __half2float(pos_emb[(size_t)s * dim + d])
    );
}

// Single-token fused gather+add: out[d] = token_emb[tok_id, d] + pos_emb[pos, d].
// Replaces the embed_one sequence of 2 gather kernels + 1 add kernel (3
// launches) with one. Followed by a separate LN kernel (needs a 2-pass reduce).
extern "C" __global__ void embed_gather_add_f16(
    __half* __restrict__ out,             // [dim]
    const __half* __restrict__ token_emb, // [vocab, dim]
    const __half* __restrict__ pos_emb,   // [max_pos, dim]
    int tok_id, int pos, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    out[idx] = __float2half(
        __half2float(token_emb[(size_t)tok_id * dim + idx]) + __half2float(pos_emb[(size_t)pos * dim + idx])
    );
}

// Sinusoidal position encoding on GPU. Matches the CPU reference in
// encoder.rs::generate_position_encoding:
//   for idx in 0..pos_len: position = tokens-1-idx
//     for dim in (0,2,4,...): div = exp(dim * (-ln(10000)/D))
//                              out[idx*D+dim]   = sin(position*div)
//                              out[idx*D+dim+1] = cos(position*div)
// One thread per output element. Replaces a host-side sin/cos loop + H2D upload.
extern "C" __global__ void position_encoding_f16(
    __half* __restrict__ out,   // [pos_len, D]
    int tokens, int pos_len, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pos_len * D;
    if (idx >= total) return;
    int d = idx % D;
    int row = idx / D;
    float position = (float)((int)tokens - 1 - (int)row);
    // d is even/odd; pair index = d/2. div uses the even index of the pair.
    int d_pair = (d / 2) * 2;
    float div = expf((float)d_pair * (-logf(10000.0f) / (float)D));
    float angle = position * div;
    out[idx] = __float2half((d & 1) ? cosf(angle) : sinf(angle));
}

// ============================================================================
// Stage 5.1 — pre-encoder conv stack (matches candle GPU f16 numerical path)
// ============================================================================

// Standard 2D conv, 3x3 kernel, stride 2, pad 1, groups=1, with ReLU.
// in: [N, Cin, H, W], weight: [Cout, Cin, 3, 3], bias: [Cout].
// out: [N, Cout, Hout, Wout], Hout=ceil(H/2), Wout=ceil(W/2).
// One thread per output element.
extern "C" __global__ void conv2d3x3_s2_relu_f16(
    __half* __restrict__ out,        // [N, Cout, Hout, Wout]
    const __half* __restrict__ in,   // [N, Cin, H, W]
    const __half* __restrict__ w,    // [Cout, Cin, 3, 3]
    const __half* __restrict__ bias, // [Cout]
    int N, int Cin, int Cout, int H, int W, int Hout, int Wout
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;
    // decompose idx -> (n, oc, oh, ow)
    int ow = idx % Wout;
    int rem = idx / Wout;
    int oh = rem % Hout;
    rem /= Hout;
    int oc = rem % Cout;
    int n = rem / Cout;

    float acc = __half2float(bias[oc]);
    int ih_base = oh * 2 - 1;  // pad=1
    int iw_base = ow * 2 - 1;
    for (int ic = 0; ic < Cin; ic++) {
        for (int kh = 0; kh < 3; kh++) {
            int ih = ih_base + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < 3; kw++) {
                int iw = iw_base + kw;
                if (iw < 0 || iw >= W) continue;
                float inv = __half2float(in[((size_t)n * Cin + ic) * H * W + ih * W + iw]);
                float wv = __half2float(w[((size_t)oc * Cin + ic) * 9 + kh * 3 + kw]);
                acc += inv * wv;
            }
        }
    }
    acc = acc > 0.0f ? acc : 0.0f;  // ReLU
    out[idx] = __float2half(acc);
}

// Depthwise 2D conv (groups = C), 3x3 kernel, stride 2, pad 1, no ReLU.
// in: [N, C, H, W], weight: [C, 1, 3, 3], bias: [C].
// out: [N, C, Hout, Wout]. Each channel convolved with its own filter.
extern "C" __global__ void depthwise_conv2d3x3_s2_f16(
    __half* __restrict__ out,        // [N, C, Hout, Wout]
    const __half* __restrict__ in,   // [N, C, H, W]
    const __half* __restrict__ w,    // [C, 1, 3, 3]
    const __half* __restrict__ bias, // [C]
    int N, int C, int H, int W, int Hout, int Wout
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Hout * Wout;
    if (idx >= total) return;
    int ow = idx % Wout;
    int rem = idx / Wout;
    int oh = rem % Hout;
    rem /= Hout;
    int c = rem % C;
    int n = rem / C;

    float acc = __half2float(bias[c]);
    int ih_base = oh * 2 - 1;
    int iw_base = ow * 2 - 1;
    size_t in_base = ((size_t)n * C + c) * H * W;
    for (int kh = 0; kh < 3; kh++) {
        int ih = ih_base + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < 3; kw++) {
            int iw = iw_base + kw;
            if (iw < 0 || iw >= W) continue;
            float inv = __half2float(in[in_base + ih * W + iw]);
            float wv = __half2float(w[(size_t)c * 9 + kh * 3 + kw]);
            acc += inv * wv;
        }
    }
    out[idx] = __float2half(acc);
}

// Pointwise (1x1) conv2d over channels + ReLU.
// in: [N, Cin, H, W], w: [Cout, Cin], bias: [Cout]. out: [N, Cout, H, W].
extern "C" __global__ void pointwise_conv_relu_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    const __half* __restrict__ w,
    const __half* __restrict__ bias,
    int N, int Cin, int Cout, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * H * W;
    if (idx >= total) return;
    int wpos = idx % W;
    int rem = idx / W;
    int hpos = rem % H;
    rem /= H;
    int oc = rem % Cout;
    int n = rem / Cout;
    size_t spatial = (size_t)H * W;
    float acc = __half2float(bias[oc]);
    const __half* in_n = in + (size_t)n * Cin * spatial + (size_t)hpos * W + wpos;
    for (int ic = 0; ic < Cin; ic++) {
        acc += __half2float(in_n[(size_t)ic * spatial]) * __half2float(w[(size_t)oc * Cin + ic]);
    }
    out[idx] = __float2half(acc > 0.0f ? acc : 0.0f);
}

// Reshape NCHW [1, C, T, F] → [T, C*F]. out[t, c*F + f] = in[c*T*F + t*F + f].
extern "C" __global__ void nchw_to_tokens_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int C, int T, int F
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * C * F;
    if (idx >= total) return;
    int f = idx % F;
    int rem = idx / F;
    int c = rem % C;
    int t = rem / C;
    out[(size_t)t * C * F + (size_t)c * F + f] = in[(size_t)c * T * F + (size_t)t * F + f];
}
extern "C" __global__ void split_qkv_step_cached_f16(
    const __half* __restrict__ qkv,    // [1, 3*D] (D = heads*head_dim)
    const __half* __restrict__ bias,   // [3*D]
    __half* __restrict__ q_out,        // [heads, head_dim]
    __half* __restrict__ k_cache,      // [heads, max_seq, head_dim]
    __half* __restrict__ v_cache,      // [heads, max_seq, head_dim]
    int pos, int max_seq, int heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * head_dim;
    if (idx >= total) return;
    int h = idx / head_dim;
    int d = idx % head_dim;
    int D = heads * head_dim;
    size_t qkv_off = (size_t)h * head_dim + d;        // q block base offset for (h,d)
    size_t k_base = (size_t)D + qkv_off;
    size_t v_base = (size_t)2 * D + qkv_off;
    size_t cache_off = (size_t)h * max_seq * head_dim + (size_t)pos * head_dim + d;

    q_out[idx] = __float2half(__half2float(qkv[qkv_off]) + __half2float(bias[qkv_off]));
    k_cache[cache_off] = __float2half(__half2float(qkv[k_base]) + __half2float(bias[k_base]));
    v_cache[cache_off] = __float2half(__half2float(qkv[v_base]) + __half2float(bias[v_base]));
}

// Batch QKV split + bias + K/V scatter for prefill.
// qkv:   [seq, 3*D]  (D = heads*head_dim)
// bias:  [3*D]
// q_out: [heads, seq, head_dim]   (row-major: h * seq*hd + s*hd + d)
// k/v_cache: [heads, max_seq, head_dim] — first `seq` rows written.
// One thread per (h, s, d) output element; each also writes K and V to cache.
extern "C" __global__ void split_qkv_batch_scatter_f16(
    const __half* __restrict__ qkv,     // [seq, 3*D]
    const __half* __restrict__ bias,    // [3*D]
    __half* __restrict__ q_out,         // [heads, seq, head_dim]
    __half* __restrict__ k_cache,       // [heads, max_seq, head_dim]
    __half* __restrict__ v_cache,       // [heads, max_seq, head_dim]
    int seq, int max_seq, int heads, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * seq * head_dim;
    if (idx >= total) return;
    int D = heads * head_dim;
    int h = idx / (seq * head_dim);
    int rem = idx % (seq * head_dim);
    int s = rem / head_dim;
    int d = rem % head_dim;

    // qkv layout: row s has [q-block | k-block | v-block], each D wide.
    size_t row = (size_t)s * 3 * D;
    size_t q_off = row + (size_t)h * head_dim + d;
    size_t k_off = row + (size_t)D + (size_t)h * head_dim + d;
    size_t v_off = row + (size_t)2 * D + (size_t)h * head_dim + d;
    size_t cache_off = (size_t)h * max_seq * head_dim + (size_t)s * head_dim + d;

    // bias is [3*D] (no token dim) — broadcast over all `seq` tokens. Index by
    // channel offset within the q/k/v block, NOT by the per-token q_off (which
    // would run past [3*D] for s >= 1).
    size_t bq = (size_t)h * head_dim + d;
    size_t bk = (size_t)D + (size_t)h * head_dim + d;
    size_t bv = (size_t)2 * D + (size_t)h * head_dim + d;

    q_out[idx] = __float2half(__half2float(qkv[q_off]) + __half2float(bias[bq]));
    k_cache[cache_off] = __float2half(__half2float(qkv[k_off]) + __half2float(bias[bk]));
    v_cache[cache_off] = __float2half(__half2float(qkv[v_off]) + __half2float(bias[bv]));
}

// ============================================================================
// Fused encoder attention scores: rel_shift(bd) + ac, scale, softmax in one.
// ac:  [heads, q_len, k_len]   (k_len == q_len for non-batched Conformer).
// bd:  [heads, q_len, pos_len] with pos_len = 2*k_len - 1.
// out: [heads, q_len, k_len] softmax over last dim. One block per (head, q).
// Replaces 4 separate kernels (rel_shift, add, scale, softmax) with one.

// Relative-position shift source value: for output (h, i_dst, j) read
// bd[h, i_dst, (k_len - 1 - i_dst + j)]. This is candle's
// FusedAttentionScoresShifted (src/app.rs:3182-3189), where the pos column
// (k_len-1-q+k) is always in [0, pos_len-1] for q,j in [0,k_len). bd rows have
// stride pos_len.
static __device__ float bd_shifted(const __half* bd, int h, int i_dst, int j,
                                   int q_len, int k_len, int pos_len) {
    int pos_col = k_len - 1 - i_dst + j;
    return __half2float(bd[((size_t)h * q_len + i_dst) * pos_len + pos_col]);
}

extern "C" __global__ void fused_attn_scores_softmax_f16(
    __half* __restrict__ out,        // [heads, q_len, k_len]
    const __half* __restrict__ ac,   // [heads, q_len, k_len]
    const __half* __restrict__ bd,   // [heads, q_len, pos_len], pos_len = 2*k_len-1
    int heads, int q_len, int k_len,
    float scale
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;  // = h * q_len + i_dst
    int total_rows = heads * q_len;
    if (row >= total_rows) return;
    int h = row / q_len;
    int i_dst = row % q_len;
    int tid = threadIdx.x;
    int dim = k_len;
    int pos_len = 2 * k_len - 1;

    const __half* acr = ac + (size_t)row * dim;
    __half* outr = out + (size_t)row * dim;

    // Pass 1: score[j] = (ac[j] + shifted_bd[j]) * scale, find max.
    float local_max = -__int_as_float(0x7f800000);
    for (int j = tid; j < dim; j += blockDim.x) {
        float s = (__half2float(acr[j]) + bd_shifted(bd, h, i_dst, j, q_len, k_len, pos_len)) * scale;
        local_max = fmaxf(local_max, s);
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float mx = smem[0];

    // Pass 2: exp + sum
    float local_sum = 0.0f;
    for (int j = tid; j < dim; j += blockDim.x) {
        float s = (__half2float(acr[j]) + bd_shifted(bd, h, i_dst, j, q_len, k_len, pos_len)) * scale;
        local_sum += expf(s - mx);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];

    // Pass 3: write
    for (int j = tid; j < dim; j += blockDim.x) {
        float s = (__half2float(acr[j]) + bd_shifted(bd, h, i_dst, j, q_len, k_len, pos_len)) * scale;
        outr[j] = __float2half(expf(s - mx) * inv_sum);
    }
}
// ============================================================================
// INT8 DP4A path (Stage 6) — quantize / dequant helpers for FFN GEMMs
// ============================================================================

// Reduce max(|a|) over n f16 elements into *out_max (stored as int bit-repr of
// the positive float, so atomicMax on int works for positive floats).
extern "C" __global__ void max_abs_reduce_f16(const __half* __restrict__ a, int n, int* out_max) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int local = 0;
    for (int i = tid; i < n; i += stride) {
        float v = fabsf(__half2float(a[i]));
        int iv = __float_as_int(v);
        if (iv > local) local = iv;
    }
    __shared__ int smem[1024];
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && smem[threadIdx.x + s] > smem[threadIdx.x]) smem[threadIdx.x] = smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(out_max, smem[0]);
}

// Quantize: aq[i] = round(a[i] * 127 / max). max read from max_buf (int bit-repr).
extern "C" __global__ void quantize_f16_i8(signed char* __restrict__ aq, const __half* __restrict__ a, int n, const int* max_buf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float mx = __int_as_float(*max_buf);
    float inv = (mx > 0.0f) ? (127.0f / mx) : 0.0f;
    float v = __half2float(a[i]) * inv;
    int q = (int)rintf(v);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    aq[i] = (signed char)q;
}

// Dequantize int32 GEMM output: out[i] = c_i32[i] * (max/127) * wt_inv[out_ch].
// max from max_buf (int bit-repr); wt_inv is per-output-channel (f16).
extern "C" __global__ void dequant_i32_f16(
    __half* __restrict__ out, const int* __restrict__ c_i32,
    const __half* __restrict__ wt_inv, const int* max_buf,
    int numel, int out_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    int out_ch = i % out_dim;
    float act_inv = __int_as_float(*max_buf) / 127.0f;
    float v = (float)c_i32[i] * act_inv * __half2float(wt_inv[out_ch]);
    out[i] = __float2half(v);
}

// ============================================================================
// Argmax over f16 logits → single int index. One block, strided load + shared
// mem reduction. For vocab=16384 a single 1024-thread block covers it in one
// stride loop. out[0] = argmax index (0-based within the `n` elements starting
// at `offset`). (Ties broken to lowest index, matching the CPU reference's
// strict `>` comparison.)
// ============================================================================
extern "C" __global__ void argmax_f16(
    int* __restrict__ out,        // [1]
    const __half* __restrict__ x, // logits buffer
    int offset,                   // start index of the window to reduce
    int n                         // number of elements to reduce
) {
    extern __shared__ float smem[];   // [2 * blockDim.x]: vals | idxs
    int* smem_idx = (int*)&smem[blockDim.x];

    int tid = threadIdx.x;
    float best_val = -__int_as_float(0x7f800000);
    int best_idx = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = __half2float(x[offset + i]);
        if (v > best_val) {   // strict > keeps lowest-index tie winner
            best_val = v;
            best_idx = i;
        }
    }
    smem[tid] = best_val;
    smem_idx[tid] = best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float va = smem[tid], vb = smem[tid + s];
            int ia = smem_idx[tid], ib = smem_idx[tid + s];
            // Prefer the larger value; on a strict tie keep the lower index.
            if (vb > va || (vb == va && ib < ia)) {
                smem[tid] = vb;
                smem_idx[tid] = ib;
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = smem_idx[0];
}
