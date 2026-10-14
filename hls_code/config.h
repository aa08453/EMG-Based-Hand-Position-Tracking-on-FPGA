#ifndef CONFIG_H
#define CONFIG_H
// v2 model: checkpoint patch_both_prev0_K0_qat5_v2.pt (roformer_output_v2/hls_export/.../config.h)
#define OUT_DIM 10
#define SEQ_LEN 25            // tokens after the strided embedding
#define RAW_SEQ_LEN 200       // EMG samples per window
#define D_MODEL 128
#define IN_CHANNELS 12
#define OUT_CHANNELS 128
#define KERNEL_SIZE 8
#define CONV_STRIDE 8
#define CONV_PADDING 0
#define NUM_HEADS 8
#define HEAD_DIM 16
#define D_FF 256
#define ROPE_HALF_DIM 64
#define NUM_LAYERS 1
#define USE_ATTENTION 1
#define CAUSAL_ATTENTION 0
#define NORM_IS_RMS 0
#define POOL_LAST_TOKEN 0
#define HAS_ABS_HEAD 1
#define HAS_DELTA_HEAD 1
#define COND_PREV 0

// Hardware parallelism: multiply-accumulates per cycle in each dense layer.
// 4 -> about 10 ms per inference at 100 MHz (estimate); 8 -> about 5 ms at roughly twice the DSPs.
#ifndef MAC_UNROLL
#define MAC_UNROLL 4
#endif

#define NUM_OUTPUTS (OUT_DIM * (HAS_ABS_HEAD + HAS_DELTA_HEAD))  // [abs(10) | delta_std(10)]
#define IN_WORDS (IN_CHANNELS * RAW_SEQ_LEN)

#if COND_PREV || CAUSAL_ATTENTION || NORM_IS_RMS || POOL_LAST_TOKEN || !USE_ATTENTION || !HAS_ABS_HEAD || NUM_LAYERS != 1
#error "This HLS build implements: 1 layer, attention, LayerNorm, mean pooling, abs head (+ optional delta head), no prev-angle input"
#endif
#if (D_MODEL % MAC_UNROLL) || (D_FF % MAC_UNROLL) || (HEAD_DIM % MAC_UNROLL && MAC_UNROLL < HEAD_DIM)
#error "MAC_UNROLL must divide D_MODEL, D_FF and HEAD_DIM"
#endif
#endif // CONFIG_H
