#ifndef CONFIG_DEBUG_H
#define CONFIG_DEBUG_H

// Uncomment exactly ONE of these to select the debug checkpoint:
// #define DEBUG_CONV1D   // output: [SEQ_LEN * OUT_CHANNELS]  = 200*128 = 25600
// #define DEBUG_MHA      // output: [SEQ_LEN * D_MODEL]        = 200*128 = 25600
// #define DEBUG_FFN      // output: [SEQ_LEN * D_MODEL]        = 200*128 = 25600
// #define DEBUG_POOL     // output: [D_MODEL]                  = 128
#define DEBUG_FULL     // output: [OUT_DIM]                  = 10  (normal run)

// Derived: output element count for each checkpoint
#if   defined(DEBUG_CONV1D)
  #define DEBUG_OUT_LEN (SEQ_LEN * OUT_CHANNELS)
  #define DEBUG_LABEL   "conv1d"
#elif defined(DEBUG_MHA)
  #define DEBUG_OUT_LEN (SEQ_LEN * D_MODEL)
  #define DEBUG_LABEL   "mha"
#elif defined(DEBUG_FFN)
  #define DEBUG_OUT_LEN (SEQ_LEN * D_MODEL)
  #define DEBUG_LABEL   "ffn"
#elif defined(DEBUG_POOL)
  #define DEBUG_OUT_LEN (D_MODEL)
  #define DEBUG_LABEL   "pool"
#else
  #define DEBUG_OUT_LEN (OUT_DIM)
  #define DEBUG_LABEL   "full"
#endif

#endif // CONFIG_DEBUG_H