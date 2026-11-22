#ifndef TOP_CALL_H
#define TOP_CALL_H
// run_top(in, out): one call of RoFormer_Top with array input/output, for either top-level interface.
//   TOP_AXIS_STREAM 1: AXI-Stream ports (hls::stream<data_t> in, out), the dataflow design
//   TOP_AXIS_STREAM 0: memory-mapped ports (data_t in[IN_WORDS], out[NUM_OUTPUTS]), the layer-loop design
// Set in config.h; it must match the signature in RoFormer_Top.cpp.
#include "../types.h"
#include "../config.h"
#ifndef TOP_AXIS_STREAM
#define TOP_AXIS_STREAM 0
#endif

#if TOP_AXIS_STREAM
#include <hls_stream.h>
void RoFormer_Top(hls::stream<data_t>& input_stream, hls::stream<data_t>& output_stream);
static void run_top(const data_t in[IN_WORDS], data_t out[NUM_OUTPUTS]) {
    hls::stream<data_t> s_in("input_stream"), s_out("output_stream");
    for (int i = 0; i < IN_WORDS; ++i) s_in.write(in[i]);
    RoFormer_Top(s_in, s_out);
    for (int i = 0; i < NUM_OUTPUTS; ++i) out[i] = s_out.read();
}
#else
void RoFormer_Top(const data_t input_axi[IN_WORDS], data_t output_axi[NUM_OUTPUTS]);
static void run_top(const data_t in[IN_WORDS], data_t out[NUM_OUTPUTS]) { RoFormer_Top(in, out); }
#endif
#endif // TOP_CALL_H
