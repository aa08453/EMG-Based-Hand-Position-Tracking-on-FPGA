#ifndef FCLAYER_H
#define FCLAYER_H
#include "types.h"

// Final FC layer: data_t input, data_t weights, data_t output
void linear(const data_t input[], data_t output[],
            const data_t weights[], const data_t bias[],
            int in_dim, int out_dim)
{
#pragma HLS INLINE off
    FC_OUTPUT:
    for (int i = 0; i < out_dim; ++i) {
        mha_sum_t sum = (mha_sum_t)bias[i];
        FC_MAC:
        for (int j = 0; j < in_dim; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS BIND_OP variable=sum op=mul impl=dsp
#pragma HLS BIND_OP variable=sum op=add impl=dsp
            sum += (mha_sum_t)weights[i*in_dim+j] * (mha_sum_t)input[j];
        }
        output[i] = (data_t)sum;
    }
}

void linear2(const data_t input[], data_t output[],
            const data_t weights[], const data_t bias[],
            int in_dim, int out_dim)
{
#pragma HLS INLINE off
    FC_OUTPUT2:
    for (int i = 0; i < out_dim; ++i) {
        mha_sum_t sum = (mha_sum_t)bias[i];
        FC_MAC2:
        for (int j = 0; j < in_dim; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS BIND_OP variable=sum op=mul impl=dsp
#pragma HLS BIND_OP variable=sum op=add impl=dsp
            sum += (mha_sum_t)weights[i*in_dim+j] * (mha_sum_t)input[j];
        }
        output[i] = (data_t)sum;
    }
}

// MHA projection: data_t input → mha_sum_t output (Q, K, V from conv1d tokens)
void linear_wide1(const data_t input[], mha_sum_t output[],
                 const data_t weights[], const data_t bias[],
                 int in_dim, int out_dim)
{
#pragma HLS INLINE off
    FC_MHA_CONV_OUTPUT:
    for (int i = 0; i < out_dim; ++i) {
        mha_sum_t sum = (mha_sum_t)bias[i];
        FC_MHA_CONV_MAC:
        for (int j = 0; j < in_dim; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS BIND_OP variable=sum op=mul impl=dsp
#pragma HLS BIND_OP variable=sum op=add impl=dsp
            sum += (mha_sum_t)weights[i*in_dim+j] * (mha_sum_t)input[j];
        }
        output[i] = sum;   // no cast — keep full mha_sum_t precision
    }
}

// O projection: data_t input (context vector) → mha_sum_t output
void linear_wide2(const data_t input[], mha_sum_t output[],
                 const data_t weights[], const data_t bias[],
                 int in_dim, int out_dim)
{
#pragma HLS INLINE off
    FC_MHA_MHA_OUTPUT:
    for (int i = 0; i < out_dim; ++i) {
        mha_sum_t sum = (mha_sum_t)bias[i];
        FC_MHA_MHA_MAC:
        for (int j = 0; j < in_dim; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS BIND_OP variable=sum op=mul impl=dsp
#pragma HLS BIND_OP variable=sum op=add impl=dsp
            sum += (mha_sum_t)weights[i*in_dim+j] * (mha_sum_t)input[j];
        }
        output[i] = sum;   // no cast
    }
}

void linear_wide3(const mha_sum_t input[], mha_sum_t output[],
                 const data_t weights[], const data_t bias[],
                 int in_dim, int out_dim)
{
#pragma HLS INLINE off
    FC_MHA_MHA_OUTPUT2:
    for (int i = 0; i < out_dim; ++i) {
        mha_sum_t sum = (mha_sum_t)bias[i];
        FC_MHA_MHA_MAC2:
        for (int j = 0; j < in_dim; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS BIND_OP variable=sum op=mul impl=dsp
#pragma HLS BIND_OP variable=sum op=add impl=dsp
            sum += (mha_sum_t)weights[i*in_dim+j] * (mha_sum_t)input[j];
        }
        output[i] = sum;   // no cast
    }
}


#endif // FCLAYER_H