typedef ap_fixed<10, 3, AP_RND, AP_SAT> data_t;
typedef ap_fixed<10, 3, AP_RND, AP_SAT> weight_t;
typedef ap_fixed<8, 2, AP_RND, AP_SAT> rope_t;
typedef ap_fixed<16, 8, AP_RND, AP_SAT> resid_t;  // branch outputs + pre-norm residual sums (ff_out, ln_in)
