# Vitis HLS batch flow for RoFormer_Top, independent of the IDE workspace paths.
# Uses whatever model is installed in hls_code/ (scripts/install_export.py).
#
#   cd <anywhere outside the repo, e.g. ~/hls_runs>
#   vitis-run --mode hls --tcl <repo>/hls_code/scripts/run_hls.tcl          (Vitis 2023.2 and later)
#   vitis_hls -f <repo>/hls_code/scripts/run_hls.tcl                         (older Vitis HLS)
#
# Environment variables (optional):
#   MAC_UNROLL=4|8        multiply-accumulates per cycle per dense layer (default 4)
#   HLS_STEPS="csim csynth export"   add "cosim" for C/RTL co-simulation (slow: tens of minutes)
#   LUTRAM_ATTN_BUFFERS=0 keep the attention buffers in block RAM
#   HLS_CLOCK_NS=10       clock period (100 MHz, the ZedBoard design's FCLK0)
# Outputs: ./roformer_hls/sol1/syn/report/RoFormer_Top_csynth.rpt and ./RoFormer_Top_ip.zip
set here [file normalize [file dirname [info script]]]
set src  [file normalize "$here/.."]
proc envor {name default} { if {[info exists ::env($name)]} { return $::env($name) } { return $default } }
set unroll [envor MAC_UNROLL 4]
set steps  [envor HLS_STEPS "csim csynth export"]
set lutram [envor LUTRAM_ATTN_BUFFERS 1]
set period [envor HLS_CLOCK_NS 10]
set cflags "-I$src -DMAC_UNROLL=$unroll -DLUTRAM_ATTN_BUFFERS=$lutram"
puts "RoFormer_Top from $src | MAC_UNROLL=$unroll LUTRAM_ATTN_BUFFERS=$lutram clock ${period} ns | steps: $steps"

open_project -reset roformer_hls
set_top RoFormer_Top
add_files $src/RoFormer_Top.cpp -cflags $cflags
add_files -tb $src/tests/test_vitis_roformer.cpp -cflags "$cflags -I$src/tests"
open_solution -reset sol1 -flow_target vivado
set_part xc7z020clg484-1
create_clock -period $period -name default
set_clock_uncertainty 12.5%

if {[lsearch $steps csim]   >= 0} { csim_design -clean }
if {[lsearch $steps csynth] >= 0} { csynth_design }
if {[lsearch $steps cosim]  >= 0} { cosim_design -trace_level none }
if {[lsearch $steps export] >= 0} { export_design -format ip_catalog -output [file normalize "./RoFormer_Top_ip.zip"] }
exit
