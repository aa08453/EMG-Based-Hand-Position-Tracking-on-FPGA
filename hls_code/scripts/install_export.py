"""
install_export.py
-----------------
Copy an HLS export (emg_common.export_hls: notebook 05 or 09 output, unzipped) into the HLS project,
replacing the model it currently holds. Everything the C simulation, Vitis and the firmware need comes
from that one export, so nothing from an older model can be left behind.

  hls_code/model_config.h, rope_tables.h                    model dimensions, fixed-point formats, RoPE tables
  hls_code/weights/*.h                                      weights, golden results, test input
  hls_code/verification_data/                               golden intermediates, fixed-point expectation
  hil_verification_samples/                                 real windows with float and fixed-point outputs
  vitis_workspace/uart_app/src/normalization_constants.h    firmware constants (EMG scale, filter, I/O sizes)

Usage (from the repository root or anywhere):
  python hls_code/scripts/install_export.py path/to/hls_export/<run_name>
  python hls_code/scripts/install_export.py EXPORT --hls /tmp/hls_copy --no-firmware   # install into a copy
Then: cd hls_code && make clean && make run
"""
import argparse
import glob
import os
import re
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WEIGHT_FILES = ('conv1d_weights.h', 'encoder_weights.h', 'fc_weights.h', 'results.h', 'input_output.h')
STALE = ('l0_mha_weights.h', 'l0_ln_weights.h', 'l0_ff_weights.h')    # single-layer layout, replaced by encoder_weights.h


def find_export(path):
    hits = sorted(glob.glob(os.path.join(path, '**', 'model_config.h'), recursive=True))
    if not hits:
        sys.exit(f'{path}: no model_config.h found. Exports made before the multi-layer HLS (config.h + l0_*.h) '
                 'cannot be installed; re-export the checkpoint with the current emg_common.py.')
    if len(hits) > 1:
        sys.exit('several exports found, pass one of:\n  ' + '\n  '.join(os.path.dirname(h) for h in hits))
    return os.path.dirname(hits[0])


def replace_dir_files(src, dst, pattern='*'):
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(dst, pattern)):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(os.path.join(src, pattern)):
        shutil.copy2(f, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('export_dir')
    ap.add_argument('--hls', default=os.path.join(REPO, 'hls_code'))
    ap.add_argument('--hil', default=None, help='default: <hls>/../hil_verification_samples')
    ap.add_argument('--firmware', default=os.path.join(REPO, 'vitis_workspace', 'uart_app', 'src'))
    ap.add_argument('--no-firmware', action='store_true')
    a = ap.parse_args()

    ex = find_export(os.path.abspath(a.export_dir))
    hls = os.path.abspath(a.hls)
    hil = os.path.abspath(a.hil or os.path.join(hls, '..', 'hil_verification_samples'))
    for f in WEIGHT_FILES:
        if not os.path.exists(os.path.join(ex, 'weights', f)):
            sys.exit(f'{ex}/weights/{f} missing: incomplete export')

    shutil.copy2(os.path.join(ex, 'model_config.h'), hls)
    shutil.copy2(os.path.join(ex, 'rope_tables.h'), hls)
    for f in WEIGHT_FILES:
        shutil.copy2(os.path.join(ex, 'weights', f), os.path.join(hls, 'weights'))
    for f in STALE:
        p = os.path.join(hls, 'weights', f)
        if os.path.exists(p):
            os.remove(p)
    vd = os.path.join(hls, 'verification_data')
    replace_dir_files(os.path.join(ex, 'verification_data'), vd)
    for f in ('fixed_point_expectation.json', 'normalization_constants.h', 'normalization_constants.json', 'EXPORT_INFO.json'):
        if os.path.exists(os.path.join(ex, f)):
            shutil.copy2(os.path.join(ex, f), vd)
    if os.path.isdir(os.path.join(ex, 'hil_verification_samples')):
        replace_dir_files(os.path.join(ex, 'hil_verification_samples'), hil)
    if not a.no_firmware and os.path.exists(os.path.join(ex, 'normalization_constants.h')):
        shutil.copy2(os.path.join(ex, 'normalization_constants.h'), a.firmware)

    cfg = open(os.path.join(hls, 'model_config.h')).read()
    get = lambda k: (re.search(rf'#define {k} (\S+)', cfg) or [None, '?'])[1]
    print(f'installed {ex}')
    print(f'  layers {get("NUM_LAYERS")}, window {get("RAW_SEQ_LEN")} samples -> {get("SEQ_LEN")} tokens (kernel/stride {get("KERNEL_SIZE")}), '
          f'D_FF {get("D_FF")}, delta head {get("HAS_DELTA_HEAD")}')
    print(f'  formats: act ({get("ACT_W")},{get("ACT_I")}) weight ({get("WEIGHT_W")},{get("WEIGHT_I")}) resid ({get("RESID_W")},{get("RESID_I")})')
    print(f'  HIL samples -> {hil}' + ('' if a.no_firmware else f'\n  firmware constants -> {a.firmware}'))
    print(f'next: cd {hls} && make clean && make run')


if __name__ == '__main__':
    main()
