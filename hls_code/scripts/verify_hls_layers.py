import argparse
import os
import platform
import subprocess
import sys
from typing import List, Optional


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _to_wsl_path(win_path: str) -> str:
    """
    Best-effort conversion for Windows local paths like:
      d:\hls_code  -> /mnt/d/hls_code
    """
    win_path = os.path.abspath(win_path)
    drive = win_path[0].lower()
    rest = win_path[2:].replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def _run(cmd: List[str], *, cwd: Optional[str] = None) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    repo = _repo_root()

    # Import "roformer.py" utilities from the scripts folder.
    sys.path.insert(0, os.path.join(repo, "scripts"))
    import roformer  # type: ignore

    parser = argparse.ArgumentParser(
        description="Dump and verify RoFormer HLS layer outputs layer-by-layer."
    )
    parser.add_argument(
        "--verification_dir",
        default=roformer.DEFAULT_VERIFICATION_DIR,
        help="Golden verification_data directory.",
    )
    parser.add_argument(
        "--hw_dir",
        default=os.path.join(repo, "vitis_outputs"),
        help="Directory where the C++ dump writes outputs.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the C++ dumper binary before running.",
    )
    parser.add_argument(
        "--skip_c",
        action="store_true",
        help="Skip running the C++ dumper; only compare HW dumps to goldens.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop comparison after the first failure.",
    )
    parser.add_argument(
        "--run-vitis",
        action="store_true",
        help="Also compile and run test_vitis_roformer.cpp with RoFormer_Top.cpp (Linux only).",
    )
    args = parser.parse_args()

    os.makedirs(args.hw_dir, exist_ok=True)

    if not args.skip_c:
        if platform.system() == 'Linux':
            # On Linux, run commands directly
            if args.run_vitis:
                print("\n--- Running test_vitis_roformer.cpp ---")
                vitis_binary = os.path.join(repo, "test_vitis")
                if args.rebuild or not os.path.exists(vitis_binary):
                    vitis_compile_cmd = (
                        f'g++ -std=c++14 -O2 -DUSE_MOCK_HLS -I. -I./tests '
                        f'-o test_vitis tests/test_vitis_roformer.cpp RoFormer_Top.cpp'
                    )
                    print(f"\n$ bash -c '{vitis_compile_cmd}'")
                    subprocess.run(["bash", "-c", vitis_compile_cmd], cwd=repo, check=True)

                # Run the test_vitis binary
                vitis_run_cmd = './test_vitis'
                print(f"\n$ bash -c '{vitis_run_cmd}'")
                subprocess.run(["bash", "-c", vitis_run_cmd], cwd=repo, check=True)
            else:
                out_dir_rel = os.path.basename(args.hw_dir)
                binary = os.path.join(repo, "test_vitis_roformer_all_layers")
                if args.rebuild or not os.path.exists(binary):
                    compile_cmd = (
                        f'mkdir -p ./vitis_outputs && '
                        f'g++ -std=c++14 -O2 -DUSE_MOCK_HLS -I. -I./tests '
                        f'-o test_vitis_roformer_all_layers '
                        f'tests/test_vitis_roformer_all_layers.cpp'
                    )
                    print(f"\n$ bash -c '{compile_cmd}'")
                    subprocess.run(["bash", "-c", compile_cmd], cwd=repo, check=True)

                # Run the dumper to produce vitis_outputs/*.txt
                run_cmd = f'./test_vitis_roformer_all_layers "{out_dir_rel}"'
                print(f"\n$ bash -c '{run_cmd}'")
                subprocess.run(["bash", "-c", run_cmd], cwd=repo, check=True)

            # Optionally compile and run test_vitis_roformer.cpp

        else:
            # On Windows with WSL
            wsl_repo = _to_wsl_path(repo)
            out_dir_rel = os.path.basename(args.hw_dir)

            binary = os.path.join(repo, "test_vitis_roformer_all_layers")
            if args.rebuild or not os.path.exists(binary):
                compile_cmd = (
                    f'cd "{wsl_repo}" && '
                    f'mkdir -p ./vitis_outputs && '
                    f'g++ -std=c++14 -O2 -DUSE_MOCK_HLS -I. -I./tests '
                    f'-o test_vitis_roformer_all_layers '
                    f'tests/test_vitis_roformer_all_layers.cpp'
                )
                _run(["wsl", "bash", "-lc", compile_cmd])

            # Run the dumper to produce vitis_outputs/*.txt
            run_cmd = f'cd "{wsl_repo}" && ./test_vitis_roformer_all_layers "{out_dir_rel}"'
            _run(["wsl", "bash", "-lc", run_cmd])

    # Compare HW dumps to golden verification_data outputs
    roformer.verify_layer_outputs(
        args.verification_dir,
        args.hw_dir,
        fail_fast=args.fail_fast,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

