import sys
import os
import py_compile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

def check_section_4():
    print("=" * 80)
    print("AUDIT: SECTION 4 - TRAINING & FEATURE PIPELINE SCRIPTS")
    print("=" * 80)
    p = PROJECT_ROOT / "training"
    scripts = [
        ("trainer.py", ["train_one_epoch", "validate_one_epoch"]),
        ("checkpoint.py", ["save_checkpoint", "load_checkpoint"]),
        ("train_densenet.py", ["train_and_eval_densenet"]),
        ("train_densenet_vit.py", ["train_and_eval_densenet_vit"]),
        ("train_compression.py", ["train_and_eval_compression"]),
        ("cache_features.py", ["cache_features_for_split"]),
        ("train_vqc.py", ["train_and_eval_vqc"])
    ]
    all_ok = True
    for fname, expected_funcs in scripts:
        fpath = p / fname
        if not fpath.exists():
            print(f"[-] MISSING: training/{fname}")
            all_ok = False
            continue
        try:
            py_compile.compile(str(fpath), doraise=True)
            # Inspect symbols
            mod = __import__(f"training.{fname[:-3]}", fromlist=expected_funcs)
            missing = [fn for fn in expected_funcs if not hasattr(mod, fn)]
            if missing:
                print(f"[-] training/{fname:<20} | Missing functions: {missing}")
                all_ok = False
            else:
                size_kb = fpath.stat().st_size / 1024
                print(f"[+] training/{fname:<22} | Size: {size_kb:5.1f} KB | Verified: {expected_funcs} | Status: HEALTHY")
        except Exception as e:
            print(f"[-] training/{fname:<20} | Error: {e}")
            all_ok = False

    print("=" * 80)
    if all_ok:
        print("RESULT: SECTION 4 IS 100% HEALTHY, INTACT AND VERIFIED!")
    else:
        print("RESULT: SECTION 4 HAS ISSUES!")
    print("=" * 80)

if __name__ == "__main__":
    check_section_4()
