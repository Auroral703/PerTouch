import os
import multiprocessing as mp
from utils.gen_param_maps import run_multi_gpu_generate
from utils.param_maps_norm import collect_global_stats, normalize_scores

def main():
    # Configuration settings
    BASE_CONFIG = {
        'CKPT_PATH': "../model/sam3/sam3.pt",
        'OUTPUT_DIR': 'train/parameter_maps',
        'NORM_DIR': 'train/parameter_maps_norm',
        'VERSION_DIRS': [
            'train/Expert/01-Experts-A',
            'train/Expert/02-Experts-B',
            'train/Expert/03-Experts-C',
            'train/Expert/04-Experts-D',
            'train/Expert/05-Experts-E'
        ],
        'NUM_GPUS': 2,
        'POINTS_PER_SIDE': 32,
        'MIN_AREA_THRESHOLD': 400,
        'TARGET_SIZE': (512, 512),
        'SAMPLE_PER_FILE': 1000,
    }

    # Execution stages
    print("[INFO] Starting Stage 1: SAM3 Generation")
    run_multi_gpu_generate(BASE_CONFIG)

    print("[INFO] Starting Stage 2: Statistics Collection")
    stats = collect_global_stats(
        root_dir=BASE_CONFIG['OUTPUT_DIR'],
        sample_per_file=BASE_CONFIG['SAMPLE_PER_FILE']
    )

    print("[INFO] Starting Stage 3: Normalization")
    normalize_scores(
        root_dir=BASE_CONFIG['OUTPUT_DIR'],
        dst_root_dir=BASE_CONFIG['NORM_DIR'],
        stats=stats,
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()