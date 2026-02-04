import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer

def load_scores(path):
    """Keep the original loading and cleaning logic"""
    data = np.load(path)['data'].astype(np.float32)
    data = np.nan_to_num(data)
    data = np.clip(data, -20000, 20000)
    return torch.from_numpy(data)

def collect_global_stats(root_dir, sample_per_file=1000):
    """
    Fully retain the original single-threaded statistics logic.
    Parameters passed from external data_preparation.py.
    """
    stats = {
        'min': torch.full((4,), float('inf')),
        'max': torch.full((4,), float('-inf')),
        'samples': {c: [] for c in range(4)}
    }

    file_count = 0
    # Iterate through all files
    for root, _, files in os.walk(root_dir):
        for f in tqdm(files, desc="Collecting stats"):
            if f.endswith('_parameter_map.npz'):
                data = load_scores(os.path.join(root, f))
                for c in range(4):
                    channel = data[c]
                    # Original logic: update min/max values
                    stats['min'][c] = min(stats['min'][c], channel.min())
                    stats['max'][c] = max(stats['max'][c], channel.max())
                    
                    arr = channel.flatten().numpy()
                    if arr.size > 0:
                        # Original logic: sampling points
                        idx = np.random.choice(arr.size, min(sample_per_file, arr.size), replace=False)
                        stats['samples'][c].append(arr[idx])
                file_count += 1

    # 拟合 QuantileTransformer 并计算 map/scale
    stats['qt'] = {}
    stats['zero_map'] = {}
    stats['scale'] = {}
    for c in range(4):
        samples = np.concatenate(stats['samples'][c])[:, None]  # shape (N,1)
        qt = QuantileTransformer(output_distribution='uniform', random_state=0)
        Qt = qt.fit(samples)
        
        # Original zero-point mapping logic
        zero_q = Qt.transform([[0.0]])[0, 0]
        scale = max(zero_q, 1 - zero_q)
        
        stats['qt'][c] = Qt
        stats['zero_map'][c] = zero_q
        stats['scale'][c] = scale

    print(f"[INFO] Processed {file_count} files; fitted QuantileTransformers for 4 channels.")
    print(f"[INFO] Global mins: {stats['min'].tolist()}\nGlobal maxs: {stats['max'].tolist()}")
    return stats

def normalize_scores(root_dir, dst_root_dir, stats):
    """
    Fully retain the original single-threaded normalization logic.
    root_dir: original data directory
    dst_root_dir: destination save directory
    """
    for root, _, files in os.walk(root_dir):
        for f in tqdm(files, desc="Normalizing with QuantileQT"):
            if f.endswith('_parameter_map.npz'):
                src_path = os.path.join(root, f)

                # Maintain relative path structure
                relative_path = os.path.relpath(root, root_dir)
                dst_dir = os.path.join(dst_root_dir, relative_path)
                os.makedirs(dst_dir, exist_ok=True)

                dst_path = os.path.join(dst_dir, f.replace('_parameter_map.npz', '_parameter_map_norm.npz'))

                if os.path.exists(dst_path):
                    continue

                data = load_scores(src_path)
                normalized = torch.zeros_like(data)

                for c in range(4):
                    arr = data[c].flatten().numpy()[:, None]
                    # Original conversion logic: u01 -> u_norm [-1, 1]
                    u01 = stats['qt'][c].transform(arr).flatten()
                    u_norm = 2 * u01 - 1
                    normalized[c] = torch.from_numpy(u_norm.reshape(data[c].shape)).to(data.dtype)

                score_np = normalized.cpu().numpy().astype(np.float16)
                np.savez_compressed(dst_path, data=score_np)