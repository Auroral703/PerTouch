import os
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
import multiprocessing as mp

# Relative imports
from .cal_metrics import (
    image_colorfulness,
    contrastpro,
    image_tonepropro,
    image_brightness
)

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def post_process_refined(raw_map, min_area):
    h, w = raw_map.shape
    unique_ids = np.unique(raw_map)
    cleaned_map = np.zeros_like(raw_map)
    kernel = np.ones((3, 3), np.uint8)
    for inst_id in unique_ids:
        if inst_id == 0: continue
        mask = (raw_map == inst_id).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if np.sum(mask) >= min_area:
            cleaned_map[mask > 0] = inst_id
    void_mask = (cleaned_map == 0)
    # if np.any(void_mask) and np.any(cleaned_map > 0):
    #     _, indices = distance_transform_edt(void_mask, return_indices=True)
    #     return cleaned_map[indices[0], indices[1]]
    return cleaned_map

def worker(gpu_id, image_list, config):
    """
    Pass parameters via config to ensure multiprocessing safety
    """
    # 1. Explicitly set GPU device
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    # 2. Extract parameters from config (key fix: don't use global variables directly)
    ckpt_path = config['CKPT_PATH']
    output_dir = config['OUTPUT_DIR']
    pts_per_side = config['POINTS_PER_SIDE']
    min_area = config['MIN_AREA_THRESHOLD']
    target_size = config['TARGET_SIZE']
    version_dirs = config['VERSION_DIRS']

    # 3. Initialize model
    model = build_sam3_image_model(checkpoint_path=ckpt_path, enable_inst_interactivity=True).to(device)
    processor = Sam3Processor(model)
    model.eval()

    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, pts_per_side), np.linspace(0, 1, pts_per_side))
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

    pbar = tqdm(image_list, desc=f"GPU {gpu_id}", position=gpu_id)

    for img_name in pbar:
        # 4. 加载图片组
        versions = []
        for d in version_dirs:
            path = os.path.join(d, img_name)
            versions.append(Image.open(path).convert('RGB'))

        for version_idx, img_pil in enumerate(versions):
            # --- Keep the following logic unchanged ---
            w, h = img_pil.size
            inference_state = processor.set_image(img_pil)
            input_points = (grid_points * np.array([w, h])).astype(np.float32)
            input_labels = np.ones(len(input_points), dtype=np.int32)

            with torch.no_grad():
                masks, scores, ious = model.predict_inst(
                    inference_state,
                    point_coords=input_points[:, None, :],
                    point_labels=input_labels[:, None],
                    multimask_output=False,
                )
                if not torch.is_tensor(masks): masks = torch.as_tensor(masks, device=device)
                if not torch.is_tensor(scores): scores = torch.as_tensor(scores, device=device)
                
                masks = masks.squeeze().float()
                scores = scores.squeeze().float()

                bg_score = torch.full((1, h, w), 1e-4, device=device)
                score_volumes = masks * scores.view(-1, 1, 1)
                combined_scores = torch.cat([bg_score, score_volumes], dim=0)
                raw_class_map = torch.argmax(combined_scores, dim=0).cpu().numpy()

            refined_map = post_process_refined(raw_class_map, min_area)
            
            img_np = np.array(img_pil)
            resized_img = cv2.resize(img_np, target_size)
            img_tensor = torch.from_numpy(resized_img[..., ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)

            unique_instances = np.unique(refined_map)
            masks_resized = []
            for inst_id in unique_instances:
                m_res = cv2.resize((refined_map == inst_id).astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
                masks_resized.append(m_res.astype(bool))
            
            n_reg = len(masks_resized)
            score_tensor = torch.zeros((4,) + target_size, device=device)

            if n_reg > 0:
                imgs_batch = img_tensor.repeat(n_reg, 1, 1, 1)
                masks_batch = torch.tensor(np.stack(masks_resized), dtype=torch.bool, device=device)

                with torch.no_grad():
                    cf_scores = image_colorfulness(imgs_batch, masks_batch).float()
                    ct_scores = contrastpro(imgs_batch, masks_batch).float()
                    tp_scores = image_tonepropro(imgs_batch, masks_batch).float()
                    br_scores = image_brightness(imgs_batch, masks_batch).float()

                for i in range(n_reg):
                    m_i = masks_batch[i]
                    score_tensor[0][m_i] = cf_scores[i]
                    score_tensor[1][m_i] = ct_scores[i]
                    score_tensor[2][m_i] = tp_scores[i]
                    score_tensor[3][m_i] = br_scores[i]

            # 5. 动态保存路径
            base_name = os.path.splitext(img_name)[0]
            v_folder = version_dirs[version_idx].rstrip('/').split('/')[-1]
            cat_dir = version_dirs[version_idx].rstrip('/').split('/')[-2]
            output_subdir = os.path.join(output_dir, cat_dir, v_folder)
            os.makedirs(output_subdir, exist_ok=True)
            
            # 使用 clamp 保证 float16 安全
            score_np = torch.clamp(score_tensor, -30000, 30000).cpu().numpy().astype(np.float16)
            np.savez_compressed(os.path.join(output_subdir, f"{base_name}_parameter_map.npz"), data=score_np)

def run_multi_gpu_generate(config):
    """
    Fixed task distribution logic
    """
    all_images = sorted(os.listdir(config['VERSION_DIRS'][0]))
    num_gpus = config['NUM_GPUS']
    num_images = len(all_images)
    
    # Use more robust chunking method
    chunks = []
    for i in range(num_gpus):
        # 计算每个进程的起始和结束索引
        start_idx = i * num_images // num_gpus
        end_idx = (i + 1) * num_images // num_gpus
        chunks.append(all_images[start_idx:end_idx])

    # Print for verification
    for i, chunk in enumerate(chunks):
        print(f"[INFO] GPU {i} assigned {len(chunk)} images.")

    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=worker, args=(i, chunks[i], config))
        p.start()
        processes.append(p)
    for p in processes: 
        p.join()