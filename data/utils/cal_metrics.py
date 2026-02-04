import os
import time
from pprint import pprint

import colour
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def image_colorfulness(masked_images_batch, masks_batch):
    """
    Compute colorfulness of the masked region of each image in the batch.
    Args:
        masked_images_batch (torch.Tensor): A batch of images with shape (n, c, h, w) in BGR format.
        masks_batch (torch.Tensor): A batch of masks with shape (n, h, w), where 1 indicates the region to calculate.
    Returns:
        torch.Tensor: Colorfulness values for each image in the batch.
    """
    device = masked_images_batch.device
    n, c, h, w = masked_images_batch.shape

    masked_images_batch = masked_images_batch.permute(0, 2, 3, 1).cpu().numpy()  # Shape: (n, h, w, 3)
    masked_images_batch = masked_images_batch.astype("float32")
    masks_batch = masks_batch.cpu().numpy().astype(bool)  # Shape: (n, h, w)

    colorfulness_values = []

    for img, mask in zip(masked_images_batch, masks_batch):
        if not np.any(mask):
            colorfulness_values.append(0)  # Assign 0 brightness if mask is empty
            continue

        mask_3d = np.stack([mask] * 3, axis=-1)  # Shape: (h, w, 3)
        img_masked = np.where(mask_3d, img, np.nan)  # Set non-masked regions to NaN

        (B, G, R) = cv2.split(img_masked)

        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)

        rg_mean = np.nanmean(rg)
        rg_std = np.nanstd(rg)
        yb_mean = np.nanmean(yb)
        yb_std = np.nanstd(yb)

        std_root = np.sqrt((rg_std ** 2) + (yb_std ** 2))
        mean_root = np.sqrt((rg_mean ** 2) + (yb_mean ** 2))

        colorfulness_values.append(std_root + (0.3 * mean_root))

    return torch.tensor(colorfulness_values, device=device)


def image_brightness(masked_images_batch, masks_batch):
    """
    Compute brightness of the masked region of each image in the batch.
    """
    device = masked_images_batch.device
    n, c, h, w = masked_images_batch.shape

    masked_images_batch = masked_images_batch.permute(0, 2, 3, 1).cpu().numpy()  # Shape: (n, h, w, 3)
    masked_images_batch = masked_images_batch.astype("float32")
    masks_batch = masks_batch.cpu().numpy().astype(bool)  # Shape: (n, h, w)

    brightness_values = []

    for img, mask in zip(masked_images_batch, masks_batch):
        if not np.any(mask):
            brightness_values.append(0)  # Assign 0 brightness if mask is empty
            continue

        mask_3d = np.stack([mask] * 3, axis=-1)  # Shape: (h, w, 3)
        img_masked = np.where(mask_3d, img, np.nan)  # Set non-masked regions to NaN

        (B, G, R) = cv2.split(img_masked)

        r_mean = np.nanmean(R)
        g_mean = np.nanmean(G)
        b_mean = np.nanmean(B)

        brightness = np.sqrt(0.241 * (r_mean ** 2) + 0.691 * (g_mean ** 2) + 0.068 * (b_mean ** 2))
        brightness_values.append(brightness)

    # Return as a tensor
    return torch.tensor(brightness_values, device=device)

def image_tonepropro(masked_images_batch, masks_batch):
    """
    Compute color temperature (CCT) of the masked region of each image in the batch.
    """
    device = masked_images_batch.device
    n, c, h, w = masked_images_batch.shape

    masked_images_batch = masked_images_batch.permute(0, 2, 3, 1).cpu().numpy()  # Shape: (n, h, w, 3)
    masked_images_batch = masked_images_batch.astype("float32")
    masks_batch = masks_batch.cpu().numpy().astype(bool)  # Shape: (n, h, w)

    tone_values = []

    for img, mask in zip(masked_images_batch, masks_batch):
        if not np.any(mask):
            tone_values.append(0)  # Assign 0 brightness if mask is empty
            continue

        mask_3d = np.stack([mask] * 3, axis=-1)  # Shape: (h, w, 3)
        img_masked = np.where(mask_3d, img, np.nan)  # Set non-masked regions to NaN

        (B, G, R) = cv2.split(img_masked)

        r_mean = np.nanmean(R)
        g_mean = np.nanmean(G)
        b_mean = np.nanmean(B)

        rgb = np.clip(np.array([r_mean, g_mean, b_mean]) / 255.0, 1e-3, 0.999) # 核心修复：限制范围
        XYZ = colour.sRGB_to_XYZ(rgb)
        xy = colour.XYZ_to_xy(XYZ)
        CCT = colour.xy_to_CCT(xy, 'hernandez1999')
        tone_values.append(-CCT)

    return torch.tensor(tone_values, device=device)

def contrastpro(masked_images_batch, masks_batch):
    """
    Compute contrast of the masked region of each image in the batch.
    Optimized to use tensor operations on GPU while preserving original logic.
    """
    device = masked_images_batch.device
    n, c, h, w = masked_images_batch.shape

    masks_batch = masks_batch.to(dtype=torch.bool, device=device)

    total_diffs = torch.zeros(n, device=device)
    valid_pixel_counts = torch.zeros(n, device=device)

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted_mask = torch.roll(masks_batch, shifts=(dx, dy), dims=(1, 2))
        shifted_images = torch.roll(masked_images_batch, shifts=(dx, dy), dims=(2, 3))

        valid_neighbors = masks_batch & shifted_mask

        diffs = torch.sum((masked_images_batch - shifted_images) ** 2, dim=1)  # Sum over channels
        diffs *= valid_neighbors  # Only consider valid neighbors

        total_diffs += diffs.sum(dim=(1, 2))
        valid_pixel_counts += valid_neighbors.sum(dim=(1, 2))

    valid_pixel_counts = valid_pixel_counts.clamp(min=1)  # Prevent division by zero
    contrast_values = total_diffs / (4 * valid_pixel_counts)
    contrast_values = torch.clamp(contrast_values, max=20000)
    return contrast_values


def measure_time(func, *args, **kwargs):
    """
    Measure the execution time of a function.
    Args:
        func (callable): The function to measure.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
    Returns:
        float: Execution time in seconds.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result


if __name__ == '__main__':
    img_name = 'a0005.jpg'
    image_paths = [
        os.path.join('train/Expert/01-Experts-A', img_name),
        os.path.join('train/Expert/02-Experts-B', img_name),
        os.path.join('train/Expert/03-Experts-C', img_name),
        os.path.join('train/Expert/04-Experts-D', img_name),
        os.path.join('train/Expert/05-Experts-E', img_name),
    ]

    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (448, 448))
        images.append(img)

    images_batch = torch.tensor(np.stack(images).transpose(0, 3, 1, 2), dtype=torch.float32)
    masks_batch = torch.ones(len(images), 448, 448)

    time_colorfulness, result_colorfulness = measure_time(image_colorfulness, images_batch, masks_batch)
    time_contrast, result_contrast = measure_time(contrastpro, images_batch, masks_batch)
    time_brightness, result_brightness = measure_time(image_brightness, images_batch, masks_batch)
    time_tone, result_tone = measure_time(image_tonepro, images_batch, masks_batch)

    metrics = ["Colorfulness", "Contrast", "Brightness", "Tone (CCT)"]
    results = [result_colorfulness, result_contrast, result_brightness, result_tone]

    plt.figure(figsize=(20, 10))
    for i, img in enumerate(images):
        plt.subplot(2, len(images), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        result_text = "\n".join(f"{metric}: {results[j][i]:.2f}" for j, metric in enumerate(metrics))
        plt.subplot(2, len(images), i + 1 + len(images))
        plt.text(0.5, 1, result_text, fontsize=15, ha="center", va="center", bbox=dict(facecolor='white', alpha=0.5))
        plt.axis("off")

    plt.tight_layout(pad=2.0)
    plt.show()