import argparse
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import diffusers
import kornia
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import DatasetDict
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from packaging import version
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel
from models.utils import UnetConvout

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from transformers import RobertaModel

        return RobertaModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training script for ControlNet with metric conditioning.")

    # Model parameters
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained models or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of the pretrained model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files from huggingface.co/models, e.g., 'fp16'."
    )
    parser.add_argument(
        "--model_saved_dir",
        type=str,
        default="controlnet-models",
        help="Directory to save trained models and checkpoints."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to store downloaded models and datasets."
    )

    # Training parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Input image resolution (must be divisible by 8)."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size per device for training."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps (overrides num_train_epochs if provided)."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps."
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to keep."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for gradient accumulation."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate."
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale learning rate by batch size and accumulation steps."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="Learning rate scheduler type (e.g., 'linear', 'cosine')."
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for the LR scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles for cosine annealing LR scheduler."
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor for polynomial LR scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam optimizer from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading."
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam optimizer beta1 parameter."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam optimizer beta2 parameter."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Adam optimizer weight decay."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Adam optimizer epsilon parameter."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping."
    )

    # Logging and hardware
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory for TensorBoard logs."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TensorFloat-32 on supported GPUs."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Reporting integration (e.g., 'tensorboard', 'wandb')."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode (fp16/bf16)."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help="Set gradients to None instead of zeroing them."
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset name."
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Dataset configuration name."
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Directory containing training data (images and metadata)."
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="img",
        help="Dataset column name for original images."
    )
    parser.add_argument(
        "--metric_column",
        type=str,
        default="img_metric",
        help="Dataset column name for conditioning metrics."
    )
    parser.add_argument(
        "--expert_column",
        type=str,
        default="img_expert",
        help="Dataset column name for expert images."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Probability of replacing a region with an alternative expert image."
    )
    parser.add_argument(
        "--num_feathers",
        type=int,
        default=4,
        help="Number of feature channels for the metric conditioning."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Validations
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either --dataset_name or --train_data_dir")
    if args.resolution % 8 != 0:
        raise ValueError("Resolution must be divisible by 8")

    return args


def disturbance(tensor, sigma=5.0, max_shift=20, min_kernel_size=5, max_kernel_size=20):
    """
    Apply random Gaussian blur and translation perturbation
    args:
    tensor: Input tensor, shape (bs, 4, 512, 512)
    sigma_min: Minimum σ value for Gaussian blur
    sigma_max: Maximum σ value for Gaussian blur
    max_shift: Maximum pixel shift for translation
    return:
    (bs, 4, 512, 512)
    """
    dtype = tensor.dtype
    tensor = tensor.float()
    bs, c, h, w = tensor.shape
    device = tensor.device

    tensor = tensor.view(bs * c, 1, h, w)

    sigma = torch.full((bs * c, 2), sigma, device=device)

    k = torch.randint((min_kernel_size + 1) // 2, (max_kernel_size + 1) // 2, (1,)).item() * 2 + 1

    blurred = kornia.filters.gaussian_blur2d(
        input=tensor,
        kernel_size=(k, k),
        sigma=sigma
    )

    translations = torch.randint(
        -max_shift,
        max_shift + 1,
        (bs * c, 2),
        dtype=torch.float,
        device=device
    )

    affine_matrices = torch.zeros(bs * c, 2, 3, device=device)
    affine_matrices[:, 0, 0] = 1.0
    affine_matrices[:, 1, 1] = 1.0
    affine_matrices[:, :, 2] = translations

    translated = kornia.geometry.transform.warp_affine(
        blurred,
        affine_matrices,
        dsize=(h, w),
        mode='bilinear',
        padding_mode='border'
    )

    result = translated.view(bs, c, h, w).to(dtype=dtype)

    return result

def blend_images_with_mask(img1, img2, mask, sigma=1):
    """
    Blend the mask portion of img2 into img1, with edge smoothing using Gaussian blur.

    Args:
        img1 (PIL.Image): First image.
        img2 (PIL.Image): Second image.
        mask (torch.Tensor): Mask tensor with shape (H, W).
        sigma (float): Standard deviation for Gaussian blur, controlling the smoothness of edge transition.

    Returns:
        PIL.Image: Blended image.
    """
    img1_np = np.array(img1).astype(np.float32) / 255.0
    img2_np = np.array(img2).astype(np.float32) / 255.0

    mask_np = mask.numpy().astype(np.float32)
    mask_np = np.stack([mask_np] * 3, axis=-1)

    # alpha = gaussian_filter(mask_np, sigma=sigma)
    # blended_np = img1_np * (1 - alpha) + img2_np * alpha

    blended_np = img1_np * (1 - mask_np) + img2_np * mask_np

    blended_np = (blended_np * 255).astype(np.uint8)
    blended_img = Image.fromarray(blended_np)

    return blended_img

class ImageDatasetNewtest(Dataset):
    def __init__(
        self,
        input_root_dir,
        expert_root_dir,
        metric_root_dir,
        input_subfolder=None,
        image_transforms=None,
        metric_transforms=None,
    ):
        # 1) Input source subdirectory
        all_input_dirs = sorted([
            os.path.join(input_root_dir, d)
            for d in os.listdir(input_root_dir)
            if os.path.isdir(os.path.join(input_root_dir, d))
        ])
        if input_subfolder:
            # User-specified
            chosen = os.path.join(input_root_dir, input_subfolder)
            if not os.path.isdir(chosen):
                raise ValueError(f"Input subdir not exist: {chosen}")
            self.input_dirs = [chosen]
        else:
            self.input_dirs = all_input_dirs

        # 2) Expert subdirectory
        self.expert_dirs = sorted([
            os.path.join(expert_root_dir, d)
            for d in os.listdir(expert_root_dir)
            if os.path.isdir(os.path.join(expert_root_dir, d))
        ])

        # 3) Metric directory
        self.metric_dirs = sorted([
            os.path.join(metric_root_dir, d)
            for d in os.listdir(metric_root_dir)
            if os.path.isdir(os.path.join(metric_root_dir, d))
        ])

        # 4) Filenames
        self.image_files = sorted(os.listdir(self.input_dirs[0]))

        self.image_transforms = image_transforms
        self.metric_transforms = metric_transforms
        self.alpha = args.alpha

        self.num_inputs = len(self.input_dirs)
        self.num_experts = len(self.expert_dirs)
        self.num_images = len(self.image_files)
        self.num_data = self.num_images * self.num_inputs * self.num_experts

        self.column_names = ["img", "img_metric", "img_expert"]

    def __len__(self):
        return self.num_data

    def load_scores(self, path):
        arr = np.load(path)['data'].astype(np.float32)
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        # Three-level index decomposition
        img_idx = idx // (self.num_inputs * self.num_experts)
        rem = idx % (self.num_inputs * self.num_experts)
        input_idx = rem // self.num_experts
        expert_idx = rem % self.num_experts

        img_name = self.image_files[img_idx]

        # 1) Load input source image
        input_path = os.path.join(self.input_dirs[input_idx], img_name)
        img = Image.open(input_path).convert("RGB")

        # 2) Load expert segmentation map
        expert_path = os.path.join(self.expert_dirs[expert_idx], img_name)
        img_expert = Image.open(expert_path).convert("RGB")

        # 3) Load metric file (.npz)
        metric_fname = img_name.replace('.jpg', '_scores_norm_q.npz')
        metric_path = os.path.join(self.metric_dirs[expert_idx], metric_fname)
        img_metric = self.load_scores(metric_path)

        # 4) Optional preprocessing
        if self.image_transforms:
            img = self.image_transforms(img)
            img_expert = self.image_transforms(img_expert)
        if self.metric_transforms:
            img_metric = self.metric_transforms(img_metric)

        # 5) Randomly replace region (to increase diversity)
        if random.random() < self.alpha:
            # Assume img_metric's first channel is the segmentation mask
            seg_map = img_metric[0]
            unique_ids = torch.unique(seg_map)
            blocks = []
            for bid in unique_ids:
                mask = (seg_map == bid)
                area = mask.sum().item()
                if area < 1024:  # Ignore too small areas
                    continue
                blocks.append((mask, area))
            if blocks:
                # Select region according to area probability
                areas = [b[1] for b in blocks]
                probs = [a / sum(areas) for a in areas]
                mask_sel, _ = random.choices(blocks, weights=probs, k=1)[0]

                # Current region features
                current_feat = img_metric[:, mask_sel]

                # Find the best expert with the largest difference
                best_idx, max_diff = expert_idx, -1.0
                for cand in range(self.num_experts):
                    if cand == expert_idx:
                        continue
                    cand_metric_path = os.path.join(
                        self.metric_dirs[cand], metric_fname
                    )
                    cand_metric = self.load_scores(cand_metric_path)
                    cand_feat = cand_metric[:, mask_sel]
                    diff = torch.norm(current_feat - cand_feat, p=2).item()
                    if diff > max_diff:
                        max_diff, best_idx = diff, cand

                # Replace
                if best_idx != expert_idx:
                    # Replace expert image region
                    best_exp_img = Image.open(
                        os.path.join(self.expert_dirs[best_idx], img_name)
                    ).convert("RGB")
                    if self.image_transforms:
                        best_exp_img = self.image_transforms(best_exp_img)
                    img_expert[:, mask_sel] = best_exp_img[:, mask_sel]

                    # Replace metric region
                    best_metric = self.load_scores(
                        os.path.join(self.metric_dirs[best_idx], metric_fname)
                    )
                    img_metric[:, mask_sel] = best_metric[:, mask_sel]

        return {
            "img": img,
            "img_metric": img_metric,
            "img_expert": img_expert,
        }

class ImageDatasetNew(Dataset):
    def __init__(self, image_dir, expert_dirs_dir, metric_dir, image_transforms=None, metric_transforms=None):
        self.image_dir = image_dir
        self.expert_image_dir = expert_dirs_dir
        self.metric_dir = metric_dir

        self.image_transforms = image_transforms
        self.metric_transforms = metric_transforms

        self.image_files = sorted(os.listdir(image_dir))
        self.expert_dirs = sorted(os.listdir(expert_dirs_dir))
        self.metric_files = sorted(os.listdir(metric_dir))

        self.num_exp_dir = len(self.expert_dirs)
        self.num_data = len(self.image_files) * self.num_exp_dir

        self.column_names = ["img", "img_metric", "img_expert"]

    def __len__(self):
        return self.num_data

    def load_scores(self, path):
        data = np.load(path)['data'].astype(np.float32)
        return torch.from_numpy(data)

    def __getitem__(self, idx):
        img_idx = idx // self.num_exp_dir
        exp_dirs_idx = idx % self.num_exp_dir

        img_name = self.image_files[img_idx]
        exp_img_name = self.expert_dirs[exp_dirs_idx]
        metric_name = self.metric_files[exp_dirs_idx]

        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        expert_img_path = os.path.join(self.expert_image_dir, exp_img_name, img_name)
        img_expert = Image.open(expert_img_path).convert("RGB")

        metric_path = os.path.join(self.metric_dir, metric_name, img_name.replace('.jpg', '_scores_norm_q.npz'))
        img_metric = self.load_scores(metric_path)

        if self.image_transforms:
            img = self.image_transforms(img)
            img_expert = self.image_transforms(img_expert)

        if self.metric_transforms:
            img_metric = self.metric_transforms(img_metric)

        if random.random() < args.alpha:
            # Extract segmentation map from metric's first channel
            seg_map = img_metric[0]  # 假设第一个通道存储分割信息

            # Get all independent segmentation blocks
            unique_blocks = torch.unique(seg_map)
            blocks = []
            for block_id in unique_blocks:
                mask = (seg_map == block_id)
                area = torch.sum(mask).item()
                if area < 1024:
                    continue
                blocks.append({'mask': mask, 'area': area})

            # Select block according to area probability
            if blocks:
                total_area = sum(b['area'] for b in blocks)
                probabilities = [b['area'] / total_area for b in blocks]
                selected_block = random.choices(blocks, weights=probabilities, k=1)[0]
                elected_mask = selected_block['mask']

                # Calculate current block feature (geometric distance)
                current_block = img_metric[:, elected_mask]  # [4, N]

                # Find the best expert with the largest difference  
                max_diff = -1
                best_exp_idx = exp_dirs_idx

                for candidate_idx in range(self.num_exp_dir):
                    if candidate_idx == exp_dirs_idx:
                        continue

                    # Load candidate expert metric
                    candidate_metric_path = os.path.join(
                        self.metric_dir,
                        self.metric_files[candidate_idx],
                        img_name.replace('.jpg', '_scores_norm_q.npz')
                    )
                    candidate_metric = self.load_scores(candidate_metric_path)

                    # Calculate candidate feature difference
                    candidate_block = candidate_metric[:, elected_mask]  # [4, N]
                    diff = torch.norm(current_block - candidate_block, p=2).item()  # L2范数

                    if diff > max_diff:
                        max_diff = diff
                        best_exp_idx = candidate_idx

                # Execute replacement operation
                if best_exp_idx != exp_dirs_idx:
                    # Load best expert image
                    best_exp_img = Image.open(
                        os.path.join(self.expert_image_dir, self.expert_dirs[best_exp_idx], img_name)
                    ).convert("RGB")
                    best_exp_img = self.image_transforms(best_exp_img)

                    # Mix images (directly replace mask region)
                    img_expert[:, elected_mask] = best_exp_img[:, elected_mask]

                    # Load and replace metric
                    best_metric = self.load_scores(os.path.join(
                        self.metric_dir,
                        self.metric_files[best_exp_idx],
                        img_name.replace('.jpg', '_scores_norm_q.npz')
                    ))
                    img_metric[:, elected_mask] = best_metric[:, elected_mask]

        return {
            "img": img,
            "img_metric": img_metric,
            "img_expert": img_expert,
        }

class ImageDataset(Dataset):
    def __init__(self, image_dir, mask_json_dir, metric_json_dir, expert_image_dir, image_transforms=None, metric_transforms=None):
        self.image_dir = image_dir
        self.expert_image_dir = expert_image_dir
        self.mask_json_dir = mask_json_dir
        self.metric_json_dir = metric_json_dir
        self.image_transforms = image_transforms
        self.metric_transforms = metric_transforms

        self.image_files = sorted(os.listdir(image_dir))
        self.expert_files = sorted(os.listdir(expert_image_dir))

        self.num_exp = len(self.expert_files)
        self.num_data = len(self.image_files) * self.num_exp

        with open(metric_json_dir, "r") as f:
            self.class_metric = json.load(f)
        with open(mask_json_dir, "r") as f:
            self.image_seg = json.load(f)

        self.column_names = ["img", "img_metric", "img_expert"]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        img_idx = idx // self.num_exp
        exp_idx = idx % self.num_exp

        img_name = self.image_files[img_idx]
        exp_img_name = self.expert_files[exp_idx]

        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        expert_img_path = os.path.join(self.expert_image_dir, exp_img_name, img_name)
        img_expert = Image.open(expert_img_path).convert("RGB")

        seg = self.image_seg[img_name]
        h, w = seg[0]["segmentation"]["size"]
        seg_binary = [torch.tensor(mask_util.decode(i["segmentation"]), dtype=torch.uint8) for i in seg]

        metric = self.class_metric[img_name][exp_img_name]
        metric_binary = [list(_.values()) for _ in metric.values()]

        img_metric = torch.empty(args.num_feathers, h, w)
        for i in range(args.num_feathers):  # num feathers
            for j in range(len(seg_binary)):  # num class
                img_metric[i][seg_binary[j] == 1] = metric_binary[j][i]

        if random.random() < args.alpha:
            # print(img_metric.shape)  # [num_feathers, 448, 448]
            mask_areas = [torch.sum(mask).item() for mask in seg_binary]
            probabilities = [area / sum(mask_areas) for area in mask_areas]
            elected_mask = random.choices(seg_binary, weights=probabilities, k=1)[0]  # 448, 448
            elected_index = None
            for idx, mask in enumerate(seg_binary):
                if torch.equal(mask, elected_mask):
                    elected_index = idx
                    break
            mask_indices = torch.nonzero(elected_mask)
            y, x = mask_indices[0] if len(mask_indices) > 0 else (0, 0)
            mask_values_per_channel = img_metric[:, y, x]

            min_distance = 0.0
            best_img_exp = None
            best_met_binary = None
            for i in range(self.num_exp):
                if i == exp_idx:
                    continue
                exp_img_name = self.expert_files[i]
                expert_img_path = os.path.join(self.expert_image_dir, exp_img_name, img_name)
                img_exp = Image.open(expert_img_path).convert("RGB")
                met = self.class_metric[img_name][exp_img_name]
                met_binary = [list(_.values()) for _ in met.values()][elected_index]  # [num_class, num_feathers]
                distance = np.linalg.norm(np.array(mask_values_per_channel) - np.array(met_binary))
                if distance > min_distance:
                    min_distance = distance
                    best_img_exp = img_exp
                    # best_met_binary = np.where(np.array(met_binary) > np.array(mask_values_per_channel), 1, -1)
                    best_met_binary = met_binary

            for i in range(img_metric.shape[0]):
                img_metric[i, elected_mask == 1] = best_met_binary[i]

            img_expert = blend_images_with_mask(img_expert, best_img_exp, elected_mask)

        if self.image_transforms:
            img = self.image_transforms(img)
            img_expert = self.image_transforms(img_expert)

        if self.metric_transforms:
            img_metric = self.metric_transforms(img_metric)

        return {
            "img": img,
            "img_metric": img_metric,
            "img_expert": img_expert,
        }

def make_train_dataset(args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    # normalize = transforms.Lambda(lambda x: x / 127.5 - 1)
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ]
    )

    metric_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        ]
    )

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            train_dataset = ImageDatasetNewtest(
                input_root_dir=os.path.join(args.train_data_dir, "Input"),
                expert_root_dir=os.path.join(args.train_data_dir, "Expert"),
                metric_root_dir=os.path.join(args.train_data_dir, "score_tensors_norm_q/Expert"),
                image_transforms=image_transforms,
                metric_transforms=metric_transforms
            )

            dataset = DatasetDict({
                "train": train_dataset,
            })
        else:
            raise ValueError(
                f"lack of `--train_data_dir` value"
            )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.metric_column is None:
        metric_column = column_names[1]
        logger.info(f"conditioning metric column defaulting to {metric_column}")
    else:
        metric_column = args.metric_column
        if metric_column not in column_names:
            raise ValueError(
                f"`--metric_column` value '{args.metric_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.expert_column is None:
        expert_column = column_names[2]
        logger.info(f"expert column defaulting to {expert_column}")
    else:
        expert_column = args.expert_column
        if expert_column not in column_names:
            raise ValueError(
                f"`--expert_column` value '{args.expert_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        # train_dataset = dataset["train"].with_transform(preprocess_train)

    return dataset["train"]

def collate_fn(examples):
    img = torch.stack([example["img"] for example in examples])
    img = img.to(memory_format=torch.contiguous_format).float()

    img_metric = torch.stack([example["img_metric"] for example in examples])
    img_metric = img_metric.to(memory_format=torch.contiguous_format).float()

    img_expert = torch.stack([example["img_expert"] for example in examples])
    img_expert = img_expert.to(memory_format=torch.contiguous_format).float()

    return {
        "img": img,
        "img_metric": img_metric,
        "img_expert": img_expert,
    }

def main(args):
    logging_dir = Path(args.model_saved_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.model_saved_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.model_saved_dir is not None:
            os.makedirs(args.model_saved_dir, exist_ok=True)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = args.mixed_precision
    if weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # load data
    train_dataset = make_train_dataset(args, accelerator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    unet_conv_out = UnetConvout(unet.conv_out.weight, unet.conv_out.bias)

    # Load trainable models
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.model_saved_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        logger.info(f"  Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")

        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=4)

        args.resume_from_checkpoint = None
        initial_global_step = 0
    else:
        logger.info(f"  Resuming from checkpoint {path}.")
        global_step = int(path.split("-")[1])

        train_model_path = os.path.join(args.model_saved_dir, str(path))
        controlnet = ControlNetModel.from_pretrained(os.path.join(train_model_path, 'controlnet'), conditioning_channels=4)
        unet_conv_out.load_state_dict(torch.load(os.path.join(train_model_path, 'unet_conv_out.pth'), weights_only=True))

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch

    # adjust trainable
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    controlnet.requires_grad_(True)
    controlnet.train()
    unet_conv_out.requires_grad_(True)
    unet_conv_out.train()

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, model_saved_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(model_saved_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into models
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all models weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the models in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        list(controlnet.parameters()) + list(unet_conv_out.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    controlnet, unet_conv_out, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, unet_conv_out, optimizer, train_dataloader, lr_scheduler
    )

    # Move vae, unet, md and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    unet_conv_out.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)

    # get null prompt embedding
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    text_input_ids = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)

    with torch.no_grad():
        null_prompt_embeds = text_encoder(text_input_ids)[0]
    encoder_hidden_states = null_prompt_embeds.to(dtype=weight_dtype).repeat(args.train_batch_size, 1, 1)
    del text_encoder, tokenizer

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        runid = {"wandb": {"id": args.run_id}} if args.run_id is not None else None

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=runid)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet, unet_conv_out):
            # Convert images to latent space (-1 to 1)
                img_orig = batch["img"].to(dtype=weight_dtype)
                latents = vae.encode(img_orig).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                img_expt = batch["img_expert"].to(dtype=weight_dtype)
                expert_latents = vae.encode(img_expt).latent_dist.sample()
                expert_latents = expert_latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(expert_latents)

                # Sample a random timestep for each image and Add noise to the latents
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (expert_latents.shape[0],), device=expert_latents.device).long()
                noisy_latents = noise_scheduler.add_noise(expert_latents.float(), noise.float(), timesteps).to(dtype=weight_dtype)

                # Controlnet input
                metrics = batch["img_metric"].to(dtype=weight_dtype)  # bs, 4, 512, 512

                metrics = disturbance(metrics)  # bs, 4, 512, 512

                # Controlnet result
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    original_img_vae=latents,  # bs, 4, 64, 64
                    controlnet_cond=metrics,  # bs, 4, 512, 512
                    return_dict=False,
                )

                # Pred model result
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]  # bs, 320, 64, 64

                noise_pred = unet_conv_out(model_pred)  # bs, 4, 64, 64

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(expert_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                Lrec = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                loss = Lrec

                accelerator.backward(loss)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(list(controlnet.parameters()) + list(unet_conv_out.parameters()), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.model_saved_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.model_saved_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.model_saved_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)

                        controlnet.module.save_pretrained(os.path.join(save_path, 'controlnet'))
                        torch.save(unet_conv_out.module.state_dict(), os.path.join(save_path, 'unet_conv_out.pth'))

                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)

        controlnet.module.save_pretrained(os.path.join(save_path, 'controlnet'))

        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)

