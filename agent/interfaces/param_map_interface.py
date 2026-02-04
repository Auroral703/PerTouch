import numpy as np
import cv2
from config import Config
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class ParamMapInterface:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state = {}
        
        # Internal state for SAM3
        self.processor = None
        self.inference_state = None
        self.current_image_shape = (512, 512)

        print("Loading SAM3 Model...")
        # Ensure the checkpoint path is correct in your environment
        self.model = build_sam3_image_model(checkpoint_path=Config.SAM_CKPT) 
        self.model.to(self.device)
        self.processor = Sam3Processor(self.model)
        print("SAM3 Model Loaded.")
    
    def reset(self):
        self.state = {}

    def set_state(self, state_json):
        self.state = state_json if state_json else {}

    def set_image(self, image_np):
        """
        Set the current image for SAM3 encoding.
        Must be called whenever the original image changes.
        image_np: RGB Numpy array (H, W, 3)
        """
        if image_np is None:
            return

        self.current_image_shape = image_np.shape[:2] # (H, W)
        
        # SAM3 expects PIL Image
        pil_image = Image.fromarray(image_np.astype('uint8'))
        
        # Heavy operation: Encode image once
        with torch.no_grad():
            self.inference_state = self.processor.set_image(pil_image)

    def update_state(self, update_dict):
        """
        Batch update parameters.
        Logic:
        1. If updating a specific region: Set value directly.
        2. If updating 'global': Calculate delta and apply to ALL other regions to maintain relative difference.
        3. Clamp all values between -1.0 and 1.0.
        """
        if not update_dict:
            return

        # Helper to clamp values
        def clamp(n):
            return max(-1.0, min(1.0, float(n)))

        # 1. Process Global updates first to calculate deltas
        if 'global' in update_dict:
            if 'global' not in self.state:
                self.state['global'] = {}

            for attr, new_val in update_dict['global'].items():
                new_val = clamp(new_val)
                
                # Get old global value (default to 0.0 if not set)
                old_val = self.state['global'].get(attr, 0.0)
                delta = new_val - old_val

                # Update Global
                self.state['global'][attr] = new_val

                # Propagate Delta to all other existing regions
                for region in self.state:
                    if region == 'global': 
                        continue
                    
                    # Only update if the region already has this attribute set
                    if attr in self.state[region]:
                        current_region_val = self.state[region][attr]
                        # Apply delta and clamp
                        self.state[region][attr] = clamp(current_region_val + delta)

        # 2. Process specific region updates (Non-Global)
        for region, params in update_dict.items():
            if region == 'global':
                continue # Already handled

            if region not in self.state:
                self.state[region] = {}
            
            for attr, val in params.items():
                # Direct assignment for local adjustments
                self.state[region][attr] = clamp(val)

    def _get_mask_for_tag(self, tag, target_shape=(512, 512)):
        """
        Generates a binary mask (0.0 to 1.0) for a specific tag using SAM3.
        """
        try:
            with torch.no_grad():
                # SAM3 Inference
                # output contains keys: "masks", "boxes", "scores"
                output = self.processor.set_text_prompt(
                    state=self.inference_state, 
                    prompt=tag
                )
            
            masks = output["masks"] # List of tensors
            
            if len(masks) == 0:
                print(f"SAM3: No object found for tag '{tag}'")
                return np.zeros(target_shape, dtype=np.float32)

            # Take the first mask (usually the highest score for unambiguous prompts)
            # Convert Tensor -> Numpy
            mask_tensor = masks[0] 
            mask_np = mask_tensor.cpu().numpy().astype(np.float32)
            
            # Handle potential dimension issues (SAM3 might return boolean [1, H, W])
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            
            # Resize to target shape (512x512) for the parameter map
            if mask_np.shape != target_shape:
                mask_np = cv2.resize(mask_np, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Normalize to 0-1 range just in case
            mask_np = np.clip(mask_np, 0.0, 1.0)
            
            return mask_np

        except Exception as e:
            print(f"SAM3 Error for tag '{tag}': {e}")
            return None

    def generate_parameter_map(self):
        """
        Convert JSON state to 512x512x4 Numpy array: Parameter Map
        """
        # 0: Colorfulness, 1: Contrast, 2: Temp, 3: Brightness
        param_map = np.zeros((Config.IMG_SIZE[0], Config.IMG_SIZE[1], 4), dtype=np.float32)
        
        # Attribute name to channel index mapping
        attr_map = {
            'colorfulness': 0, 'saturation': 0,
            'contrast': 1,
            'temperature': 2, 'warmth': 2,
            'brightness': 3, 'exposure': 3
        }

        # 1. Base Layer: Global
        if 'global' in self.state:
            for attr, val in self.state['global'].items():
                idx = attr_map.get(attr.lower())
                if idx is not None:
                    param_map[:, :, idx] = val

        # 2. Local Layer: Region Overwrite
        for region, params in self.state.items():
            if region == 'global': 
                continue
            
            # Generate Mask (Using SAM3 if available)
            mask = self._get_mask_for_tag(region)
            if mask is None:
                continue
            
            for attr, val in params.items():
                idx = attr_map.get(attr.lower())
                if idx is not None:
                    # Logic: old * (1-mask) + new * mask
                    # Uses numpy broadcasting
                    param_map[:, :, idx] = param_map[:, :, idx] * (1 - mask) + val * mask
                    
        return param_map