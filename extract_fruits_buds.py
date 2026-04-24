import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import cv2
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm

def main():
    text_prompt = 'bud or fruit'
    input_base_dir = '/home/cvl/farmos_env/Rawframes_21apr'
    output_base_dir = '/home/cvl/farmos_env/Segmented_Fruits_Buds_21apr'
    overlay_base_dir = '/home/cvl/farmos_env/Segmented_Fruits_Buds_21apr_Overlays'
    
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(overlay_base_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 3 PCS on {device}...")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    
    video_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
    
    print(f"\n--- Processing ALL Frames ---")
    
    for plant_dir in video_dirs:
        plant_path = os.path.join(input_base_dir, plant_dir)
        
        out_plant_dir = os.path.join(output_base_dir, plant_dir)
        os.makedirs(out_plant_dir, exist_ok=True)
        
        overlay_plant_dir = os.path.join(overlay_base_dir, plant_dir)
        os.makedirs(overlay_plant_dir, exist_ok=True)
        
        frames = sorted(glob.glob(os.path.join(plant_path, '*.jpg')) + glob.glob(os.path.join(plant_path, '*.png')))
        print(f"\n--- Processing Plant: {plant_dir} ({len(frames)} frames) ---")
        
        for frame_path in tqdm(frames, desc=plant_dir):
            fname = os.path.basename(frame_path)
            out_fname = fname.rsplit('.', 1)[0] + '.png'
            out_path = os.path.join(out_plant_dir, out_fname)
            overlay_out_path = os.path.join(overlay_plant_dir, fname)
            
            if os.path.exists(out_path) and os.path.exists(overlay_out_path):
                continue
                
            raw_image = Image.open(frame_path).convert("RGB")
            
            try:
                inputs = processor(images=raw_image, text=text_prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                best_mask = np.zeros((raw_image.height, raw_image.width), dtype=bool)
                
                import torch.nn.functional as F
                low_res_masks = outputs.pred_masks.squeeze(0).unsqueeze(0)
                scaled_masks = F.interpolate(
                    low_res_masks,
                    size=(raw_image.height, raw_image.width),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)
                
                if hasattr(outputs, 'pred_logits'):
                    logits = outputs.pred_logits[0]
                    if logits.dim() > 1:
                        scores = torch.sigmoid(logits).max(dim=-1)[0]
                    else:
                        scores = torch.sigmoid(logits)
                    
                    best_idx = scores.argmax()
                    if scores[best_idx] > 0.3:
                        valid_mask = scaled_masks[best_idx]
                        if valid_mask.dim() == 4: valid_mask = valid_mask.squeeze(0)
                        probs = torch.sigmoid(valid_mask)
                        best_mask = (probs > 0.5).cpu().numpy().astype(bool)
                        
                elif hasattr(outputs, 'iou_scores'):
                    scores = outputs.iou_scores[0]
                    if scores.dim() > 1: scores = scores.max(dim=-1)[0]
                    
                    best_idx = scores.argmax()
                    if scores[best_idx] > 0.3:
                        valid_mask = scaled_masks[best_idx]
                        if valid_mask.dim() == 4: valid_mask = valid_mask.squeeze(0)
                        probs = torch.sigmoid(valid_mask)
                        best_mask = (probs > 0.5).cpu().numpy().astype(bool)
                else:
                    probs = torch.sigmoid(scaled_masks)
                    best_mask = (probs[0] > 0.5).cpu().numpy().astype(bool)
                
                if best_mask.ndim == 3 and best_mask.shape[0] == 1:
                    best_mask = best_mask.squeeze(0)
                    
                if np.any(best_mask):
                    img_bgr = cv2.imread(frame_path)
                    
                    # 1. Save transparent crop
                    indices = np.where(best_mask)
                    y_min, y_max = max(0, indices[0].min()-10), min(raw_image.height, indices[0].max()+10)
                    x_min, x_max = max(0, indices[1].min()-10), min(raw_image.width, indices[1].max()+10)
                    
                    cropped = img_bgr[y_min:y_max, x_min:x_max].copy()
                    alpha_mask = best_mask[y_min:y_max, x_min:x_max]
                    
                    b, g, r = cv2.split(cropped)
                    alpha = (alpha_mask * 255).astype(np.uint8)
                    rgba = cv2.merge([b, g, r, alpha])
                    
                    cv2.imwrite(out_path, rgba)
                    
                    # 2. Save overlay image
                    overlay = img_bgr.copy()
                    overlay[best_mask] = [180, 105, 255] # Pink highlight for the mask
                    blended = cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0)
                    cv2.imwrite(overlay_out_path, blended)
                    
            except Exception as e:
                with open("extract_fruits_buds_errors.log", "a") as f:
                    f.write(f"Error processing {fname}: {str(e)}\n")

    print("\nBatch Fully Complete!")

if __name__ == "__main__":
    main()
