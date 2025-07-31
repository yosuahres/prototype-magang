#!/usr/bin/env python3

# -*- how to run -*-`
# # use sentence-transformers embedding
# python src/inference.py --config configs/st_emb.yaml --checkpoint checkpoints/st_emb.pth

# use openai embedding (make sure you've properly set OPENAI_API_KEY env variable)
# python src/inference.py --config configs/oai_emb.yaml --checkpoint checkpoints/oai_emb.pth

import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from model.network import Conv2DFiLMNet

from utils.img_utils import transform_imgs, load_pretrained_dino, get_dino_features_from_transformed_imgs
from utils.vlm_utils import get_text_embedding_options
from utils.file_utils import load_config


class AffordanceInference:
    def __init__(self, config_path, checkpoint_path, text_embedding_func):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")


        # Load config
        cfg = load_config(config_path)
        model_cfg = cfg["model"]
        
        # Build model
        self.model = Conv2DFiLMNet(**model_cfg)
        self.model.build()
        self.model.to(self.device)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        
        # Load DINO
        torch_home = cfg.get("torch_home", None)
        # self.dino = load_pretrained_dino('dinov2_vits14', use_registers=True, torch_path=torch_home).to(self.device).eval()
        self.dino = load_pretrained_dino('dinov2_vitl14', use_registers=True, torch_path=torch_home).to(self.device).eval()
        self.text_embedding_func = text_embedding_func
    
    @torch.no_grad()
    def predict(self, img_np, text, thresh=0.5, scale_factor=2):
        """
        img_np: H×W×3 numpy array (uint8 or float in [0,1])
        text: natural language description
        thresh: threshold for binary output
        scale_factor: int, factor to scale image size by
        Returns: similarity map as numpy array
        """
        # Preprocess image
        proc = transform_imgs(img_np, blur=False, scale_factor=scale_factor)[0]
        proc = proc.unsqueeze(0).to(self.device)
        
        # Get text embedding
        lang_emb = torch.from_numpy(self.text_embedding_func(text))
        lang_emb = lang_emb.to(self.device).unsqueeze(0).to(torch.float32)
        
        # Get DINO features
        feat = get_dino_features_from_transformed_imgs(self.dino, proc, repeat_to_orig_size=False)
        feat = feat.permute(0, 3, 1, 2)
        
        # Forward pass
        logits = self.model(feat, lang_emb).squeeze(0).squeeze(0)
        sim = torch.sigmoid(logits)
        
        if thresh is not None:
            sim = (sim > thresh).float()
        
        sim_np = sim.cpu().numpy()
        
        # Resize to original image size
        H, W = img_np.shape[:2]
        sim_np = np.array(
            T.functional.resize(
                Image.fromarray(sim_np.astype(np.float32), mode="F"),
                (H, W),
                interpolation=T.InterpolationMode.BILINEAR
            )
        )
        
        return sim_np, feat.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--text_query", type=str, default="hold", help="Text query for affordance detection")

    args = parser.parse_args()

    # Load config for additional settings
    cfg = load_config(args.config)

    # Get text embedding function
    text_embedding_option = cfg.get("text_embedding", "embeddings_oai")
    print(f"Using text embedding option: {text_embedding_option}")
    text_embedding_func = get_text_embedding_options(text_embedding_option)

    # Initialize inference
    inference = AffordanceInference(args.config, args.checkpoint, text_embedding_func)

    # Post-processing threshold for shutting down low affordance regions
    shutdown_thresh = cfg.get("thresh", 0.5)
    scale_factor = cfg.get("scale_factor", 2)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Convert frame to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run prediction
        result, _ = inference.predict(img_rgb, args.text_query, shutdown_thresh, scale_factor)

        # Normalize result to 0-255 for visualization
        heatmap = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Combine original frame and heatmap
        super_imposed_img = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # Display the resulting frame
        cv2.imshow('Affordance Detection', super_imposed_img)

        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
