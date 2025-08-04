#!/usr/bin/env python

# -*- how to run -*-`
# # use sentence-transformers embedding
# python src/inference.py --config configs/st_emb.yaml --checkpoint checkpoints/st_emb.pth

# use openai embedding (make sure you've properly set OPENAI_API_KEY env variable)
# python src/inference.py --config configs/oai_emb.yaml --checkpoint checkpoints/oai_emb.pth

import argparse
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

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
        self.dino = load_pretrained_dino('dinov2_vits14', use_registers=True, torch_path=torch_home).to(self.device).eval()
        self.text_embedding_func = text_embedding_func
    
    @torch.no_grad()
    def predict(self, img_np, text, thresh=0.1, scale_factor=2):
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
        
        # Debug: Print embedding statistics
        print(f"Text: '{text}'")
        print(f"Embedding shape: {lang_emb.shape}")
        print(f"Embedding mean: {lang_emb.mean().item():.4f}")
        print(f"Embedding std: {lang_emb.std().item():.4f}")
        print(f"Embedding min: {lang_emb.min().item():.4f}")
        print(f"Embedding max: {lang_emb.max().item():.4f}")
        print("---")
        
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
    
    args = parser.parse_args()
    
    # Load config for additional settings
    cfg = load_config(args.config)
    
    ## ===== Example usage ===== ##
    image_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'kursi.jpeg')
    text_query = "sit"
    # text_query = "where to hold to avoid hot surfaces"
    
    # Load image
    img = Image.open(image_path).convert("RGB")

    # Resize image if it's too large
    max_image_size = cfg.get("max_image_size", 1024)
    if max(img.size) > max_image_size:
        img.thumbnail((max_image_size, max_image_size))
    
    img = np.array(img)
    
    # Use sentence transformer for affordance mapping (what works best)
    print("=== COMPARING EMBEDDING METHODS ===")
    
    # Get both embedding functions
    st_embedding_func = get_text_embedding_options("embeddings_st")
    gemini_embedding_func = get_text_embedding_options("embeddings_gemini")
    
    # Compare embeddings for the query
    print(f"\nQuery: '{text_query}'")
    st_emb = st_embedding_func(text_query)
    gemini_emb = gemini_embedding_func(text_query)
    
    print(f"\nSentence Transformer embedding:")
    print(f"  Shape: {st_emb.shape}")
    print(f"  Mean: {st_emb.mean():.6f}")
    print(f"  Std: {st_emb.std():.6f}")
    print(f"  Min: {st_emb.min():.6f}")
    print(f"  Max: {st_emb.max():.6f}")
    print(f"  Norm: {np.linalg.norm(st_emb):.6f}")
    
    print(f"\nGemini embedding:")
    print(f"  Shape: {gemini_emb.shape}")
    print(f"  Mean: {gemini_emb.mean():.6f}")
    print(f"  Std: {gemini_emb.std():.6f}")
    print(f"  Min: {gemini_emb.min():.6f}")
    print(f"  Max: {gemini_emb.max():.6f}")
    print(f"  Norm: {np.linalg.norm(gemini_emb):.6f}")
    
    # Calculate similarity between embeddings
    similarity = np.dot(st_emb, gemini_emb) / (np.linalg.norm(st_emb) * np.linalg.norm(gemini_emb))
    print(f"\nCosine similarity between ST and Gemini embeddings: {similarity:.6f}")
    
    print("\n" + "="*50)
    
    # Use the embedding method specified in config
    text_embedding_option = cfg.get("text_embedding", "embeddings_st")
    print(f"\nUsing embedding method from config: {text_embedding_option}")
    text_embedding_func = get_text_embedding_options(text_embedding_option)

    # Initialize inference
    inference = AffordanceInference(args.config, args.checkpoint, text_embedding_func)
    
    # Post-processing threshold for shutting down low affordance regions
    shutdown_thresh = cfg.get("thresh", 0.5)
    scale_factor = cfg.get("scale_factor", 2)
    
    # Run prediction with specified method
    result, dino_features = inference.predict(img, text_query, shutdown_thresh, scale_factor)
    print(f"\nPredicted affordance map with {text_embedding_option} for '{text_query}' with shape {result.shape}")
    
    # COMPARISON: Also run with Sentence Transformer if using Gemini
    if text_embedding_option == "embeddings_gemini":
        print("\n=== COMPARISON WITH SENTENCE TRANSFORMER ===")
        inference_st = AffordanceInference(args.config, args.checkpoint, st_embedding_func)
        result_st, _ = inference_st.predict(img, text_query, shutdown_thresh, scale_factor)
        
        # Compare results
        print(f"Gemini result - Non-zero pixels: {np.count_nonzero(result)}, Max: {result.max():.4f}, Mean: {result.mean():.4f}")
        print(f"SentenceT result - Non-zero pixels: {np.count_nonzero(result_st)}, Max: {result_st.max():.4f}, Mean: {result_st.mean():.4f}")
        
        # Detailed analysis of why Gemini fails
        print("\n=== DETAILED ANALYSIS ===")
        if np.count_nonzero(result) == 0:
            print("❌ PROBLEM: Gemini produces NO activated pixels!")
            print("   This means the model thinks there's no affordance anywhere.")
        elif np.count_nonzero(result) < np.count_nonzero(result_st) * 0.1:
            print("❌ PROBLEM: Gemini produces very few activated pixels compared to SentenceT")
        
        if result.max() < 0.1:
            print("❌ PROBLEM: Gemini's maximum activation is very low")
            print("   The model has very low confidence in any region")
            
        if similarity < 0.1:
            print("❌ ROOT CAUSE: Embedding spaces are completely different!")
            print("   Cosine similarity < 0.1 means embeddings are nearly orthogonal")
            print("   The model was trained on SentenceTransformer embeddings and doesn't understand Gemini embeddings")
        elif similarity < 0.3:
            print("⚠️  WARNING: Embedding spaces are quite different")
            print("   Cosine similarity < 0.3 suggests significant semantic differences")
        
        print(f"\n💡 SOLUTIONS TO TRY:")
        print(f"1. Train a new model with Gemini embeddings")
        print(f"2. Use embedding alignment/transformation techniques")
        print(f"3. Stick with SentenceTransformer for affordance (it works!)")
        print(f"4. Use Gemini only for VLM tasks, not affordance prediction")
        
        # Store both results for visualization
        result_comparison = result_st
    else:
        result_comparison = None

    # Get Gemini response
    from get_vlm_response import get_end_to_end_matching
    image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract filename without extension
    result_vlm = get_end_to_end_matching("chair", os.path.join(os.path.dirname(__file__), '..', 'examples'), [image_name], use_vlm='gemini')
    print("VLM Analysis Result:", result_vlm)
    
    # Visualize DINO features with PCA
    from sklearn.decomposition import PCA
    
    bs, n_feats, patch_h, patch_w = dino_features.shape
    dino_features_reshaped = dino_features.transpose(0, 2, 3, 1).reshape(bs * patch_h * patch_w, n_feats)
    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(dino_features_reshaped)
    
    # Normalize PCA result to be in [0, 1] for visualization
    pca_result = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
    
    pca_img = pca_result.reshape(patch_h, patch_w, 3)
    
    # Display result
    if result_comparison is not None:
        # Show comparison: Gemini vs Sentence Transformer
        plt.figure(figsize=(24, 6))
        
        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(result, cmap='hot')
        plt.title(f"Gemini Affordance Map: '{text_query}'")
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(result_comparison, cmap='hot')
        plt.title(f"SentenceT Affordance Map: '{text_query}'")
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(pca_img)
        plt.title("DINOv2 Features (PCA)")
        plt.axis('off')
        
    else:
        # Original single result view
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(result, cmap='hot')
        plt.title(f"Affordance Map: '{text_query}'")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pca_img)
        plt.title("DINOv2 Features (PCA)")
        plt.axis('off')
    
    plt.tight_layout()
    
    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'hasil_st', 'asu.png'))


if __name__ == "__main__":
    main()
