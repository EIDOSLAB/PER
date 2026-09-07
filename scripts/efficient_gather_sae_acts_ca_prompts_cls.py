"""
Gather feature activations from a SAE for a given hookpoint and save them to a file.
Save-optimized version to prevent OOM errors during file saving.
For object classes instead of styles.
"""

import os
import sys
import json
import traceback

import fire
import torch
from diffusers.utils.import_utils import is_xformers_available

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline, HookedStableDiffusionXLPipeline
from SAE.sae import Sae
from UnlearnCanvas_resources.const import class_available

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
import pickle
import gc
import time

import tqdm


def save_tensor_to_disk(tensor, filepath):
    """Save a tensor to disk using torch.save instead of pickle for better memory efficiency."""
    try:
        torch.save(tensor, filepath)
        print(f"✓ Successfully saved tensor to {filepath}")
    except Exception as e:
        print(f"❌ ERROR saving tensor to {filepath}: {e}")
        raise


def load_tensor_from_disk(filepath):
    """Load a tensor from disk using torch.load."""
    try:
        tensor = torch.load(filepath)
        print(f"✓ Successfully loaded tensor from {filepath}")
        return tensor
    except Exception as e:
        print(f"❌ ERROR loading tensor from {filepath}: {e}")
        raise


def main(checkpoint_path, hookpoint, pipe_path, save_dir, steps=100, seed=188, resume=True):
    print("="*80)
    print("STARTING FEATURE ACTIVATION GATHERING")
    print("="*80)
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Hookpoint: {hookpoint}")
    print(f"Pipeline path: {pipe_path}")
    print(f"Save directory: {save_dir}")
    print(f"Steps: {steps}")
    print(f"Seed: {seed}")
    print(f"Resume mode: {resume}")
    print("="*80)
    
    try:
        # Create directories
        print("\n[1/6] Creating directories...")
        os.makedirs(save_dir, exist_ok=True)
        temp_dir = os.path.join(save_dir, "temp_tensors")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"✓ Save directory: {save_dir}")
        print(f"✓ Temp directory: {temp_dir}")
        
        # Setup object prompts dictionary
        print("\n[2/6] Loading prompts...")
        cls_prompts_dict = {class_avail: [] for class_avail in class_available}
        
        for class_avail in class_available:
            prompt_file_path = os.path.join(
                os.path.dirname(SCRIPT_DIR),
                "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                f"sd_prompt_{class_avail}.txt",
            )
            
            if not os.path.exists(prompt_file_path):
                print(f"❌ ERROR: Prompt file not found: {prompt_file_path}")
                raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")
            
            with open(prompt_file_path, "r") as prompt_file:
                prompts = prompt_file.readlines()
                prompt = [p.strip() for p in prompts]
                cls_prompts_dict[class_avail].extend(prompt)
            
            print(f"✓ Loaded {len(prompt)} prompts for {class_avail}")
        
        print(f"\n✓ Total classes: {len(class_available)}")
        print(f"✓ Classes: {class_available}")

        # Check for existing temp files if resume is True
        print("\n[3/6] Checking for existing temporary files...")
        cls_latents_paths = {}
        classes_to_process = []
        
        if resume:
            for class_avail in class_available:
                temp_file = os.path.join(temp_dir, f"{class_avail}_latents.pt")
                if os.path.exists(temp_file):
                    file_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
                    print(f"✓ Found existing: {class_avail} ({file_size_mb:.2f} MB)")
                    cls_latents_paths[class_avail] = temp_file
                else:
                    classes_to_process.append(class_avail)
            print(f"\n✓ {len(cls_latents_paths)} classes already processed")
            print(f"✓ {len(classes_to_process)} classes remaining")
            if classes_to_process:
                print(f"  Classes to process: {classes_to_process}")
        else:
            classes_to_process = list(class_available)
            print(f"Resume disabled - will process all {len(classes_to_process)} classes")
        
        # Only initialize pipeline if we have classes to process
        if not classes_to_process:
            print("\n✓ All classes already processed! Skipping to consolidation...")
        else:
            print(f"\n[4/6] Processing {len(classes_to_process)} classes...")
            
            # Load SAE
            print("\nLoading SAE...")
            sae_path = os.path.join(checkpoint_path, hookpoint)
            if not os.path.exists(sae_path):
                print(f"❌ ERROR: SAE checkpoint not found: {sae_path}")
                raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")
            
            sae = Sae.load_from_disk(sae_path, device="cuda").eval()
            sae = sae.to(dtype=torch.float16)
            sae.cfg.batch_topk = False
            sae.cfg.sample_topk = False
            print(f"✓ SAE loaded from {sae_path}")
            print(f"  SAE d_in: {sae.d_in}, num_latents: {sae.num_latents}")

            # Detect model type from model_index.json
            print("\nDetecting model type...")
            model_index_path = os.path.join(pipe_path, "model_index.json")
            is_sdxl = False
            
            if not os.path.exists(model_index_path):
                print(f"⚠️  WARNING: model_index.json not found at {model_index_path}")
                print("  Assuming SD1.5 model")
            else:
                with open(model_index_path, 'r') as f:
                    model_index = json.load(f)
                is_sdxl = "text_encoder_2" in model_index
                
            if is_sdxl:
                print("🎯 Detected SDXL model - using HookedStableDiffusionXLPipeline")
                PipelineClass = HookedStableDiffusionXLPipeline
            else:
                print("🎯 Detected SD1.5 model - using HookedStableDiffusionPipeline")
                PipelineClass = HookedStableDiffusionPipeline

            print("\nLoading diffusion pipeline...")
            pipe = PipelineClass.from_pretrained(
                pipe_path,
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to("cuda")
            
            if is_xformers_available():
                print("✓ Enabling xFormers memory efficient attention")
                pipe.unet.enable_xformers_memory_efficient_attention()
            else:
                print("⚠️  xFormers not available")
            
            print("✓ Pipeline loaded successfully")

            # Process classes
            progress_bar = tqdm.tqdm(classes_to_process, total=len(classes_to_process))
            generator = torch.Generator(device="cpu").manual_seed(seed)
            
            for idx, class_avail in enumerate(progress_bar):
                progress_bar.set_description(f"Processing class: {class_avail}")
                print(f"\n{'='*60}")
                print(f"Processing class {idx+1}/{len(classes_to_process)}: {class_avail}")
                print(f"{'='*60}")
                
                prompts = cls_prompts_dict[class_avail]
                print(f"Number of prompts: {len(prompts)}")
                
                # Clear memory before processing each class
                torch.cuda.empty_cache()
                gc.collect()
                
                try:
                    print("Running pipeline with cache...")
                    _, acts_cache = pipe.run_with_cache(
                        prompt=prompts,
                        generator=generator,
                        num_inference_steps=steps,
                        save_input=False,
                        save_output=True,
                        positions_to_cache=[hookpoint],
                        guidance_scale=9.0,
                        output_type="latent",
                    )
                    print("✓ Pipeline completed")
                    
                    activations = acts_cache["output"][hookpoint].cpu()
                    print(f"✓ Activations shape: {activations.shape}")
                    
                    assert activations.shape[0] == len(prompts), \
                        f"Batch size mismatch: {activations.shape[0]} != {len(prompts)}"
                    assert activations.shape[1] == steps, \
                        f"Steps mismatch: {activations.shape[1]} != {steps}"
                    print("✓ Assertions passed")
                    
                    # Process SAE latents
                    print("Processing SAE latents...")
                    sae_latents = []
                    with torch.no_grad():
                        for i in range(len(prompts)):
                            sae_in = activations[i].reshape(steps, -1, sae.d_in)
                            top_acts, top_indices = sae.encode(sae_in.to(sae.device))
                            sae_out = torch.zeros(
                                (top_acts.shape[0], sae.num_latents),
                                device=sae.device,
                                dtype=top_acts.dtype,
                            ).scatter(-1, top_indices, top_acts)
                            sae_out = sae_out.reshape(steps, -1, sae.num_latents).cpu()
                            sae_latents.append(sae_out.mean(1).to(dtype=torch.float16))
                    
                    print(f"✓ Processed {len(sae_latents)} SAE latents")
                    
                    # Save latents to temporary file
                    latent_tensor = torch.stack(sae_latents)
                    print(f"✓ Stacked latents shape: {latent_tensor.shape}")
                    
                    latent_path = os.path.join(temp_dir, f"{class_avail}_latents.pt")
                    save_tensor_to_disk(latent_tensor, latent_path)
                    cls_latents_paths[class_avail] = latent_path
                    
                    file_size_mb = os.path.getsize(latent_path) / (1024 * 1024)
                    print(f"✓ Saved {class_avail} latents ({file_size_mb:.2f} MB)")
                    
                    # Free up memory
                    del activations, sae_latents, latent_tensor, acts_cache
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("✓ Memory cleared")
                    
                except Exception as e:
                    print(f"\n❌ ERROR processing class {class_avail}:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    print("\nFull traceback:")
                    traceback.print_exc()
                    raise

        # Consolidation step
        print("\n" + "="*80)
        print("[5/6] CONSOLIDATING ALL CLASS LATENTS")
        print("="*80)
        cls_latents_dict = {}
        
        # Load all temp files (including previously processed ones)
        for class_avail in class_available:
            temp_file = os.path.join(temp_dir, f"{class_avail}_latents.pt")
            if os.path.exists(temp_file):
                file_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
                print(f"Loading {class_avail} ({file_size_mb:.2f} MB)...")
                try:
                    cls_latents_dict[class_avail] = load_tensor_from_disk(temp_file)
                except Exception as e:
                    print(f"❌ ERROR loading {class_avail}: {e}")
                    raise
            else:
                print(f"⚠️  WARNING: Missing {class_avail} at {temp_file}")
        
        # Verify all classes are present
        print(f"\n✓ Loaded {len(cls_latents_dict)}/{len(class_available)} classes")
        
        if len(cls_latents_dict) != len(class_available):
            missing = set(class_available) - set(cls_latents_dict.keys())
            print(f"\n❌ ERROR: Only {len(cls_latents_dict)}/{len(class_available)} classes available!")
            print(f"Missing classes: {missing}")
            raise RuntimeError(f"Missing {len(missing)} classes: {missing}")
        
        # Save final pickle file
        print("\n[6/6] Saving final pickle file...")
        latents_file = os.path.join(save_dir, f"cls_latents_dict_{hookpoint}.pkl")
        print(f"Output file: {latents_file}")
        
        try:
            with open(latents_file, "wb") as f:
                pickle.dump(cls_latents_dict, f)
            
            file_size_mb = os.path.getsize(latents_file) / (1024 * 1024)
            print(f"\n✅ Successfully saved to {latents_file}")
            print(f"   File size: {file_size_mb:.2f} MB")
            
            # Verify file is readable
            print("\nVerifying saved file...")
            with open(latents_file, "rb") as f:
                test_load = pickle.load(f)
            print(f"✓ File verification passed - contains {len(test_load)} classes")
            
        except Exception as e:
            print(f"\n❌ ERROR saving pickle file: {e}")
            traceback.print_exc()
            raise
        
        print("\n" + "="*80)
        print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Final output: {latents_file}")
        print("="*80)
    
    except Exception as e:
        print("\n" + "="*80)
        print("❌ SCRIPT FAILED!")
        print("="*80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("="*80)
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(main)