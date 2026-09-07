import os
import pickle
import sys
import json
import math

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from packaging import version
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import utils.noise_injection_hooks as hooks  # CHANGED: use noise_injection_hooks
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline, HookedStableDiffusionXLPipeline
from SAE.sae import Sae
from SAE.unlearning_utils import compute_feature_importance

sys.path.append("..")

import fire

from UnlearnCanvas_resources.const import (
    class_available,
    theme_available,
)

# from UnlearnCanvas_resources.const import class_available_subsample as class_available
# from UnlearnCanvas_resources.const import theme_available_subsample as theme_available

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

from diffusers.utils.import_utils import is_xformers_available


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_sae(sae_checkpoint, hookpoint, device):
    sae = Sae.load_from_disk(
        os.path.join(sae_checkpoint, hookpoint), device=device
    ).eval()
    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False
    return sae


def main(
    pipe_checkpoint,
    hookpoint=None,
    class_latents_path=None,
    sae_checkpoint=None,
    class_params_path=None,
    seed=188,
    steps=100,
    guidance_scale=9.0,
    output_dir="eval_results/mu_results/class20/",
    start_from=0,
    start_timestep=0,
    end_timestep=None,  # NEW: timestep to end applying SAE unlearning (None means apply until end)
    use_sae=True,    # Flag to enable/disable SAE unlearning
    noise_scale=1.0,  # NEW: scale of noise to add (replaces multiplier)
    noise_type='gaussian',  # NEW: type of noise ('gaussian' or 'uniform')
    noise_mode='add',  # NEW: how to apply noise - see details below
    padding=1,  # NEW: padding distance for 'replace_with_neighbor_padded' mode
):
    """
    Generate images with optional SAE-based unlearning using noise injection.
    Automatically detects whether to use SDXL or SD1.5 pipeline.
    
    Args:
        pipe_checkpoint: Path to the pretrained diffusion model checkpoint
        hookpoint: Position in the model where SAE hooks are applied (required if use_sae=True)
        class_latents_path: Path to class latents pickle file (required if use_sae=True)
        sae_checkpoint: Path to SAE checkpoint (required if use_sae=True)
        class_params_path: Path to class parameters file (required if use_sae=True) - contains percentile info
        seed: Random seed for generation
        steps: Number of inference steps
        guidance_scale: Classifier-free guidance scale
        output_dir: Directory to save generated images
        start_from: Index to start processing from (for resuming)
        start_timestep: Timestep value to start applying SAE unlearning (high value = early in denoising, e.g., 100)
        end_timestep: Timestep value to stop applying SAE unlearning (low value = late in denoising, e.g., 0). 
                     None means apply all the way to timestep 0 (end of denoising).
                     Note: start_timestep must be >= end_timestep because timesteps count DOWN during denoising.
        use_sae: Whether to apply SAE-based unlearning (default: True)
        noise_scale: Scale of noise to add when steering (default: 1.0)
        noise_type: Type of noise to add - 'gaussian' or 'uniform' (default: 'gaussian')
        noise_mode: How to apply noise (default: 'add'):
            - 'add': Add scaled noise to original activations (output = input + noise)
            - 'replace': Replace with pure noise (output = noise)
            - 'replace_with_neighbor': Replace concept patches with random non-concept patches from the same image
            - 'replace_with_closest': Replace concept patches with spatially closest non-concept patches from the same image
            - 'replace_with_neighbor_padded': Like 'replace_with_neighbor', but also replaces patches within 'padding' distance of concept patches
            - 'replace_with_neighbor_padded_averaged': Like 'replace_with_neighbor_padded', but replaces with AVERAGED K-nearest neighbors for smoother, style-preserving replacement (recommended for best style preservation)
        padding: Distance (in patches) to expand concept regions when using 'replace_with_neighbor_padded' mode (default: 1)
                Using Chebyshev distance (max of absolute differences in row/col coordinates)
                Example: padding=1 means patches at distance 1 (including diagonals) are also replaced
    """
    accelerator = Accelerator()
    device = accelerator.device

    # Validate arguments when SAE is enabled
    if use_sae:
        if hookpoint is None or class_latents_path is None or sae_checkpoint is None or class_params_path is None:
            raise ValueError(
                "When use_sae=True, you must provide: hookpoint, class_latents_path, "
                "sae_checkpoint, and class_params_path"
            )
        
        # Set default end_timestep if not provided
        if end_timestep is None:
            end_timestep = 0  # Default to applying all the way to the end (timestep 0)
        
        # Validate timestep range
        # Note: timesteps count DOWN during denoising (high value at start -> 0 at end)
        # start_timestep should be >= end_timestep (e.g., start=100, end=50 means apply from t=100 down to t=50)
        if start_timestep < 0 or start_timestep >= steps:
            raise ValueError(f"start_timestep must be between 0 and {steps-1}, got {start_timestep}")
        if end_timestep < 0 or end_timestep >= steps:
            raise ValueError(f"end_timestep must be between 0 and {steps-1}, got {end_timestep}")
        if start_timestep < end_timestep:
            raise ValueError(f"start_timestep ({start_timestep}) must be >= end_timestep ({end_timestep}) because timesteps count DOWN during denoising")
        
        # Validate padding
        if padding < 0:
            raise ValueError(f"padding must be >= 0, got {padding}")
        if noise_mode in ['replace_with_neighbor_padded', 'replace_with_neighbor_padded_averaged'] and padding == 0:
            print("Warning: padding=0 for padded mode - this is equivalent to non-padded 'replace_with_neighbor'")

    # Detect model type from model_index.json
    model_index_path = os.path.join(pipe_checkpoint, "model_index.json")
    is_sdxl = False
    
    if os.path.exists(model_index_path):
        with open(model_index_path, 'r') as f:
            model_index = json.load(f)
        # SDXL has text_encoder_2, SD1.5 doesn't
        is_sdxl = "text_encoder_2" in model_index
    
    # Load appropriate pipeline based on detected model type
    if is_sdxl:
        if accelerator.is_main_process:
            print("🎯 Detected SDXL model - using HookedStableDiffusionXLPipeline")
        PipelineClass = HookedStableDiffusionXLPipeline
        try:
            model = PipelineClass.from_pretrained(
                pipe_checkpoint,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
        except:
            # Fallback without variant
            model = PipelineClass.from_pretrained(
                pipe_checkpoint,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
    else:
        if accelerator.is_main_process:
            print("🎯 Detected SD1.5 model - using HookedStableDiffusionPipeline")
        PipelineClass = HookedStableDiffusionPipeline
        model = PipelineClass.from_pretrained(
            pipe_checkpoint,
            torch_dtype=torch.float16,
        )
    
    # Disable safety checker if it exists
    if hasattr(model.pipe, 'safety_checker'):
        model.pipe.safety_checker = None
    
    model = model.to(device)

    model.pipe.vae = model.pipe.vae.to(dtype=torch.float32)
    if accelerator.is_main_process:
        print(f"✓ VAE converted to float32 (dtype: {model.pipe.vae.dtype})")

    # For SDXL, enable VAE optimizations
    if is_sdxl and hasattr(model.pipe, 'enable_vae_slicing'):
        model.pipe.enable_vae_slicing()
    
    # Disable VAE tiling if it causes issues
    if hasattr(model.pipe, 'disable_vae_tiling'):
        model.pipe.disable_vae_tiling()

    if is_xformers_available():
        import xformers

        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            if accelerator.is_main_process:
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        model.enable_xformers_memory_efficient_attention()

    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    if accelerator.is_main_process:
        print(f"Generator device: {device}")
        print(f"Model dtype: {model.pipe.unet.dtype}")
        if hasattr(model.pipe, 'vae'):
            print(f"VAE dtype: {model.pipe.vae.dtype}")
    
    # Load SAE components only if use_sae is True
    if use_sae:
        sae = load_sae(sae_checkpoint, hookpoint, device)
        with open(class_latents_path, "rb") as f:
            class_latents_dict = pickle.load(f)
        
        # Load class_params (now always required to get percentile)
        class_params = torch.load(class_params_path)
        
        # Get total number of latents from SAE
        total_latents = sae.num_latents
        
        if accelerator.is_main_process:
            print(f"SAE noise injection enabled:")
            print(f"  - Total SAE latents: {total_latents}")
            print(f"  - Noise type: {noise_type}")
            print(f"  - Noise scale: {noise_scale}")
            print(f"  - Noise mode: {noise_mode}")
            if noise_mode == 'replace_with_neighbor_padded_averaged':
                print(f"    (Replaces concept patches + {padding}-pixel padding with AVERAGED K-nearest non-concept patches - BEST for style preservation)")
            elif noise_mode == 'replace_with_neighbor_padded':
                print(f"    (Replaces concept patches + {padding}-pixel padding with random non-concept patches)")
            elif noise_mode == 'replace_with_closest':
                print(f"    (Replaces concept patches with spatially closest non-concept patches)")
            elif noise_mode == 'replace_with_neighbor':
                print(f"    (Replaces concept patches with random non-concept patches)")
            elif noise_mode == 'replace':
                print(f"    (Replaces concept patches with pure noise)")
            elif noise_mode == 'add':
                print(f"    (Adds scaled noise to concept patches)")
            print(f"  - Using percentile from class_params to calculate top-K")
            print(f"  - Timestep range: [{start_timestep}, {end_timestep}]")
            print(f"    (Timesteps count DOWN during denoising: unlearning applied when timestep is between {start_timestep} and {end_timestep})")
    else:
        sae = None
        class_latents_dict = None
        class_params = None
        if accelerator.is_main_process:
            print("SAE unlearning disabled - generating images with base model only")

    theme_avail = [t for t in theme_available if t != "Seed_Images"]
    
    # Skip already processed classes if resuming
    classes_to_process = class_available[:]
    
    if accelerator.is_main_process:
        pipeline_type = "SDXL" if is_sdxl else "SD1.5"
        sae_status = f"with SAE (noise injection)" if use_sae else "without SAE"
        print(f"Using {pipeline_type} pipeline {sae_status}")
        print(f"Starting from index {start_from}, processing {len(classes_to_process)} classes")
        print(f"Classes to process: {classes_to_process}")
    
    progress_bar = tqdm(
        classes_to_process,
        total=len(classes_to_process),
        disable=not accelerator.is_main_process,
        initial=0,
    )
    
    for class_to_unlearn in progress_bar:
        if accelerator.is_main_process:
            if use_sae:
                progress_bar.set_description(f"Unlearning {class_to_unlearn}")
            else:
                progress_bar.set_description(f"Generating {class_to_unlearn}")
                
        output_path = os.path.join(output_dir, f"{class_to_unlearn}")
        os.makedirs(output_path, exist_ok=True)
        
        # Check if this class was already completed
        if start_from > 0:
            expected_files = len(theme_avail) * len(class_available) + len(class_available)
            existing_files = len([f for f in os.listdir(output_path) if f.endswith(f'_seed{seed}.jpg')])
            
            if existing_files >= expected_files and accelerator.is_main_process:
                print(f"Skipping {class_to_unlearn} - already completed ({existing_files} files found)")
                continue
        
        for test_theme in theme_avail:
            input_classes = []
            input_themes = []
            class_theme_pairs = [(c, test_theme) for c in class_available] + [
                (c, "") for c in class_available
            ]
            
            with accelerator.split_between_processes(class_theme_pairs) as local_classes_themes:
                # Filter out pairs that already have generated images (check on each process)
                local_classes_themes_filtered = []
                skipped_count = 0
                for object_class, theme in local_classes_themes:
                    if theme == "":
                        filename = f"{object_class}_seed{seed}.jpg"
                    else:
                        filename = f"{theme}_{object_class}_seed{seed}.jpg"
                    
                    filepath = os.path.join(output_path, filename)
                    if os.path.exists(filepath):
                        skipped_count += 1
                    else:
                        local_classes_themes_filtered.append((object_class, theme))
                
                if skipped_count > 0 and accelerator.is_main_process:
                    print(f"Skipping {skipped_count} already existing images for {class_to_unlearn} in {test_theme}")
                
                # Skip generation if nothing to generate on this process
                if len(local_classes_themes_filtered) == 0:
                    local_prompts = []
                    images = []
                else:
                    local_prompts = []
                    for object_class, theme in local_classes_themes_filtered:
                        if theme == "":
                            local_prompts.append(f"An image of {object_class}.")
                        else:
                            local_prompts.append(
                                f"An image of {object_class} in {theme.replace('_', ' ')} style."
                            )
                    
                    # Setup hooks only if SAE is enabled
                    if use_sae:
                        # Calculate top_k from percentile for this specific concept
                        concept_percentile = class_params[class_to_unlearn]["percentile"]
                        total_latents = sae.num_latents
                        
                        # Convert percentile to top_k
                        # percentile 99.999 means we keep (100 - 99.999) = 0.001% = top 0.001% of latents
                        # So top_k = ceil(total_latents * (100 - percentile) / 100)
                        top_k_for_concept = math.ceil(total_latents * (100 - concept_percentile) / 100)
                        
                        if accelerator.is_main_process and test_theme == theme_avail[0]:  # Print once per concept
                            print(f"\n{'='*80}")
                            print(f"Concept: {class_to_unlearn}")
                            print(f"  Percentile: {concept_percentile}")
                            print(f"  Total latents: {total_latents}")
                            print(f"  Top-K latents to use: {top_k_for_concept}")
                            print(f"  Percentage: {100 * top_k_for_concept / total_latents:.4f}%")
                            print(f"{'='*80}\n")
                        
                        steering_hooks = {}
                        steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
                            concept_to_unlearn=[class_to_unlearn],
                            percentile=None,  # Don't use percentile-based selection
                            top_k=top_k_for_concept,  # Use computed top_k
                            noise_scale=noise_scale,
                            feature_importance_fn=compute_feature_importance,
                            concept_latents_dict=class_latents_dict,
                            sae=sae,
                            steps=steps,
                            preserve_error=True,
                            start_timestep=start_timestep,
                            end_timestep=end_timestep,
                            noise_type=noise_type,
                            noise_mode=noise_mode,
                            padding=padding,
                        )
                    else:
                        steering_hooks = {}
                    
                    with torch.no_grad():
                        images = model.run_with_hooks(
                            prompt=local_prompts,
                            generator=generator,
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                            position_hook_dict=steering_hooks,
                        )
                
                # Collect the classes and themes that were actually generated
                for object_class, theme in local_classes_themes_filtered:
                    input_classes.extend([object_class])
                    input_themes.extend([theme])
            
            accelerator.wait_for_everyone()
            images = gather_object(images)
            input_classes = gather_object(input_classes)
            input_themes = gather_object(input_themes)
            
            if accelerator.is_main_process:
                for img, object_class, theme in zip(images, input_classes, input_themes):
                    if theme == "":
                        img.save(
                            os.path.join(
                                output_path,
                                f"{object_class}_seed{seed}.jpg",
                            )
                        )
                    else:
                        img.save(
                            os.path.join(
                                output_path,
                                f"{theme}_{object_class}_seed{seed}.jpg",
                            )
                        )
        
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)