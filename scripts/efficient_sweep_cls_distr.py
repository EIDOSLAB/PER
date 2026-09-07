"""
Memory-optimized script for hyperparameter sweep for object unlearning.
"""
import os
import pickle
import sys
import gc
import json
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from packaging import version
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import utils.hooks as hooks
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline, HookedStableDiffusionXLPipeline
from SAE.sae import Sae
from SAE.unlearning_utils import compute_feature_importance

sys.path.append("..")

import fire

# from UnlearnCanvas_resources.const import class_available, theme_available
from UnlearnCanvas_resources.const import class_available_subsample as class_available
from UnlearnCanvas_resources.const import theme_available_subsample as theme_available

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


def empty_cache():
    """Helper function to clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_class_prompts(limit_themes=50):
    """Load and prepare class prompts (moved to separate function to improve memory management)"""
    class_prompt_dict = {class_: [] for class_ in class_available}
    for class_to_unlearn in class_available:
        with open(
            os.path.join(
                os.path.dirname(SCRIPT_DIR),
                "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                f"sd_prompt_{class_to_unlearn}.txt",
            ),
            "r",
        ) as prompt_file:
            prompts = prompt_file.readlines()
            for i, theme in enumerate(theme_available):
                if i >= limit_themes:
                    break
                if theme != "Seed_Images":
                    theme_prompt = prompts[i].strip()
                    theme_prompt = (
                        theme_prompt
                        if not theme_prompt.endswith(".")
                        else theme_prompt[:-1]
                    )
                    theme_prompt = f"{theme_prompt} in {theme.replace('_', ' ')} style."
                    class_prompt_dict[class_to_unlearn].append(theme_prompt)
    return class_prompt_dict


def process_batch(model, batch_prompts, batch_classes, class_to_unlearn, percentile, multiplier,
                 hookpoint, sae, class_latents_dict, steps, seed, guidance_scale, generator,
                 residual_sae=False):
    """Process a single batch of prompts"""
    steering_hooks = {}
    steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
        concept_to_unlearn=[class_to_unlearn],
        percentile=percentile,
        multiplier=multiplier,
        feature_importance_fn=compute_feature_importance,
        concept_latents_dict=class_latents_dict,
        sae=sae,
        steps=steps,
        seed=seed,
        preserve_error=True,
        guidance_scale=guidance_scale,
        residual_sae=residual_sae,
    )
    
    with torch.no_grad():
        images = model.run_with_hooks(
            prompt=batch_prompts,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            position_hook_dict=steering_hooks,
        )
    
    return images, batch_classes


def main(
    pipe_checkpoint,
    hookpoint,
    class_latents_path,
    sae_checkpoint,
    seed=188,
    steps=4,
    percentiles=[99.99, 99.995, 99.999],
    multipliers=[-1.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0],
    guidance_scale=0.0,
    output_dir="sweep_results/mu_results/class20/",
    limit_themes=50,
    batch_size=100,  # Added parameter for controlling batch size
    residual_sae=False,  # Set True for SAEs trained on (output - input), e.g. RIEBench
):
    accelerator = Accelerator()
    device = accelerator.device

    # Detect model type from model_index.json
    model_index_path = os.path.join(pipe_checkpoint, "model_index.json")
    is_sdxl = False
    
    if os.path.exists(model_index_path):
        with open(model_index_path, 'r') as f:
            model_index = json.load(f)
        # SDXL has text_encoder_2, SD1.5 doesn't
        is_sdxl = "text_encoder_2" in model_index
        
    if is_sdxl:
        if accelerator.is_main_process:
            print("🎯 Detected SDXL model - using HookedStableDiffusionXLPipeline")
        PipelineClass = HookedStableDiffusionXLPipeline
    else:
        if accelerator.is_main_process:
            print("🎯 Detected SD1.5 model - using HookedStableDiffusionPipeline")
        PipelineClass = HookedStableDiffusionPipeline

    # Load model with memory optimization options
    model = PipelineClass.from_pretrained(
        pipe_checkpoint,
        torch_dtype=torch.float16,  # Using fp16 to save memory
        safety_checker=None,
        # Add memory optimization options
        local_files_only=True if os.path.exists(pipe_checkpoint) else False,
    )
    
    # Enable memory efficient attention
    if is_xformers_available():
        import xformers
        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            if accelerator.is_main_process:
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        model.enable_xformers_memory_efficient_attention()
    
    model = model.to(device)

    # Convert VAE to float32 to prevent black images with SDXL
    model.pipe.vae = model.pipe.vae.to(dtype=torch.float32)
    if accelerator.is_main_process:
        print(f"✓ VAE converted to float32 (dtype: {model.pipe.vae.dtype})")
    
    # For SDXL, enable VAE optimizations
    if is_sdxl and hasattr(model.pipe, 'enable_vae_slicing'):
        model.pipe.enable_vae_slicing()
    
    # Disable VAE tiling if it causes issues
    if hasattr(model.pipe, 'disable_vae_tiling'):
        model.pipe.disable_vae_tiling()
    
    # Set up generator
    seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    if accelerator.is_main_process:
        print('Loading SAE...')
    
    # Load SAE
    sae = load_sae(sae_checkpoint, hookpoint, device)
    
    if accelerator.is_main_process:
        print('Loading class latents...')
    
    # Load class latents - use context manager to control file handle
    with open(class_latents_path, "rb") as f:
        class_latents_dict = pickle.load(f)
    
    empty_cache()  # Clear memory after loading models
    
    if accelerator.is_main_process:
        print('Preparing class prompts...')
    
    # Get class prompts
    class_prompt_dict = get_class_prompts(limit_themes)
    
    # Set up progress bar
    total_iterations = len(multipliers) * len(class_available) * len(percentiles)
    if accelerator.is_main_process:
        progress_bar = tqdm(total=total_iterations)
    
    # Main processing loop
    for multiplier in multipliers:
        for percentile in percentiles:
            for class_to_unlearn in class_available:
                if accelerator.is_main_process:
                    progress_bar.set_description(
                        f"Multiplier: {multiplier} Percentile: {percentile} Class: {class_to_unlearn}"
                    )
                
                # Create output directory
                output_path = os.path.join(
                    output_dir,
                    f"percentile_{percentile}_multiplier_{multiplier}/{class_to_unlearn}",
                )
                os.makedirs(output_path, exist_ok=True)
                
                # Prepare all prompts
                all_prompts = [
                    (class_name, prompt)
                    for class_name, prompts in class_prompt_dict.items()
                    for prompt in prompts
                ]
                
                # Process in distributed manner across GPUs/processes
                all_images = []
                all_classes = []
                
                with accelerator.split_between_processes(all_prompts) as local_tuples:
                    local_prompts = [prompt.strip() for _, prompt in local_tuples]
                    local_classes = [class_name for class_name, _ in local_tuples]
                    
                    # Process in smaller batches to save memory
                    for i in range(0, len(local_prompts), batch_size):
                        batch_prompts = local_prompts[i:i+batch_size]
                        batch_classes = local_classes[i:i+batch_size]
                        
                        batch_images, batch_classes = process_batch(
                            model, batch_prompts, batch_classes, class_to_unlearn, percentile,
                            multiplier, hookpoint, sae, class_latents_dict, steps, seed,
                            guidance_scale, generator, residual_sae=residual_sae
                        )
                        
                        all_images.extend(batch_images)
                        all_classes.extend(batch_classes)
                        
                        # Clear memory after each batch
                        empty_cache()
                
                # Gather results across processes
                accelerator.wait_for_everyone()
                gathered_images = gather_object(all_images)
                gathered_classes = gather_object(all_classes)
                
                # Save results
                if accelerator.is_main_process:
                    for i, (img, object_class) in enumerate(zip(gathered_images, gathered_classes)):
                        img.save(
                            os.path.join(
                                output_path,
                                f"{object_class}_seed{seed}_{i}.jpg",
                            )
                        )
                    progress_bar.update(1)
                
                # Final memory cleanup for this iteration
                empty_cache()
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("Processing complete!")


if __name__ == "__main__":
    fire.Fire(main)