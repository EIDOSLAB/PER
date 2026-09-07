import os
import pickle
import sys
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
    steps=4,
    guidance_scale=0.0,
    output_dir="eval_results/mu_results/class20/",
    start_from=0,
    start_timestep=0,
    use_sae=True,    # Flag to enable/disable SAE unlearning
    residual_sae=False,  # Set True for SAEs trained on (output - input), e.g. RIEBench
):
    """
    Generate images with optional SAE-based unlearning.
    Automatically detects whether to use SDXL or SD1.5 pipeline.
    
    Args:
        pipe_checkpoint: Path to the pretrained diffusion model checkpoint
        hookpoint: Position in the model where SAE hooks are applied (required if use_sae=True)
        class_latents_path: Path to class latents pickle file (required if use_sae=True)
        sae_checkpoint: Path to SAE checkpoint (required if use_sae=True)
        class_params_path: Path to class parameters file (required if use_sae=True)
        seed: Random seed for generation
        steps: Number of inference steps
        guidance_scale: Classifier-free guidance scale
        output_dir: Directory to save generated images
        start_from: Index to start processing from (for resuming)
        start_timestep: Timestep from which to start applying SAE unlearning
        use_sae: Whether to apply SAE-based unlearning (default: True)
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
    generator = torch.Generator(device=device).manual_seed(seed)  # Changed from "cpu" to device
    
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
        class_params = torch.load(class_params_path)
        
        if accelerator.is_main_process:
            print(f"SAE unlearning enabled - will be applied from timestep {start_timestep} to {steps}")
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
        sae_status = "with SAE" if use_sae else "without SAE"
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
                local_prompts = []
                for object_class, theme in local_classes_themes:
                    if theme == "":
                        local_prompts.append(f"An image of {object_class}.")
                    else:
                        local_prompts.append(
                            f"An image of {object_class} in {theme.replace('_', ' ')} style."
                        )
                
                # Setup hooks only if SAE is enabled
                if use_sae:
                    steering_hooks = {}
                    steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
                        concept_to_unlearn=[class_to_unlearn],
                        percentile=class_params[class_to_unlearn]["percentile"],
                        multiplier=class_params[class_to_unlearn]["multiplier"],
                        feature_importance_fn=compute_feature_importance,
                        concept_latents_dict=class_latents_dict,
                        sae=sae,
                        steps=steps,
                        preserve_error=True,
                        start_timestep=start_timestep,
                        guidance_scale=guidance_scale,
                        residual_sae=residual_sae,
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
                
                for object_class, theme in local_classes_themes:
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