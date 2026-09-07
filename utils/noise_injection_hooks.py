import numpy as np
import torch
import torch.nn.functional as F

from SAE.unlearning_utils import get_percentile_threshold


class SAEReconstructHook:
    def __init__(
        self,
        sae,
    ):
        self.sae = sae

    @torch.no_grad()
    def __call__(self, module, input, output):
        output1, output2 = output[0].chunk(2)
        # reshape to SAE input shape
        output1 = output1.permute(0, 2, 3, 1).reshape(
            len(output1), output1.shape[-1] * output1.shape[-2], -1
        )
        output2 = output2.permute(0, 2, 3, 1).reshape(
            len(output2), output2.shape[-1] * output2.shape[-2], -1
        )
        output_cat = torch.cat([output1, output2], dim=0)
        sae_input, _, _ = self.sae.preprocess_input(output_cat)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        sae_out = (latents @ self.sae.W_dec) + self.sae.b_dec
        sae_out1 = sae_out[: output1.shape[1] * len(output1)]
        sae_out2 = sae_out[output1.shape[1] * len(output1) :]
        hook_output = torch.cat(
            [
                sae_out1.reshape(
                    len(output1),
                    int(np.sqrt(output1.shape[-2])),
                    int(np.sqrt(output1.shape[-2])),
                    -1,
                ).permute(0, 3, 1, 2),
                sae_out2.reshape(
                    len(output2),
                    int(np.sqrt(output2.shape[-2])),
                    int(np.sqrt(output2.shape[-2])),
                    -1,
                ).permute(0, 3, 1, 2),
            ],
            dim=0,
        )

        return (hook_output,)


class SAEMaskedUnlearningHook:
    def __init__(
        self,
        concept_to_unlearn,
        percentile=None,
        top_k=None,
        noise_scale=1.0,
        feature_importance_fn=None,
        concept_latents_dict=None,
        sae=None,
        steps=100,
        preserve_error=True,
        seed=None,
        start_timestep=0,
        end_timestep=None,
        noise_type='gaussian',
        noise_mode='add',
        padding=1,
        verbose=False,  # Control debug verbosity
    ):
        self.concept_to_unlearn = concept_to_unlearn
        self.percentile = percentile
        self.top_k = top_k
        self.noise_scale = noise_scale
        self.feature_importance_fn = feature_importance_fn
        self.concept_latents_dict = concept_latents_dict
        self.timestep_idx = 0
        self.sae = sae
        self.steps = steps
        self.preserve_error = preserve_error
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep if end_timestep is not None else steps
        self.noise_type = noise_type
        self.noise_mode = noise_mode
        self.padding = padding
        self.verbose = verbose
        
        # Validate inputs
        if percentile is None and top_k is None:
            raise ValueError("Must provide either 'percentile' or 'top_k'")
        if percentile is not None and top_k is not None:
            raise ValueError("Cannot provide both 'percentile' and 'top_k' - choose one")
        
        # Validate noise_mode
        if self.noise_mode not in ['add', 'replace', 'replace_with_neighbor', 'replace_with_closest', 'replace_with_neighbor_padded', 'replace_with_neighbor_padded_averaged']:
            raise ValueError(f"noise_mode must be 'add', 'replace', 'replace_with_neighbor', 'replace_with_closest', 'replace_with_neighbor_padded', or 'replace_with_neighbor_padded_averaged', got '{self.noise_mode}'")
        
        # Validate padding
        if self.padding < 0:
            raise ValueError(f"padding must be >= 0, got {self.padding}")
        if self.noise_mode in ['replace_with_neighbor_padded', 'replace_with_neighbor_padded_averaged'] and self.padding == 0 and self.verbose:
            print("Warning: padding=0 for padded mode - this is equivalent to non-padded 'replace_with_neighbor'")
        
        # Validate timestep range
        if self.start_timestep < 0 or self.start_timestep >= self.steps:
            raise ValueError(f"start_timestep must be between 0 and {self.steps-1}, got {self.start_timestep}")
        if self.end_timestep < 0 or self.end_timestep >= self.steps:
            raise ValueError(f"end_timestep must be between 0 and {self.steps-1}, got {self.end_timestep}")
        if self.start_timestep < self.end_timestep:
            raise ValueError(f"start_timestep ({self.start_timestep}) must be >= end_timestep ({self.end_timestep}) because timesteps count DOWN during denoising")
        
        # precompute the most important features for this theme on every timestep
        self.top_feature_idxs = []
        self.all_concept_avg_acts = []
        self.seed = seed
        
        # Only print initialization info if verbose is True
        if self.verbose:
            print(f'Padding distance: {self.padding}')
            if self.noise_mode == 'replace_with_neighbor_padded_averaged':
                print(f'Using averaged neighbors (K=5) for smoother, style-preserving replacement')
            if self.top_k is not None:
                print(f'Using top-K selection: K={self.top_k}')
            else:
                print(f'Using percentile selection: percentile={self.percentile}')
        
        # Precompute the most important features for each timestep
        for timestep in range(steps):
            timestep_feature_idxs = []
            timestep_all_concept_avg_acts = []
            for concept in self.concept_to_unlearn:
                feature_scores = self.feature_importance_fn(
                    self.concept_latents_dict, concept, timestep, seed=self.seed
                )
                feature_scores = feature_scores.float()
                
                if self.top_k is not None:
                    if self.top_k > len(feature_scores):
                        if self.verbose:
                            print(f"Warning: top_k={self.top_k} is larger than number of features={len(feature_scores)}. Using all features.")
                        top_feature_idxs = torch.arange(len(feature_scores))
                    else:
                        _, top_feature_idxs = torch.topk(feature_scores, self.top_k)
                else:
                    percentile_threshold = get_percentile_threshold(
                        feature_scores, self.percentile
                    )
                    top_feature_idxs = torch.where(feature_scores > percentile_threshold)[0]
                
                timestep_feature_idxs.append(top_feature_idxs)

                # precompute average activations of features on other concepts
                all_concept_avg_acts = torch.zeros((len(top_feature_idxs)))
                for concept_key in self.concept_latents_dict:
                    all_concept_avg_acts += self.concept_latents_dict[concept_key][
                        :, timestep, top_feature_idxs
                    ].mean(dim=0)
                all_concept_avg_acts /= len(self.concept_latents_dict)
                timestep_all_concept_avg_acts.append(all_concept_avg_acts)
            
            self.top_feature_idxs.append(torch.cat(timestep_feature_idxs))
            self.all_concept_avg_acts.append(torch.cat(timestep_all_concept_avg_acts))

    @torch.no_grad()
    def __call__(self, module, input, output):
        output1, output2 = output[0].chunk(2)
        # reshape to SAE input shape
        output1 = output1.permute(0, 2, 3, 1).reshape(
            len(output1), output1.shape[-1] * output1.shape[-2], -1
        )
        output2 = output2.permute(0, 2, 3, 1).reshape(
            len(output2), output2.shape[-1] * output2.shape[-2], -1
        )
        h, w = int(np.sqrt(output2.shape[-2])), int(np.sqrt(output2.shape[-2]))
        output_cat = torch.cat([output1, output2], dim=0)

        # Control debug output verbosity
        debug_this_step = self.verbose and (self.timestep_idx == 0 or self.timestep_idx % 20 == 0)

        # DEBUG: Print input statistics (reduced frequency)
        if self.timestep_idx == 0 and self.verbose:
            actual_timestep = self.steps - 1 - self.timestep_idx
            if self.top_k is not None:
                print(f"Top-K mode: K={self.top_k}")
            else:
                print(f"Percentile mode: percentile={self.percentile}")

        # encode activations to get latent features for detection
        sae_input, _, _ = self.sae.preprocess_input(output_cat)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        latents = latents.reshape(len(output_cat), -1, self.sae.num_latents)

        # DEBUG: Print latent statistics (reduced)
        if self.timestep_idx == 0 and self.verbose:
            if len(self.top_feature_idxs[self.timestep_idx]) > 0:
                print(f"Top feature indices (first 10): {self.top_feature_idxs[self.timestep_idx][:10].tolist()}")
            else:
                print("WARNING: No top features selected!")

        # Start with original activations
        steered_output = output_cat.clone()

        actual_timestep = self.steps - 1 - self.timestep_idx
        
        # Apply noise if actual_timestep is in range
        if actual_timestep <= self.start_timestep and actual_timestep >= self.end_timestep:
            # Check if important features exceed threshold
            mask = latents[
                :, :, self.top_feature_idxs[self.timestep_idx]
            ] > self.all_concept_avg_acts[self.timestep_idx].to(pre_acts.device)
            
            # spatial_mask shape: [batch, seq_len]
            spatial_mask = mask.any(dim=-1)
            
            num_affected = spatial_mask.sum().item()
            total_patches = spatial_mask.numel()
            batch_size = spatial_mask.shape[0]
            seq_len = spatial_mask.shape[1]
            
            # DEBUG: Print detection statistics (reduced frequency and no flush=True)
            if debug_this_step:
                selected_latents = latents[:, :, self.top_feature_idxs[self.timestep_idx]]
                thresholds = self.all_concept_avg_acts[self.timestep_idx].to(pre_acts.device)
                
                patches_per_image = num_affected / batch_size
                
                if num_affected == 0:
                    print("  ⚠️  WARNING: NO PATCHES DETECTED! Noise will NOT be applied!")
                else:
                    print(f"  ✓ Concept detected in patches")
            
            # Generate noise
            if self.noise_type == 'gaussian':
                noise = torch.randn(
                    steered_output.shape,
                    device=steered_output.device,
                    dtype=steered_output.dtype
                )
            elif self.noise_type == 'uniform':
                noise = torch.rand(
                    steered_output.shape,
                    device=steered_output.device,
                    dtype=steered_output.dtype
                ) * 2 - 1
            else:
                raise ValueError(f"Unknown noise_type: {self.noise_type}")
            
            noise = noise * self.noise_scale
            
            # Apply noise based on mode
            spatial_mask_expanded = spatial_mask.unsqueeze(-1)
            
            if self.noise_mode == 'add':
                steered_output = torch.where(
                    spatial_mask_expanded,
                    steered_output + noise,
                    steered_output
                )
            elif self.noise_mode == 'replace':
                steered_output = torch.where(
                    spatial_mask_expanded,
                    noise,
                    steered_output
                )
            elif self.noise_mode == 'replace_with_neighbor':
                # OPTIMIZED: Still needs per-batch loop, but inner operations are faster
                for b in range(batch_size):
                    image_mask = spatial_mask[b]
                    concept_patches = torch.where(image_mask)[0]
                    non_concept_patches = torch.where(~image_mask)[0]
                    
                    if len(concept_patches) > 0 and len(non_concept_patches) > 0:
                        num_concept = len(concept_patches)
                        random_indices = torch.randint(
                            0, len(non_concept_patches), 
                            (num_concept,),
                            device=steered_output.device
                        )
                        replacement_patches = non_concept_patches[random_indices]
                        steered_output[b, concept_patches] = steered_output[b, replacement_patches].clone()
                    elif len(concept_patches) > 0:
                        steered_output[b, concept_patches] = noise[b, concept_patches]
                        
            elif self.noise_mode == 'replace_with_neighbor_padded':
                # OPTIMIZED: Vectorized morphological dilation
                grid_size = int(np.sqrt(seq_len))
                
                if self.padding > 0:
                    # Reshape to 2D for efficient dilation
                    spatial_mask_2d = spatial_mask.view(batch_size, grid_size, grid_size).float()
                    
                    # Vectorized dilation using max pooling
                    kernel_size = 2 * self.padding + 1
                    mask_4d = spatial_mask_2d.unsqueeze(1)  # [B, 1, H, W]
                    
                    expanded_mask_4d = F.max_pool2d(
                        mask_4d,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=self.padding
                    )
                    expanded_mask_2d = expanded_mask_4d.squeeze(1) > 0  # [B, H, W]
                    expanded_mask_1d = expanded_mask_2d.view(batch_size, -1)  # [B, seq_len]
                else:
                    expanded_mask_1d = spatial_mask
                
                # Process each image
                for b in range(batch_size):
                    concept_patches = torch.where(expanded_mask_1d[b])[0]
                    non_concept_patches = torch.where(~expanded_mask_1d[b])[0]
                    
                    if len(concept_patches) > 0 and len(non_concept_patches) > 0:
                        num_to_replace = len(concept_patches)
                        random_indices = torch.randint(
                            0, len(non_concept_patches),
                            (num_to_replace,),
                            device=steered_output.device
                        )
                        replacement_patches = non_concept_patches[random_indices]
                        steered_output[b, concept_patches] = steered_output[b, replacement_patches].clone()
                    elif len(concept_patches) > 0:
                        steered_output[b, concept_patches] = noise[b, concept_patches]
                        
            elif self.noise_mode == 'replace_with_neighbor_padded_averaged':
                # OPTIMIZED: Vectorized dilation + batched distance computation
                grid_size = int(np.sqrt(seq_len))
                K_NEIGHBORS = 5
                
                if self.padding > 0:
                    # Vectorized dilation
                    spatial_mask_2d = spatial_mask.view(batch_size, grid_size, grid_size).float()
                    kernel_size = 2 * self.padding + 1
                    mask_4d = spatial_mask_2d.unsqueeze(1)
                    
                    expanded_mask_4d = F.max_pool2d(
                        mask_4d,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=self.padding
                    )
                    expanded_mask_2d = expanded_mask_4d.squeeze(1) > 0
                    expanded_mask_1d = expanded_mask_2d.view(batch_size, -1)
                else:
                    expanded_mask_1d = spatial_mask
                
                # Process each image
                for b in range(batch_size):
                    concept_patches = torch.where(expanded_mask_1d[b])[0]
                    non_concept_patches = torch.where(~expanded_mask_1d[b])[0]
                    
                    if len(concept_patches) > 0 and len(non_concept_patches) > 0:
                        # OPTIMIZED: Batch distance computation
                        # Convert to 2D coordinates
                        concept_coords = torch.stack([
                            concept_patches // grid_size,
                            concept_patches % grid_size
                        ], dim=1).float()
                        
                        non_concept_coords = torch.stack([
                            non_concept_patches // grid_size,
                            non_concept_patches % grid_size
                        ], dim=1).float()
                        
                        # OPTIMIZED: Use squared distances (avoid sqrt)
                        diff = concept_coords.unsqueeze(1) - non_concept_coords.unsqueeze(0)
                        squared_distances = (diff ** 2).sum(dim=-1)  # [num_concept, num_non_concept]
                        
                        # Find K nearest neighbors
                        k = min(K_NEIGHBORS, len(non_concept_patches))
                        _, k_nearest_indices = torch.topk(squared_distances, k, dim=1, largest=False)
                        
                        # OPTIMIZED: Vectorized averaging using gather
                        # Shape: [num_concept, k, d_in]
                        k_nearest_patches = non_concept_patches[k_nearest_indices]  # [num_concept, k]
                        
                        # Gather features: [num_concept, k, d_in]
                        gathered_features = steered_output[b, k_nearest_patches]
                        
                        # Average and assign
                        averaged_features = gathered_features.mean(dim=1)  # [num_concept, d_in]
                        steered_output[b, concept_patches] = averaged_features
                        
                    elif len(concept_patches) > 0:
                        steered_output[b, concept_patches] = noise[b, concept_patches]
                        
            elif self.noise_mode == 'replace_with_closest':
                # OPTIMIZED: Batched distance computation
                grid_size = int(np.sqrt(seq_len))
                
                for b in range(batch_size):
                    image_mask = spatial_mask[b]
                    concept_patches = torch.where(image_mask)[0]
                    non_concept_patches = torch.where(~image_mask)[0]
                    
                    if len(concept_patches) > 0 and len(non_concept_patches) > 0:
                        concept_coords = torch.stack([
                            concept_patches // grid_size,
                            concept_patches % grid_size
                        ], dim=1).float()
                        
                        non_concept_coords = torch.stack([
                            non_concept_patches // grid_size,
                            non_concept_patches % grid_size
                        ], dim=1).float()
                        
                        # OPTIMIZED: Use squared distances
                        diff = concept_coords.unsqueeze(1) - non_concept_coords.unsqueeze(0)
                        squared_distances = (diff ** 2).sum(dim=-1)
                        
                        closest_indices = squared_distances.argmin(dim=1)
                        replacement_patches = non_concept_patches[closest_indices]
                        steered_output[b, concept_patches] = steered_output[b, replacement_patches].clone()
                        
                    elif len(concept_patches) > 0:
                        steered_output[b, concept_patches] = noise[b, concept_patches]
            
            # DEBUG: Print output statistics (reduced frequency and verbosity)
            if debug_this_step and num_affected > 0:
                print(f"  Output - mean: {steered_output.mean().item():.6f}, std: {steered_output.std().item():.6f}")

        # Reshape back to spatial format
        hook_output = steered_output.reshape(
            len(output_cat),
            h,
            w,
            -1,
        ).permute(0, 3, 1, 2)
        
        self.timestep_idx += 1

        return (hook_output,)