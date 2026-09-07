import numpy as np
import torch

from SAE.unlearning_utils import get_percentile_threshold


class SAEReconstructHook:
    def __init__(
        self,
        sae,
        guidance_scale=7.5,
    ):
        self.sae = sae
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def __call__(self, module, input, output):
        if self.guidance_scale > 1.0:
            output1, output2 = output[0].chunk(2)
            # reshape to SAE input shape
            output1 = output1.permute(0, 2, 3, 1).reshape(
                len(output1), output1.shape[-1] * output1.shape[-2], -1
            )
            output2 = output2.permute(0, 2, 3, 1).reshape(
                len(output2), output2.shape[-1] * output2.shape[-2], -1
            )
            output_cat = torch.cat([output1, output2], dim=0)
        else:
            out = output[0]
            output_cat = out.permute(0, 2, 3, 1).reshape(
                len(out), out.shape[-1] * out.shape[-2], -1
            )

        sae_input, _, _ = self.sae.preprocess_input(output_cat)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        sae_out = (latents @ self.sae.W_dec) + self.sae.b_dec

        if self.guidance_scale > 1.0:
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
        else:
            h, w = out.shape[2], out.shape[3]
            hook_output = sae_out.reshape(len(out), h, w, -1).permute(0, 3, 1, 2)

        return (hook_output,)


class SAEMaskedUnlearningHook:
    def __init__(
        self,
        concept_to_unlearn,
        percentile,
        multiplier,
        feature_importance_fn,
        concept_latents_dict,
        sae,
        steps=100,
        preserve_error=True,
        seed=None,
        start_timestep=0,  # NEW: timestep from which to start applying unlearning
        use_beta=False,  # NEW: use norm-based scaling instead of mean-based
        guidance_scale=7.5,
        residual_sae=False,  # Set True for SAEs trained on (output - input)
    ):
        self.concept_to_unlearn = concept_to_unlearn
        self.percentile = percentile
        self.multiplier = multiplier
        self.feature_importance_fn = feature_importance_fn
        self.concept_latents_dict = concept_latents_dict
        self.timestep_idx = 0
        self.sae = sae
        self.steps = steps
        self.preserve_error = preserve_error
        self.start_timestep = start_timestep  # NEW: store start_timestep
        self.use_beta = use_beta  # NEW: store use_beta flag
        self.guidance_scale = guidance_scale
        self.residual_sae = residual_sae
        # precompute the most important features for this theme on every timestep
        self.scaling_factors = []
        self.top_feature_idxs = []
        self.avg_feature_acts = []
        self.all_concept_avg_acts = []
        self.seed = seed
        print('SAE seed: ', str(self.seed))
        print(f'SAE unlearning will start at timestep: {self.start_timestep}')
        print(f'SAE use_beta: {self.use_beta}')
        # then compute the percentile threshold for each timestep based on distribution of all scores
        for timestep in range(steps):
            timestep_feature_idxs = []
            timestep_scaling_factors = []
            timestep_all_concept_avg_acts = []
            for concept in self.concept_to_unlearn:
                feature_scores = self.feature_importance_fn(
                    self.concept_latents_dict, concept, timestep, seed=self.seed
                )
                feature_scores = feature_scores.float()
                percentile_threshold = get_percentile_threshold(
                    feature_scores, self.percentile
                )
                top_feature_idxs = torch.where(feature_scores > percentile_threshold)[0]
                timestep_feature_idxs.append(top_feature_idxs)
                concept_acts = self.concept_latents_dict[concept][
                    :, timestep, top_feature_idxs
                ]
                avg_acts = concept_acts.mean(0)
                scaling_factors = avg_acts * self.multiplier
                timestep_scaling_factors.append(scaling_factors)

                # precompute average activations of features on other styles
                all_concept_avg_acts = torch.zeros((len(top_feature_idxs)))
                for concept in self.concept_latents_dict:
                    all_concept_avg_acts += self.concept_latents_dict[concept][
                        :, timestep, top_feature_idxs
                    ].mean(dim=0)
                all_concept_avg_acts /= len(self.concept_latents_dict)
                timestep_all_concept_avg_acts.append(all_concept_avg_acts)
            self.top_feature_idxs.append(torch.cat(timestep_feature_idxs))
            self.scaling_factors.append(torch.cat(timestep_scaling_factors))
            self.all_concept_avg_acts.append(torch.cat(timestep_all_concept_avg_acts))

    @torch.no_grad()
    def __call__(self, module, input, output):
        inp_cat = None
        if self.guidance_scale > 1.0:
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
            if self.residual_sae:
                inp1, inp2 = input[0].chunk(2)
                inp1 = inp1.permute(0, 2, 3, 1).reshape(len(inp1), inp1.shape[-1] * inp1.shape[-2], -1)
                inp2 = inp2.permute(0, 2, 3, 1).reshape(len(inp2), inp2.shape[-1] * inp2.shape[-2], -1)
                inp_cat = torch.cat([inp1, inp2], dim=0)
                output_cat = output_cat - inp_cat
        else:
            out = output[0]
            h, w = out.shape[2], out.shape[3]
            output_cat = out.permute(0, 2, 3, 1).reshape(len(out), h * w, -1)
            if self.residual_sae:
                inp_cat = input[0].permute(0, 2, 3, 1).reshape(len(out), h * w, -1)
                output_cat = output_cat - inp_cat

        # encode activations
        sae_input, _, _ = self.sae.preprocess_input(output_cat)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        recon_acts_original = (latents @ self.sae.W_dec) + self.sae.b_dec
        latents = latents.reshape(len(output_cat), -1, self.sae.num_latents)
        recon_acts_original = recon_acts_original.reshape(
            len(output_cat), -1, self.sae.d_in
        )

        if self.preserve_error:
            error_original = (recon_acts_original - output_cat).float()

        # NEW: Only apply unlearning if we've reached start_timestep
        if self.timestep_idx >= self.start_timestep:
            # mask selecting on which patches ablate which features
            mask = latents[
                :, :, self.top_feature_idxs[self.timestep_idx]
            ] > self.all_concept_avg_acts[self.timestep_idx].to(pre_acts.device)

            if self.use_beta:
                # NEW: Compute beta-based scaling dynamically
                # beta_i = ||latents||_2 / ||D_{:,i}||_2 for each feature i
                
                # Compute norm of full latent vector at each position: [batch, seq_len, 1]
                latents_norm = torch.norm(latents, dim=-1, keepdim=True)
                
                # Get decoder weights
                decoder = self.sae.W_dec.to(pre_acts.device)  # [num_latents, d_in]
                
                # Get decoder columns for the top features at this timestep
                top_features = self.top_feature_idxs[self.timestep_idx]
                decoder_cols = decoder[top_features]  # [num_top_features, d_in]
                
                # Compute norm of each decoder column: [num_top_features]
                decoder_norms = torch.norm(decoder_cols, dim=-1)
                
                # Compute beta for each feature: [batch, seq_len, num_top_features]
                # beta_i = ||latents||_2 / ||D_{:,i}||_2
                beta = latents_norm / (decoder_norms.view(1, 1, -1) + 1e-8)
                
                if self.timestep_idx % 10 == 0:
                    print(f"[Timestep {self.timestep_idx}] Beta stats - "
                          f"mean: {beta.mean().item():.4f}, "
                          f"min: {beta.min().item():.4f}, "
                          f"max: {beta.max().item():.4f}, "
                          f"std: {beta.std().item():.4f}")

                # Apply multiplier to beta
                scaling = beta * self.multiplier
            else:
                # Original mean-based scaling
                scaling = self.scaling_factors[self.timestep_idx].to(pre_acts.device)
                scaling = scaling.view(1, 1, -1).expand(mask.size(0), mask.size(1), -1)

            # Apply mask and scaling
            selected_latents = latents[:, :, self.top_feature_idxs[self.timestep_idx]]
            selected_latents = torch.where(
                mask, selected_latents * scaling, selected_latents
            )
            latents[:, :, self.top_feature_idxs[self.timestep_idx]] = selected_latents

        recon_acts_ablated = (latents @ self.sae.W_dec) + self.sae.b_dec
        if self.preserve_error:
            recon_acts_ablated = (recon_acts_ablated + error_original).to(output_cat.dtype)
        else:
            recon_acts_ablated = recon_acts_ablated.to(output_cat.dtype)

        if self.residual_sae:
            hook_output = (inp_cat + recon_acts_ablated).reshape(
                len(output_cat), h, w, -1
            ).permute(0, 3, 1, 2)
        else:
            hook_output = recon_acts_ablated.reshape(
                len(output_cat), h, w, -1
            ).permute(0, 3, 1, 2)
        self.timestep_idx += 1

        return (hook_output,)