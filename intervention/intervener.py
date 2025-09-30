import torch
from transformer_lens import HookedTransformer, utils
from typing import List

class Intervener:
    def __init__(self, model: HookedTransformer, intervention_type='mlp_act', replace=False):
        """
        intervention_type can be customized to switch hooking locations if desired.
        E.g. 'resid_post' -> blocks.{L}.hook_resid_post
             'mlp_act'    -> blocks.{L}.mlp.hook_post
        """
        self._model = model
        self.intervention_type = intervention_type
        self.replace = replace

    def get_intervention_location(self, layer_to_ablate: int) -> str:
        """
        Return the name of the hook point in the model where we want to intervene.
        Adapt as needed for your specific hooking location.
        """
        if self.intervention_type == 'mlp_act':
            # This is the standard hook name for MLP output in transformer_lens.
            return f"blocks.{layer_to_ablate}.mlp.hook_post"
        elif self.intervention_type == 'resid_post':
            # Alternatively, you could intervene on the residual stream.
            return f"blocks.{layer_to_ablate}.hook_resid_post"
        elif self.intervention_type == "mlp_out":
            return f"blocks.{layer_to_ablate}.hook_mlp_out"
        elif self.intervention_type == "mlp_in":
            return f"blocks.{layer_to_ablate}.hook_mlp_in"
        else:
            raise ValueError(f"Unsupported intervention_type: {self.intervention_type}")
    
    def gaussian_steer_hook(
        self,
        mu: torch.Tensor,               # [d_model]
        Sigma_inv: torch.Tensor,       # [d_model, d_model]
        alpha: float
    ):
        """
        Returns a hook function that steers activations toward a Gaussian blob 
        by ascending the log-likelihood gradient under a multivariate normal.

        x' = x - alpha * Sigma_inv @ (x - mu)
        """

        def hook(value: torch.Tensor, hook) -> torch.Tensor:
            # value: [batch, seq, d_model]
            delta = value - mu.view(1, 1, -1)                         # [batch, seq, d_model]
            grad_logp = -torch.einsum('bij,jk->bik', delta, Sigma_inv)  # [batch, seq, d_model]
            return value + alpha * grad_logp

        return hook
    
    def gaussian_steer_hook_interp(
        self,
        mu: torch.Tensor,               # [d_model]
        Sigma_inv: torch.Tensor,        # [d_model, d_model]
        alpha: float
    ):
        """
        Returns a hook that:
        1) rescales each activation x so that (x-μ)^T Σ^{-1} (x-μ) = 1
        2) interpolates:  x' = (1 - alpha) * x  +  alpha * x_rescaled
        """

        def hook(value: torch.Tensor, hook) -> torch.Tensor:
            # value: [batch, seq, d_model]
            B, T, D = value.shape

            # broadcast μ to [1,1,D]
            mu_b = mu.view(1, 1, D)

            # deviation from mean
            delta = value - mu_b                   # [B, T, D]

            # Mahalanobis squared: (delta @ Σ^{-1}) · delta, summed over D
            tmp = torch.matmul(delta, Sigma_inv)   # [B, T, D]
            m2  = (tmp * delta).sum(dim=-1, keepdim=True)  # [B, T, 1]

            # distance and avoid zero‐div
            dist = torch.sqrt(torch.clamp(m2, min=1e-12))  # [B, T, 1]

            # rescale so Mahalanobis‐norm = 1
            x_rescaled = mu_b + delta / dist      # [B, T, D]

            # convex interpolation
            return (1 - alpha) * value + alpha * x_rescaled

        return hook



    def get_mlp_post_hook(
        self, 
        direction: torch.Tensor, 
        alpha: float, 
        eps: float = 1e-8
    ):
        """
        Returns a hook function that intervenes on the MLP output.
        1) Normalizes 'direction'
        2) Sums alpha * scaled_direction into the existing activation.
        """
        # Normalize direction once upfront, so it’s purely directional.
        direction = direction / (torch.norm(direction) + eps)
        
        def mlp_post_hook(value: torch.Tensor, hook) -> torch.Tensor:
            # Operate on all tokens in the sequence.
            if self.replace:
                return torch.ones_like(value) * alpha * direction.view(1, 1, -1)
            value_to_change = value[:, :, :].clone()
            direction_reshaped = direction.view(1, 1, -1)
            # Add the intervention vector.
            new_value = value_to_change + alpha * direction_reshaped
            # Replace the intervened part in the full tensor.
            value[:, :, :] = new_value
            return value

        return mlp_post_hook

    def intervene(
        self, 
        prompt: str, 
        intervention_vectors: List[torch.Tensor],  # each of shape [d_model]
        layers: List[int], 
        alpha: float,
    ) -> torch.Tensor:
        """
        Single forward pass: apply the intervention at the specified layers with given parameters.
        Returns the final logits after applying the hooks.
        """
        tokens = self._model.to_tokens(prompt)
        
        # Build our hook information: (hook_name, hook_function)
        intervene_hooks = [
            (self.get_intervention_location(layer), self.get_mlp_post_hook(intervention_vector, alpha))
            for layer, intervention_vector in zip(layers, intervention_vectors)
        ]

        with torch.no_grad():
            ablated_logits = self._model.run_with_hooks(
                tokens,
                fwd_hooks=intervene_hooks,
            )
        return ablated_logits
    
    def gaussian_intervene(
        self, 
        prompt: str, 
        layers: List[int], 
        alpha: float,
        mu: torch.Tensor,
        Sigma_inv: torch.Tensor
    ) -> torch.Tensor:
        """
        Single forward pass: apply the intervention at the specified layers with given parameters.
        Returns the final logits after applying the hooks.
        """
        tokens = self._model.to_tokens(prompt)
        
        # Build our hook information: (hook_name, hook_function)
        intervene_hooks = [
            (self.get_intervention_location(layer), self.gaussian_steer_hook_interp(mu, Sigma_inv, alpha))
            for layer in layers
        ]

        with torch.no_grad():
            ablated_logits = self._model.run_with_hooks(
                tokens,
                fwd_hooks=intervene_hooks,
            )
        return ablated_logits
    
    @torch.inference_mode()
    def generate_with_manipulation_sampling(
        self, 
        prompt: str, 
        intervention_vectors: List[torch.Tensor],  
        layers: List[int], 
        alpha: float,
        max_new_tokens: int = 50,
        top_k: int = None,
        top_p: float = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        m: int = 1,
        use_past_kv_cache: bool = True
    ) -> List[str]:
        """
        Autoregressive generation with intervention on the activations.
        Generates m sentences in parallel, using KV caching to speed up generation.
        """
        # Convert prompt to tokens and repeat for m sequences.
        tokens = self._model.to_tokens(prompt)    # shape: [1, seq_len]
        tokens = tokens.repeat(m, 1)               # shape: [m, seq_len]
        
        # Import and initialize KV cache if enabled.
        past_kv_cache = None
        if use_past_kv_cache:
            from transformer_lens import HookedTransformerKeyValueCache
            past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                self._model.cfg, self._model.cfg.device, m
            )
        
        # Build our hook information: (hook_name, hook_function)
        intervene_hooks = [
            (self.get_intervention_location(layer), 
             self.get_mlp_post_hook(intervention_vector, alpha))
            for layer, intervention_vector in zip(layers, intervention_vectors)
        ]
        
        for i in range(max_new_tokens):
            with torch.no_grad():
                if use_past_kv_cache:
                    # For the first token, pass the full context.
                    if i == 0:
                        logits = self._model.run_with_hooks(
                            tokens, 
                            fwd_hooks=intervene_hooks,
                            past_kv_cache=past_kv_cache
                        )
                    else:
                        # For subsequent tokens, only pass the most recent token.
                        logits = self._model.run_with_hooks(
                            tokens[:, -1:], 
                            fwd_hooks=intervene_hooks,
                            past_kv_cache=past_kv_cache
                        )
                else:
                    logits = self._model.run_with_hooks(
                        tokens, 
                        fwd_hooks=intervene_hooks
                    )
            # Grab logits for the last token in each sequence; shape: [m, d_vocab]
            final_logits = logits[:, -1, :]
            # Sample one token per sequence using your sample_logits function.
            sampled_token = utils.sample_logits(
                final_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                freq_penalty=freq_penalty,
                tokens=tokens
            )
            # Reshape sampled_token to [m, 1] and append to each sequence.
            sampled_token = sampled_token.unsqueeze(1)
            tokens = torch.cat([tokens, sampled_token], dim=1)
        
        # Convert each sequence of tokens back to a string.
        generated_sentences = [self._model.to_string(tokens[i]) for i in range(m)]
        return generated_sentences
    
    def find_alpha_for_kl_targets(
        self,
        prompt: str,
        intervention_vectors: List[torch.Tensor],
        layers: List[int],
        target_kls: List[float] = [0.25, 0.5, 3.0],
        tol: float = 0.01,
        max_iter: int = 20
    ) -> dict:
        """
        For the given prompt, performs a binary search over alpha for each of the target KL divergences.
        
        The method:
         1. Computes the baseline logits (without intervention).
         2. For each target KL value, it performs a binary search over alpha (starting from 0)
            until the KL divergence (computed between the distribution from the baseline and
            the intervened logits for the last token) is within 'tol' of the target.
            
        Returns a dictionary mapping each target KL to the corresponding alpha value.
        """
        # Convert prompt to tokens and get baseline logits (no intervention).
        tokens = self._model.to_tokens(prompt)
        with torch.no_grad():
            baseline_logits = self._model.run_with_hooks(tokens)
        baseline_last = baseline_logits[:, -1, :]
        baseline_probs = torch.softmax(baseline_last, dim=-1)
        eps = 1e-8  # for numerical stability

        def compute_kl(alpha: float) -> float:
            """
            Computes the KL divergence between the baseline and intervened distributions 
            for the final token's logits.
            """
            intervened_logits = self.intervene(prompt, intervention_vectors, layers, alpha)
            intervened_last = intervened_logits[:, -1, :]
            intervened_probs = torch.softmax(intervened_last, dim=-1)
            # Calculate KL divergence: sum_i P(i) * (log(P(i)) - log(Q(i)))
            kl = torch.sum(baseline_probs * (torch.log(baseline_probs + eps) - torch.log(intervened_probs + eps)))
            return kl.item()

        results = {}
        # For each target KL divergence, find an alpha that yields approximately that KL.
        for target in target_kls:
            low = 1
            high = 2000

            # Perform binary search between low and high.
            mid = high  # initialize mid for clarity
            for _ in range(max_iter):
                mid = (low + high) / 2.0
                kl_mid = compute_kl(mid)
                if abs(kl_mid - target) < tol:
                    results[target] = mid
                if kl_mid < target:
                    low = mid
                else:
                    high = mid
            
        return results
    
    @torch.inference_mode()
    def generate_with_gaussian_manipulation_sampling(
        self, 
        prompt: str, 
        layers: List[int], 
        mu: torch.tensor,
        Sigma_inv: torch.tensor,
        alpha: float,
        max_new_tokens: int = 50,
        top_k: int = None,
        top_p: float = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        m: int = 1,
        use_past_kv_cache: bool = True
    ) -> List[str]:
        """
        Autoregressive generation with intervention on the activations.
        Generates m sentences in parallel, using KV caching to speed up generation.
        """
        # Convert prompt to tokens and repeat for m sequences.
        tokens = self._model.to_tokens(prompt)    # shape: [1, seq_len]
        tokens = tokens.repeat(m, 1)               # shape: [m, seq_len]
        
        # Import and initialize KV cache if enabled.
        past_kv_cache = None
        if use_past_kv_cache:
            from transformer_lens import HookedTransformerKeyValueCache
            past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                self._model.cfg, self._model.cfg.device, m
            )
        
        # Build our hook information: (hook_name, hook_function)
        intervene_hooks = [
            (self.get_intervention_location(layer), 
             self.gaussian_steer_hook_interp(mu, Sigma_inv, alpha))
            for layer in layers
        ]
        
        for i in range(max_new_tokens):
            with torch.no_grad():
                if use_past_kv_cache:
                    # For the first token, pass the full context.
                    if i == 0:
                        logits = self._model.run_with_hooks(
                            tokens, 
                            fwd_hooks=intervene_hooks,
                            past_kv_cache=past_kv_cache
                        )
                    else:
                        # For subsequent tokens, only pass the most recent token.
                        logits = self._model.run_with_hooks(
                            tokens[:, -1:], 
                            fwd_hooks=intervene_hooks,
                            past_kv_cache=past_kv_cache
                        )
                else:
                    logits = self._model.run_with_hooks(
                        tokens, 
                        fwd_hooks=intervene_hooks
                    )
            # Grab logits for the last token in each sequence; shape: [m, d_vocab]
            final_logits = logits[:, -1, :]
            # Sample one token per sequence using your sample_logits function.
            sampled_token = utils.sample_logits(
                final_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                freq_penalty=freq_penalty,
                tokens=tokens
            )
            # Reshape sampled_token to [m, 1] and append to each sequence.
            sampled_token = sampled_token.unsqueeze(1)
            tokens = torch.cat([tokens, sampled_token], dim=1)
        
        # Convert each sequence of tokens back to a string.
        generated_sentences = [self._model.to_string(tokens[i]) for i in range(m)]
        return generated_sentences
    
    def find_alpha_for_kl_targets_gaussian(
        self,
        prompt: str,
        mu: torch.Tensor,
        Sigma_inv: torch.Tensor,
        layers: List[int],
        target_kls: List[float] = [0.25, 0.5, 3.0],
        tol: float = 0.01,
        max_iter: int = 20
    ) -> dict:
        """
        For the given prompt, performs a binary search over alpha for each of the target KL divergences.
        
        The method:
         1. Computes the baseline logits (without intervention).
         2. For each target KL value, it performs a binary search over alpha (starting from 0)
            until the KL divergence (computed between the distribution from the baseline and
            the intervened logits for the last token) is within 'tol' of the target.
            
        Returns a dictionary mapping each target KL to the corresponding alpha value.
        """
        # Convert prompt to tokens and get baseline logits (no intervention).
        tokens = self._model.to_tokens(prompt)
        with torch.no_grad():
            baseline_logits = self._model.run_with_hooks(tokens)
        baseline_last = baseline_logits[:, -1, :]
        baseline_probs = torch.softmax(baseline_last, dim=-1)
        eps = 1e-8  # for numerical stability

        def compute_kl(alpha: float) -> float:
            """
            Computes the KL divergence between the baseline and intervened distributions 
            for the final token's logits.
            """
            intervened_logits = self.gaussian_intervene(prompt, layers, alpha, mu, Sigma_inv)
            intervened_last = intervened_logits[:, -1, :]
            intervened_probs = torch.softmax(intervened_last, dim=-1)
            # Calculate KL divergence: sum_i P(i) * (log(P(i)) - log(Q(i)))
            kl = torch.sum(baseline_probs * (torch.log(baseline_probs + eps) - torch.log(intervened_probs + eps)))
            return kl.item()

        results = {}
        # For each target KL divergence, find an alpha that yields approximately that KL.
        for target in target_kls:
            low = 1
            high = 2000

            # Perform binary search between low and high.
            mid = high  # initialize mid for clarity
            for _ in range(max_iter):
                mid = (low + high) / 2.0
                kl_mid = compute_kl(mid)
                if abs(kl_mid - target) < tol:
                    results[target] = mid
                if kl_mid < target:
                    low = mid
                else:
                    high = mid
            
        return results