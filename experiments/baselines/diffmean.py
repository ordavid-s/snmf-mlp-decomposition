import torch
import torch.nn as nn
from transformer_lens import utils  # Make sure transformer_lens is installed
from tqdm import tqdm


class DiffMean:
    def __init__(self, model, layer, tokenizer, device="cpu", mode="mlp"):
        """
        Args:
            model: A HookedTransformer model.
            layer: The layer number from which to extract activations.
            tokenizer: The tokenizer for processing sentences.
            steering_magnitude: Scaling factor for intervention.
            device: Device to run the model and computations.
            mode: Either "residual" or "mlp", determining which activations to use.
        """
        super().__init__()
        self.model = model
        self.layer = layer
        self.tokenizer = tokenizer
        self.device = device
        self.mode = mode
        self.concept_vector = None  # Will hold the computed intervention direction

    def _get_hook_string(self, layer_number: int) -> str:
        """
        Returns the hook string corresponding to the chosen mode and layer.
        """
        if self.mode == "mlp":
            return f"blocks.{layer_number}.mlp.hook_post"
        if self.mode == "mlp_out":
            return f"blocks.{layer_number}.hook_mlp_out"
        elif self.mode == "resid_post":
            return f"blocks.{layer_number}.hook_resid_post"
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    @torch.no_grad()
    def fit(self, positive_sentences, negative_sentences, prefix_length=0):
        """
        Computes the DiffMean intervention direction from a list of sentences.
        
        Args:
            sentences (List[str]): List of input sentences.
            labels (List[int]): List of labels corresponding to each sentence.
            target_label (int): The label considered as positive.
            prefix_length (int): Number of initial tokens to ignore (e.g., prompt tokens).
        """
        positive_activations = []
        negative_activations = []
        hook_str = self._get_hook_string(self.layer)
        
        for positive_sentence in positive_sentences:
            # Tokenize the sentence
            inputs = self.tokenizer(positive_sentence, return_tensors="pt", padding="longest")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run the model with cache to get activations
            _, cache = self.model.run_with_cache(inputs["input_ids"])
            activations = cache[hook_str]  # Shape: (1, seq_len, hidden_dim)
            
            # Remove prefix tokens
            activations = activations[:, prefix_length:]  # (1, seq_len - prefix, hidden_dim)
            mask = inputs["attention_mask"][:, prefix_length:]  # (1, seq_len - prefix)
            
            # Remove the batch dimension and flatten valid tokens via the mask
            activations = activations.squeeze(0)       # (seq_len - prefix, hidden_dim)
            mask = mask.squeeze(0)                     # (seq_len - prefix)
            valid_activations = activations[mask.bool()] # (num_valid_tokens, hidden_dim)

            positive_activations.append(valid_activations)
        
        for negative_sentence in negative_sentences:
            # Tokenize the sentence
            inputs = self.tokenizer(negative_sentence, return_tensors="pt", padding="longest")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run the model with cache to get activations
            _, cache = self.model.run_with_cache(inputs["input_ids"])
            activations = cache[hook_str]  # Shape: (1, seq_len, hidden_dim)
            
            # Remove prefix tokens
            activations = activations[:, prefix_length:]  # (1, seq_len - prefix, hidden_dim)
            mask = inputs["attention_mask"][:, prefix_length:]  # (1, seq_len - prefix)
            
            # Remove the batch dimension and flatten valid tokens via the mask
            activations = activations.squeeze(0)       # (seq_len - prefix, hidden_dim)
            mask = mask.squeeze(0)                     # (seq_len - prefix)
            valid_activations = activations[mask.bool()] # (num_valid_tokens, hidden_dim)

            negative_activations.append(valid_activations)

        # Concatenate activations across sentences
        pos_tokens = torch.cat(positive_activations, dim=0)
        neg_tokens = torch.cat(negative_activations, dim=0)
        print(f"Got {len(positive_activations)} positive activations\tGot {len(negative_activations)} negative activations")
        # Compute mean activations
        mean_positive = pos_tokens.mean(dim=0)
        mean_negative = neg_tokens.mean(dim=0)
        
        # Compute and normalize the difference vector
        diff_vector = mean_positive - mean_negative
        diff_vector = diff_vector / (diff_vector.norm() + 1e-8)
        self.concept_vector = diff_vector  # Shape: (hidden_dim,)
        print("DiffMean direction computed.")

