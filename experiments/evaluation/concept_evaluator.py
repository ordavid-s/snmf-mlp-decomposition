import torch
import torch.nn.functional as F
from tqdm import tqdm


def explanation_score(tensor_a, tensor_b, metric='r2', scale='None'):
    """
    Computes how well tensor_a explains tensor_b using a given metric.
    
    Args:
        tensor_a (torch.Tensor): Predictor or explanatory tensor
        tensor_b (torch.Tensor): Target tensor to be explained
        metric (str): Metric to compute ('r2', 'cosine', 'corr', 'mse')
        scale (str or None): Optional scaling method: 
            - None: no scaling
            - 'standardize': zero-mean, unit variance
            - 'normalize': unit norm (L2)
            - 'minmax': scale to [0, 1]
    
    Returns:
        float: Explanation score
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("Both tensors must have the same shape.")
    
    # Flatten
    a = tensor_a.flatten().float()
    b = tensor_b.flatten().float()

    # Apply optional scaling
    def apply_scale(x):
        if scale == 'standardize':
            return (x - x.mean()) / (x.std() + 1e-8)
        elif scale == 'normalize':
            return x / (x.norm(p=2) + 1e-8)
        elif scale == 'minmax':
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        else:
            return x

    a = apply_scale(a)
    b = apply_scale(b)

    if metric == 'r2':
        # R-squared (Coefficient of Determination)
        ss_res = torch.sum((b - a) ** 2)
        ss_tot = torch.sum((b - b.mean()) ** 2)
        score = 1 - ss_res / (ss_tot + 1e-8)
        score = score.item()
        
    elif metric == 'cosine':
        # Cosine similarity
        score = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    elif metric == 'corr':
        # Pearson correlation coefficient
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        score = torch.sum(a_centered * b_centered) / (
            torch.sqrt(torch.sum(a_centered ** 2)) * torch.sqrt(torch.sum(b_centered ** 2)) + 1e-8
        )
        score = score.item()

    elif metric == 'mse':
        # Mean Squared Error
        score = F.mse_loss(a, b).item()

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return score


class ConceptEvaluator:

    def __init__(self, model, hook_template="blocks.{layer_number}.mlp.hook_post"):
        self.model = model
        self.hook_template = hook_template

    def evaluate_nmf(self, prompts, nmf, layers):
        # Ensure prompt is a list (if a single string is provided)
        if isinstance(prompts, str):
            prompts = [prompts]

        # Convert prompts to tokens (batch processing; using left padding for consistency)
        tokens = self.model.to_tokens(prompts, prepend_bos=True, padding_side="left")

        
        # Run the model and keep the batch dimension
        _, model_cache = self.model.run_with_cache(
            tokens, 
            remove_batch_dim=False, 
            stop_at_layer=max(layers) + 1, 
        )
        output = []
        for layer_number in layers:
            hook_name = self.hook_template.format(layer_number=layer_number) # f"blocks.{layer_number}.mlp.hook_post"
            activations = model_cache[hook_name][:, -1, :].to(nmf[0].H.device)            
            for i in range(nmf[layer_number].H.size(0)):
                # Extract the activation for the last token for each prompt
                metrics = ['r2', 'cosine', 'corr', 'mse']
                scores = {m: 0 for m in metrics}
                for idx in range(activations.size(0)):
                # Average the activations over the prompt (batch) dimension
                    act = activations[idx] # - random_activations
                    for metric in metrics:
                        result = explanation_score(nmf[layer_number].H[i], act, metric=metric)
                        scores[metric] += result
                # Compute metrics comparing the concept tensor with the averaged activation
                final_results = {}
                for metric in metrics:
                    final_results[metric] = scores[metric]/activations.size(0)
                output.append((final_results, layer_number, i))
        return output

    def evaluate_tensor(self, prompts, layer_number, concept_tensor, sentence_reduction=lambda x: max(x)):
        # Ensure prompt is a list (if a single string is provided)
        if isinstance(prompts, str):
            prompts = [prompts]

        # Convert prompts to tokens (batch processing; using left padding for consistency)
        tokens = self.model.to_tokens(prompts, prepend_bos=True, padding_side="left")
        
        # Create an attention mask that excludes both pad and BOS tokens
        pad_token_id = self.model.tokenizer.pad_token_id
        bos_token_id = self.model.tokenizer.bos_token_id
        attention_mask = (tokens != pad_token_id) & (tokens != bos_token_id)

        hook_name = self.hook_template.format(layer_number=layer_number) # f"blocks.{layer_number}.mlp.hook_post"
        
        # Run the model and keep the batch dimension
        _, model_cache = self.model.run_with_cache(
            tokens, 
            remove_batch_dim=False, 
            stop_at_layer=layer_number + 1, 
            names_filter=[hook_name]
        )

        activations = model_cache[hook_name].to(concept_tensor.device)
        
        metrics = ['r2', 'cosine', 'corr', 'mse']
        scores = {m: [] for m in metrics}

        # Iterate over the batch
        for idx in range(activations.size(0)):
            sample_acts = activations[idx]
            sample_mask = attention_mask[idx]
            sample_scores = {m: [] for m in metrics}
            
            # Iterate over tokens in the sequence
            for token_act, mask in zip(sample_acts, sample_mask):
                if not mask:
                    # Skip padding or BOS tokens
                    continue
                for metric in metrics:
                    result = explanation_score(concept_tensor, token_act, metric=metric)
                    sample_scores[metric].append(result)
                        
            for metric in metrics:
                scores[metric].append(sentence_reduction(sample_scores[metric]))
            
        return scores



    def isolate_best_concept(self, prompts, nmf, layer_number=-1, metric='corr', reverse=True):
        # Ensure prompts is a list (if a single string is provided)
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Determine which layers to scan
        if layer_number == -1:
            layers_to_scan = list(range(len(nmf.models)))
        else:
            layers_to_scan = [layer_number]
        
        # Convert prompts to tokens (batch-processing)
        tokens = self.model.to_tokens(prompts, prepend_bos=True, padding_side="left")
        # random_tokens = self.model.to_tokens(RANDOM_WORDS, prepend_bos=True, padding_side="left")
        _, model_cache = self.model.run_with_cache(tokens, remove_batch_dim=False)
        # _, random_cache = self.model.run_with_cache(random_tokens, remove_batch_dim=False)
        
        # Dictionary to accumulate scores for each concept (indexed by (layer, concept_index))
        concept_scores = {}
        
        # Loop over each prompt in the batch
        # for idx in tqdm(range(len(prompts))):
        for layer in layers_to_scan:
            hook_name = f"blocks.{layer}.mlp.hook_post"
            activation = model_cache[hook_name].mean(dim=0)[-1, :].squeeze().to(nmf[0].device)
            # random_activations = random_cache[hook_name].mean(dim=0)[-1, :].squeeze().to(nmf[0].device)
            # final_activation = activation - random_activations
            for i in range(nmf[0].H.size(0)):
                score1 = explanation_score(nmf[layer].H[i], activation, metric=metric)
                # score2 = explanation_score(nmf[layer].H[i], random_activations, metric=metric)
                score = score1
                key = (layer, i)
                if key not in concept_scores:
                    concept_scores[key] = []
                concept_scores[key].append(score)
        
        # Compute the average score for each concept
        averaged_scores = [(sum(scores) / len(scores), key) for key, scores in concept_scores.items()]
        
        # Sort concepts by the average score
        averaged_scores_sorted = sorted(averaged_scores, key=lambda x: x[0], reverse=reverse)
        
        return averaged_scores_sorted
