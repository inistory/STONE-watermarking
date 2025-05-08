from typing import Union

import torch
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from .base_processor import WmProcessorBase
from .message_models.PDA_message_model import PDAMessageModel

HfTokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]

class PDAProcessorMessageModel(WmProcessorBase):
    def __init__(self, message_model: PDAMessageModel, tokenizer, gamma, encode_ratio=10.,
                 seed=42):
        super().__init__(seed=seed)
        
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
            
        self.message_model: PDAMessageModel = message_model
        
        self.encode_ratio = encode_ratio
        self.start_length = 0
        self.tokenizer = tokenizer
        self.gamma = gamma


    def set_random_state(self):
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        try:
            # Validate input tensors
            if input_ids is None or scores is None:
                return scores
                
            if input_ids.numel() == 0 or scores.numel() == 0:
                return scores

            # Handle NaN/Inf values silently
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)

            # Ensure input_ids has correct shape
            if input_ids.dim() != 2:
                return scores

            batch_size = input_ids.size(0)
            vocab_size = scores.size(-1)  # Get vocabulary size from scores
            input_ids = input_ids[:, self.start_length:]

            # Move to CPU for tokenization to avoid CUDA issues
            input_ids_cpu = input_ids.cpu()
            
            # Handle single text input
            if input_ids_cpu.dim() == 2 and input_ids_cpu.size(0) == 1:
                input_texts = self.tokenizer.decode(input_ids_cpu[0], skip_special_tokens=True)
            else:
                input_texts = self.tokenizer.batch_decode(input_ids_cpu, skip_special_tokens=True)

            try:
                # Ensure input_texts is a list
                if isinstance(input_texts, str):
                    input_texts = [input_texts]
                    
                predicted_class = self.message_model.get_pda_predictions(input_texts)
                
                # Handle invalid prediction
                if predicted_class is None or predicted_class == -1:
                    return scores

                # Ensure predicted_class has correct shape and type
                if isinstance(predicted_class, torch.Tensor):
                    if predicted_class.numel() == 0:
                        return scores
                        
                    # Ensure predicted_class has correct dimensions
                    if predicted_class.dim() == 1:
                        predicted_class = predicted_class.unsqueeze(0)  # Add batch dimension
                    
                    # Ensure predicted_class has correct batch size
                    if predicted_class.size(0) != batch_size:
                        if predicted_class.size(0) == 1:
                            predicted_class = predicted_class.expand(batch_size, -1)
                        else:
                            predicted_class = predicted_class[:batch_size]
                            
                    # Ensure predicted_class has correct vocabulary size
                    if predicted_class.size(-1) != vocab_size:
                        predicted_class = predicted_class[..., :vocab_size]
                        
                    predicted_class = predicted_class.to(scores.device)
                else:
                    # Convert scalar to tensor with correct batch size and vocabulary size
                    predicted_class = torch.full((batch_size, vocab_size), predicted_class, 
                                              device=scores.device, dtype=scores.dtype)

                try:
                    # Ensure all tensors are on the same device
                    input_ids = input_ids.to(scores.device)
                    
                    # Validate tensor shapes before processing
                    if input_ids.size(0) != predicted_class.size(0):
                        return scores
                        
                    # Ensure scores are finite and in a reasonable range
                    scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Apply gamma with safety checks
                    scores = self.message_model.cal_addition_scores(
                        input_ids,
                        lm_predictions=predicted_class,
                        scores=scores,
                        gamma=self.gamma
                    )
                    
                    # Ensure scores remain finite and in a reasonable range
                    scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Scale scores to prevent extreme values
                    scores = scores / (torch.abs(scores).max() + 1e-8)
                    
                except RuntimeError:
                    return scores
                    
            except Exception:
                return scores

        except Exception:
            pass

        return scores
