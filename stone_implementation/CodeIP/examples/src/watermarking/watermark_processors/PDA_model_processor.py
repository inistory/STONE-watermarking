from typing import Union

import torch
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from .base_processor import WmProcessorBase
from .message_models.PDA_message_model import PDAMessageModel

HfTokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]

class PDAProcessorMessageModel(WmProcessorBase):
    def __init__(self, message_model: PDAMessageModel, tokenizer, gamma, beta,encode_ratio=10.,
                 seed=42):
        super().__init__(seed=seed)
        
      
        self.message_model: PDAMessageModel = message_model
        
        self.encode_ratio = encode_ratio
        self.start_length = 0
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.beta = beta

    def cal_watermark_scores(self, input_ids, beta, vocab_size):
        # vocab_size 크기로 생성
        wm_scores = torch.zeros((input_ids.shape[0], vocab_size), dtype=torch.float, device=input_ids.device)

        selected_tokens = self.message_model.get_selected_tokens(input_ids)
        for idx in selected_tokens:
            # 인덱스가 vocab_size 범위를 넘지 않도록 필터링
            idx = [i for i in idx if 0 <= i < vocab_size]
            if len(idx) > 0:
                wm_scores[:, idx] += beta
        return wm_scores
    
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = input_ids[:, self.start_length:]

        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        predicted_class = self.message_model.get_pda_predictions(input_texts)

        if predicted_class == -1:
            print("⚠️ Skipped PDA addition: prediction returned -1.")
            return scores  
        
        vocab_size = scores.shape[-1]
        wm_scores = self.cal_watermark_scores(input_ids, self.beta, vocab_size)

        scores = self.message_model.cal_addition_scores(
            input_ids,
            lm_predictions=predicted_class,
            scores=scores,
            gamma=self.gamma
        )
        scores += wm_scores
        
        return scores
