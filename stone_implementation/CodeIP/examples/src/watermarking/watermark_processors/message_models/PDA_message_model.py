from typing import Union
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
import torch

from ...utils.hash_fn import Hash1
from ...arg_classes.wm_arg_class import WmBaseArgs

HfTokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]

import torch.nn.functional as F
from pygments.lexers import PythonLexer, GoLexer, JavaLexer, JavascriptLexer, PhpLexer
from pygments.token import  STANDARD_TYPES


class PDAMessageModel():
    # def __init__(self, tokenizer: HfTokenizer, pda_model,
    #              delta, seed=42, lm_topk=1000, message_code_len=10,
    #              random_permutation_num=100, hash_prefix_len=1, hash_fn=Hash1):
    def __init__(self, tokenizer: HfTokenizer, pda_model,
                delta, seed=42, lm_topk=1000, message_code_len=10,
                random_permutation_num=100, hash_prefix_len=1, hash_fn=Hash1,
                vocab=None): 
        self.vocab = vocab  
        self.seq_length = 30
        self.tokenizer = tokenizer
        self.delta = delta
        self.message_code_len = message_code_len
        self.lexer = PythonLexer()
        if WmBaseArgs.language == "python":
            self.lexer = PythonLexer()
        elif WmBaseArgs.language == "go":
            self.lexer = GoLexer()
        elif WmBaseArgs.language == "java":
            self.lexer = JavaLexer()
        elif WmBaseArgs.language == "javascript":
            self.lexer = JavascriptLexer()
        elif WmBaseArgs.language == "php":
            self.lexer = PhpLexer()
        self.nonterminal2id = {nonterminal: i for i, nonterminal in enumerate(STANDARD_TYPES.values())}
        self.lex_and_tokenizer_id_list = self.construct_lex_and_tokenizer_id_list()
        self.max_vocab_size = 100000
        self.pda_model = pda_model

    def get_selected_tokens(self, input_ids):
        """
        Re-implements the token selection based on hashing mechanism used in cal_addition_scores.
        """
        selected_idx = []
        for w in range(len(self.lex_and_tokenizer_id_list)):
            idx = self.lex_and_tokenizer_id_list[w]
            if 0 <= w < len(self.lex_and_tokenizer_id_list):
                selected_idx.append(idx)
        return selected_idx

    @property
    def random_delta(self):
        return self.delta * 1.

    @property
    def device(self):
        print("not implemented")

    def set_random_state(self):
        pass
    
    def construct_lex_and_tokenizer_id_list(self):
        lex_and_tokenizer_id_list = [set() for i in range(len(self.nonterminal2id))]
        for i in range(self.tokenizer.vocab_size):
            v = self.tokenizer.decode([i])
            tokens = self.lexer.get_tokens(v)
            for token in tokens:
                token_type, value = token
                lex_and_tokenizer_id_list[self.nonterminal2id[STANDARD_TYPES[token_type]]].add(i)

        return lex_and_tokenizer_id_list
        
    def get_pda_predictions(self, strings):
        try:
            tokens = self.lexer.get_tokens(strings[0])
            token_types = [STANDARD_TYPES[token_type] for token_type, value in tokens]
            # print(f"🧪 Token types: {token_types}")
            lex_array = [self.nonterminal2id.get(t, -1) for t in token_types]
            # print(f"🧪 Lexical indices (may include -1): {lex_array}")

            if -1 in lex_array or len(lex_array) == 0:
                print("⚠️ PDA prediction skipped: invalid lexical indices.")
                return -1

            lex_array = torch.tensor([lex_array]).to(WmBaseArgs.device)
            score = self.pda_model(lex_array)
            output_probs = F.log_softmax(score, dim=1)
            predicted_class = torch.argmax(output_probs, dim=1)
            return predicted_class.item()
        except Exception as e:
            print(f"🚨 Exception during PDA prediction: {e}")
            return -1
        

    def cal_addition_scores(self, input_ids, lm_predictions, scores, gamma):
        selected_idx = []
        if isinstance(lm_predictions, int):
            lm_predictions = [lm_predictions]
        try:
            selected_idx = [self.lex_and_tokenizer_id_list[idx] for idx in lm_predictions
                            if 0 <= idx < len(self.lex_and_tokenizer_id_list)]
        except Exception as e:
            print(f"⚠️ Error while selecting PDA indices: {e}")
            selected_idx = []

        if not selected_idx:
            print("⚠️ Skipping sample due to empty selected_idx (invalid predictions?)")
            return scores

        for idx in selected_idx:
            scores[:, list(idx)] = scores[:, list(idx)] + gamma
        return scores
