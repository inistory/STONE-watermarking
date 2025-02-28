# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================
# stone.py
# Description: Implementation of STONE algorithm
# ==============================================
import torch
from math import sqrt

from functools import partial
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization

class STONEConfig:
    """Config class for STONE algorithm, load config file and initialize parameters."""

    def __init__(self, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the STONE configuration.

            Parameters:
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """

        self.gamma = kwargs.get('gamma')
        self.delta = kwargs.get('delta')
        self.hash_key = kwargs.get('hash_key')
        self.z_threshold = kwargs.get('z_threshold')
        self.prefix_length = kwargs.get('prefix_length')
        self.language = kwargs.get('language')

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs
    

class STONEUtils:
    """Utility class for STONE algorithm, contains helper functions."""

    def __init__(self, config: STONEConfig, *args, **kwargs):
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.skipping_rule = kwargs.get('skipping_rule')
        self.watermark_on_pl = kwargs.get('watermark_on_pl')
        self.language = kwargs.get('language')

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the random number generator with the last prefix_length tokens of the input_ids."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * prev_token)
        return

    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        self._seed_rng(input_ids)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size] 
        return greenlist_ids
    
    def _compute_z_score(self, observed_count: int, T: int) -> float: 
        """Compute z-score for the observed count of green tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T 
        denom = sqrt(T * expected_count * (1 - expected_count))  
        z = numer / denom
        return z

    def score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int], list[int]]:
        
        if self.language == 'python':
            keywords = [
                'True', 'False', 'None', 'and', 'as', 'assert', 'async', 'await',
                'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
                'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
                'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
                'while', 'with', 'yield'
            ]
            operators = [
                '+', '-', '*', '/', '%', '**', '//', '=', '==', '!=', '>', '<',
                '>=', '<=', '+=', '-=', '*=', '/=', '%=', '//=', '**=', '&', '|',
                '<<', '>>','^', '~'
            ]
            delimiters = [
                '(', ')', '[', ']', '{', '}', ',', ':', '.', ';', '@', '->', '...'
            ]
            whitespaces = [' ', '\t', '\n']
            types = [
                'int', 'float', 'complex', 'str', 'bytes', 'bool', 'list', 'tuple',
                'set', 'dict', 'NoneType'
            ]

        elif self.language == 'cpp':
            keywords = [
                'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor',
                'break', 'case', 'catch', 'class', 'compl', 'concept', 'const', 'consteval', 'constexpr',
                'constinit', 'const_cast', 'continue', 'co_await', 'co_return',
                'co_yield', 'decltype', 'default', 'delete', 'do', 'dynamic_cast',
                'else', 'enum', 'explicit', 'export', 'extern', 'false', 'for',
                'friend', 'goto', 'if', 'inline', 'mutable', 'namespace', 'new',
                'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq',
                'private', 'protected', 'public', 'register', 'reinterpret_cast',
                'requires', 'return', 'sizeof', 'static', 'static_assert',
                'static_cast', 'struct', 'switch', 'template',
                'this', 'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid',
                'typename', 'union', 'using', 'virtual', 'volatile', 'while',
                'xor', 'xor_eq', 'override'
            ]
            operators = [
                '+', '-', '*', '/', '%', '++', '--', '=', '==', '!=', '>', '<',
                '>=', '<=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '+=',
                '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=','.*', '->*'
            ]
            delimiters = ['(', ')', '[', ']', '{', '}', ',', ':', '.', ';', '->', '::','...']
            whitespaces = [' ', '\t', '\n']
            types = [
                'int', 'float', 'double', 'bool', 'char', 'short', 'long',
                'void', 'unsigned', 'signed', 'size_t', 'ptrdiff_t', 
                'wchar_t', 'char8_t', 'char16_t','char32_t'
            ]
            
        elif self.language == 'java':
            keywords = [
                'abstract', 'assert', 'break', 'case', 'catch', 'class', 'const',
                'continue', 'default', 'do', 'else', 'enum', 'extends', 'final',
                'finally', 'for', 'goto', 'if', 'implements', 'import', 'instanceof',
                'interface', 'native', 'new', 'null', 'package', 'private',
                'protected', 'public', 'return', 'static', 'strictfp', 'super',
                'switch', 'synchronized', 'this', 'throw', 'throws', 'transient',
                'try', 'void', 'volatile', 'while', 'true', 'false'
            ]
            operators = [
                '+', '-', '*', '/', '%', '++', '--', '=', '==', '!=', '>', '<',
                '>=', '<=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '>>>',
                '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '>>>='
            ]
            delimiters = ['(', ')', '[', ']', '{', '}', ',', ':', '.', ';', '@', '->', '::', '...']
            whitespaces = [' ', '\t', '\n']
            types = [
                'byte', 'short', 'int', 'long', 'float', 'double', 'boolean', 'char',
                'String', 'Object'
            ]        

        syntax_tokens = []
        if self.skipping_rule == 'all_pl':
            syntax_tokens = keywords + operators + delimiters + whitespaces + types
        elif self.skipping_rule == 'keywords':
            syntax_tokens = keywords
        elif self.skipping_rule == 'operators':
            syntax_tokens = operators
        elif self.skipping_rule == 'delimiters':
            syntax_tokens = delimiters
        elif self.skipping_rule == 'whitespaces':
            syntax_tokens = whitespaces
        elif self.skipping_rule == 'types':
            syntax_tokens = types

        decoded_texts = []
        for i in range(len(input_ids)):
            decoded_texts.append(self.config.generation_tokenizer.decode(input_ids[i].item(), skip_special_tokens=True))

        target_num = 0
        for d in decoded_texts:
            if self.watermark_on_pl == "True":
                if d in syntax_tokens or d.strip() in syntax_tokens:
                    target_num += 1
                else:
                    pass
            else: 
                if d in syntax_tokens or d.strip() in syntax_tokens:
                    pass
                else:
                    target_num += 1

        num_tokens_scored = (len(input_ids) - self.config.prefix_length - target_num) 
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                )
            )

        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        weights = [-1 for _ in range(self.config.prefix_length)]

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)

            if self.watermark_on_pl == "True":
                if decoded_texts[idx] in syntax_tokens or decoded_texts[idx].strip() in syntax_tokens:
                    weights.append(1)
                else:
                    weights.append(0)
            else: 
                if decoded_texts[idx] in syntax_tokens or decoded_texts[idx].strip() in syntax_tokens:
                    weights.append(0)
                else:
                    weights.append(1)    

        # calculate number of green tokens where weight is 1
        green_token_count = sum([1 for i in range(len(green_token_flags)) if green_token_flags[i] == 1 and weights[i] == 1])
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        
        return z_score, green_token_flags, weights


class STONELogitsProcessor(LogitsProcessor):
    """Logits processor for STONE algorithm, contains the logic to bias the logits."""

    def __init__(self, config: STONEConfig, utils: STONEUtils, *args, **kwargs) -> None:
        """
            Initialize the STONE logits processor.

            Parameters:
                config (STONEConfig): Configuration for the STONE algorithm.
                utils (STONEUtils): Utility class for the STONE algorithm.
        """
        self.config = config
        self.utils = utils
        self.skipping_rule = kwargs.get('skipping_rule')
        self.watermark_on_pl = kwargs.get('watermark_on_pl')
        self.language = kwargs.get('language')

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)
        raw_probs = torch.softmax(scores, dim=-1)  

        # token will be generated in the next time step
        next_token = self.config.generation_tokenizer.decode(torch.argmax(raw_probs), skip_special_tokens=True)
        keywords = []
        operators = []
        delimiters = []
        whitespaces = []
        types = []
        if self.language == 'python':
            keywords = [
                'True', 'False', 'None', 'and', 'as', 'assert', 'async', 'await',
                'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
                'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
                'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
                'while', 'with', 'yield'
            ]
            operators = [
                '+', '-', '*', '/', '%', '**', '//', '=', '==', '!=', '>', '<',
                '>=', '<=', '+=', '-=', '*=', '/=', '%=', '//=', '**=', '&', '|',
                '<<', '>>','^', '~'
            ]
            delimiters = [
                '(', ')', '[', ']', '{', '}', ',', ':', '.', ';', '@', '->', '...'
            ]
            whitespaces = [' ', '\t', '\n']
            types = [
                'int', 'float', 'complex', 'str', 'bytes', 'bool', 'list', 'tuple',
                'set', 'dict', 'NoneType'
            ]

        elif self.language == 'cpp':
            keywords = [
                'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor',
                'break', 'case', 'catch', 'class', 'compl', 'concept', 'const', 'consteval', 'constexpr',
                'constinit', 'const_cast', 'continue', 'co_await', 'co_return',
                'co_yield', 'decltype', 'default', 'delete', 'do', 'dynamic_cast',
                'else', 'enum', 'explicit', 'export', 'extern', 'false', 'for',
                'friend', 'goto', 'if', 'inline', 'mutable', 'namespace', 'new',
                'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq',
                'private', 'protected', 'public', 'register', 'reinterpret_cast',
                'requires', 'return', 'sizeof', 'static', 'static_assert',
                'static_cast', 'struct', 'switch', 'template',
                'this', 'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid',
                'typename', 'union', 'using', 'virtual', 'volatile', 'while',
                'xor', 'xor_eq', 'override'
            ]
            operators = [
                '+', '-', '*', '/', '%', '++', '--', '=', '==', '!=', '>', '<',
                '>=', '<=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '+=',
                '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=','.*', '->*'
            ]
            delimiters = ['(', ')', '[', ']', '{', '}', ',', ':', '.', ';', '->', '::','...']
            whitespaces = [' ', '\t', '\n']
            types = [
                'int', 'float', 'double', 'bool', 'char', 'short', 'long',
                'void', 'unsigned', 'signed', 'size_t', 'ptrdiff_t', 
                'wchar_t', 'char8_t', 'char16_t','char32_t'
            ]
            
        elif self.language == 'java':
            keywords = [
                'abstract', 'assert', 'break', 'case', 'catch', 'class', 'const',
                'continue', 'default', 'do', 'else', 'enum', 'extends', 'final',
                'finally', 'for', 'goto', 'if', 'implements', 'import', 'instanceof',
                'interface', 'native', 'new', 'null', 'package', 'private',
                'protected', 'public', 'return', 'static', 'strictfp', 'super',
                'switch', 'synchronized', 'this', 'throw', 'throws', 'transient',
                'try', 'void', 'volatile', 'while', 'true', 'false'
            ]
            operators = [
                '+', '-', '*', '/', '%', '++', '--', '=', '==', '!=', '>', '<',
                '>=', '<=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '>>>',
                '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '>>>='
            ]
            delimiters = ['(', ')', '[', ']', '{', '}', ',', ':', '.', ';', '@', '->', '::', '...']
            whitespaces = [' ', '\t', '\n']
            types = [
                'byte', 'short', 'int', 'long', 'float', 'double', 'boolean', 'char',
                'String', 'Object'
            ]        

           

        syntax_tokens = []
        if self.skipping_rule == 'all_pl':
            syntax_tokens = keywords + operators + delimiters + whitespaces + types
        elif self.skipping_rule == 'keywords':
            syntax_tokens = keywords
        elif self.skipping_rule == 'operators':
            syntax_tokens = operators
        elif self.skipping_rule == 'delimiters':
            syntax_tokens = delimiters
        elif self.skipping_rule == 'whitespaces':
            syntax_tokens = whitespaces
        elif self.skipping_rule == 'types':
            syntax_tokens = types

        if self.watermark_on_pl == "True":
            if next_token in syntax_tokens or next_token.strip() in syntax_tokens:
                pl_mask = torch.tensor([[True]], device=scores.device)
                # print("Watermarked token: ", next_token)
            else:
                pl_mask = torch.tensor([[False]], device=scores.device)
                # print("Non-watermarked token: ", next_token)
        else:
            if next_token in syntax_tokens or next_token.strip() in syntax_tokens:
                pl_mask = torch.tensor([[False]], device=scores.device)
                # print("Non-watermarked token: ", next_token)
            else:
                pl_mask = torch.tensor([[True]], device=scores.device)
                # print("Watermarked token: ", next_token)            

        green_tokens_mask = green_tokens_mask * pl_mask

        # bias the greenlist tokens
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.config.delta)
        return scores

class STONE(BaseWatermark):
    """Top-level class for STONE algorithm."""

    def __init__(self, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the STONE algorithm.

            Parameters:
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = STONEConfig(transformers_config, skipping_rule = kwargs.get('skipping_rule'), watermark_on_pl = kwargs.get('watermark_on_pl'), gamma = kwargs.get('gamma'), delta = kwargs.get('delta'), hash_key = kwargs.get('hash_key'), prefix_length = kwargs.get('prefix_length'), z_threshold = kwargs.get('z_threshold'), language = kwargs.get('language'))
        self.utils = STONEUtils(self.config, skipping_rule = kwargs.get('skipping_rule'), watermark_on_pl = kwargs.get('watermark_on_pl'), gamma = kwargs.get('gamma'), delta = kwargs.get('delta'), hash_key = kwargs.get('hash_key'), prefix_length = kwargs.get('prefix_length'), z_threshold = kwargs.get('z_threshold'), language = kwargs.get('language'))
        self.logits_processor = STONELogitsProcessor(self.config, self.utils, skipping_rule = kwargs.get('skipping_rule'), watermark_on_pl = kwargs.get('watermark_on_pl'), gamma = kwargs.get('gamma'), delta = kwargs.get('delta'), hash_key = kwargs.get('hash_key'), prefix_length = kwargs.get('prefix_length'), z_threshold = kwargs.get('z_threshold'), language = kwargs.get('language'))

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """Generate watermarked text."""
        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""
       
        # encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # compute z_score
        z_score, _, _ = self.utils.score_sequence(encoded_text)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)
