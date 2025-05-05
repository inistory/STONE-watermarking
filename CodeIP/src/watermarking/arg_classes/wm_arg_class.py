from dataclasses import dataclass, field
from typing import List

@dataclass
class WmBaseArgs:
    temperature: float = 0.75 
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    language: str = "python"
    sample_num: int = 164
    sample_seed: int = 42
    seed: int = 42
    num_beams: int = 1
    delta: float = 5.0
    gamma: float = 3.0
    repeat_penalty: float = 1.2
    ngram_size: int = 10
    message: List[int] = field(default_factory=lambda: [2024])
    prompt_length: int = 300
    generated_length: int = 200
    message_code_len: int = 20 
    encode_ratio: float = 10.
    device: str = 'cuda:1'
    save_path: str = ""
    top_k = 40