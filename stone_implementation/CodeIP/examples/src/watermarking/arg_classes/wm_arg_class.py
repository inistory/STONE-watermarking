from dataclasses import dataclass, field
from typing import List

@dataclass
class WmBaseArgs:
    temperature: float = 0.75 
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    language: str = "python"
    sample_num: int = 164  # Default value for HumanEval
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
    device: str = 'cuda:0'
    save_path: str = ""
    dataset_type: str = "humaneval"
    top_k = 40

    def __post_init__(self):
        # Set sample_num based on dataset_type
        if self.dataset_type == "mbpp":
            self.sample_num = 399
        elif self.dataset_type == "humaneval":
            self.sample_num = 164
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")