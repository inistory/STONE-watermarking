from dataclasses import dataclass, field
from typing import List

@dataclass
class WmBaseArgs:
    temperature: float = field(default=0.75)
    model_name: str = field(default="Qwen/Qwen2.5-Coder-7B-Instruct")
    language: str = field(default="python")
    sample_num: int = field(default=164)  # Default for HumanEval+
    sample_seed: int = field(default=42)
    seed: int = field(default=42)
    num_beams: int = field(default=1)
    delta: float = field(default=5.0)
    gamma: float = field(default=3.0)
    repeat_penalty: float = field(default=1.2)
    ngram_size: int = field(default=10)
    message: List[int] = field(default_factory=lambda: [2024])
    prompt_length: int = field(default=300)
    generated_length: int = field(default=200)
    message_code_len: int = field(default=20)
    encode_ratio: float = field(default=10.)
    device: str = field(default='cuda:0')
    save_path: str = field(default="")
    top_k: int = field(default=40)
    dataset_type: str = field(default="humaneval")

    def __post_init__(self):
        # Set sample_num based on dataset type
        if self.dataset_type in ["humaneval", "humanevalpack"]:
            self.sample_num = 164
        elif self.dataset_type == "mbpp":
            self.sample_num = 399