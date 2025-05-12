from dataclasses import dataclass, field
from typing import List

@dataclass
class WmBaseArgs:
    temperature: float = 0.75 
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    language: str = "python"  # Will be automatically set based on dataset_type
    sample_num: int = 164  # Default value for HumanEval
    sample_seed: int = 42
    seed: int = 42
    num_beams: int = 1
    delta: float = 5.0
    gamma: float = 3.0
    beta: float = 3.0
    repeat_penalty: float = 1.2
    ngram_size: int = 10
    message: List[int] = field(default_factory=lambda: [2024, 2024, 2024, 2024, 2024])
    prompt_length: int = 300
    generated_length: int = 200
    message_code_len: int = 20 
    encode_ratio: float = 10.0
    device: str = 'cuda:0'
    save_path: str = ""
    dataset_type: str = "humaneval"  # Can be "humaneval", "mbpp", or "humanevalpack"
    top_k = 40

    def __post_init__(self):
        # Set language and sample_num based on dataset_type
        if self.dataset_type == "mbpp":
            self.sample_num = 399
            self.language = "python"
        elif self.dataset_type == "humaneval":
            self.sample_num = 164
            self.language = "python"
        elif self.dataset_type == "humanevalpack":
            self.sample_num = 164  # HumanEvalPack has same number of samples as HumanEval
            self.generated_length = 512  # Set generated_length to 512 for HumanEvalPack
            self.encode_ratio = 20.0  # Increase encode_ratio for HumanEvalPack
            if self.language not in ["java", "cpp"]:
                raise ValueError(f"HumanEvalPack only supports 'java' or 'cpp' languages, got {self.language}")
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")