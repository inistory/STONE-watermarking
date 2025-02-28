import math

def calculate_CWEM(perplexity_wm, perplexity_H, pass_k, AUROC):
    try: 
        correctness = round((pass_k[0]+pass_k[1])/2,3)
        detectability = round(AUROC,3)
        naturalness = round(1-(abs(perplexity_wm-perplexity_H)) / perplexity_H ,3)
        CWEM =  round((correctness + detectability + naturalness)/3,3)
        print("correctness:", correctness, "detectability:", detectability, "naturalness:", naturalness, "CWEM:", CWEM)
    
    except ZeroDivisionError:
        print("ZeroDivisionError: perplexity_H or F_no_wm is 0")
        return None
    except ValueError as e:
        print(f"ValueError: {e}")
        return None

datasets = {
    "mbppplus": {
        "KGW": {"AUROC":0.831, "pass_k": [0.414, 0.585], "perplexity_H": 3.504, "perplexity_wm": 3.525},
        "EWD": {"AUROC":0.965, "pass_k": [0.414, 0.585], "perplexity_H": 3.504, "perplexity_wm": 3.525},
        "SWEET": {"AUROC":0.867, "pass_k": [0.419, 0.585], "perplexity_H": 3.504, "perplexity_wm": 3.533},
        "STONE": {"AUROC":0.982, "pass_k": [0.441, 0.701], "perplexity_H": 3.504, "perplexity_wm": 3.539}
    },
    "humanevalplus": {
        "KGW": {"AUROC":0.523, "pass_k": [0.445, 0.701], "perplexity_H": 3.276, "perplexity_wm": 3.322},
        "EWD": {"AUROC":0.730, "pass_k": [0.445, 0.701], "perplexity_H": 3.276, "perplexity_wm": 3.322},
        "SWEET": {"AUROC":0.710, "pass_k": [0.470, 0.677], "perplexity_H": 3.276, "perplexity_wm": 3.347},
        "STONE": {"AUROC":0.777, "pass_k": [0.472, 0.701], "perplexity_H": 3.276, "perplexity_wm": 3.349}
    },
    "humanevalpack_cpp": {
        "KGW": {"AUROC":0.621, "pass_k": [0.451, 0.701], "perplexity_H": 2.621, "perplexity_wm": 2.639},
        "EWD": {"AUROC":0.681, "pass_k": [0.451, 0.701], "perplexity_H": 2.621, "perplexity_wm": 2.639},
        "SWEET": {"AUROC":0.641, "pass_k": [0.473, 0.695], "perplexity_H": 2.621, "perplexity_wm": 2.676},
        "STONE": {"AUROC":0.729, "pass_k": [0.500, 0.744], "perplexity_H": 2.621, "perplexity_wm": 2.647}
    },
    "humanevalpack_java": {
        "KGW": {"AUROC":0.546, "pass_k": [0.230, 0.543], "perplexity_H": 2.426, "perplexity_wm": 2.443},
        "EWD": {"AUROC":0.646, "pass_k": [0.230, 0.543], "perplexity_H": 2.426, "perplexity_wm": 2.443},
        "SWEET": {"AUROC":0.580, "pass_k": [0.254, 0.573], "perplexity_H": 2.426, "perplexity_wm": 2.666},
        "STONE": {"AUROC":0.721, "pass_k": [0.274, 0.616], "perplexity_H": 2.426, "perplexity_wm": 2.478}
    }
}


def run_experiment():
    for dataset, baselines in datasets.items():
        print(f"Dataset: {dataset}")
        if dataset == "humanevalpack_java":
            for baseline, values in baselines.items():
                print("Baseline:", baseline)
                pass_k = values["pass_k"]
                AUROC = values["AUROC"]
                calculate_CWEM(values["perplexity_wm"], values["perplexity_H"], pass_k,AUROC)
            print()


run_experiment() 