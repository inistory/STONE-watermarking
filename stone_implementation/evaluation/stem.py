import math
"""
Description of calculate_STEM function parameters:
- pass_k: pass@k score
- AUROC: AUROC (Area Under the Receiver Operating Characteristic Curve)
- perplexity_H: perplexity result of human-generated code
- perplexity_wm: perplexity result of watermarked code
"""
def calculate_STEM(pass_k, AUROC, perplexity_H, perplexity_wm):
    try: 
        correctness = round((pass_k)/2,3)
        detectability = round(AUROC,3)
        naturalness = round(1-(abs(perplexity_wm-perplexity_H)) / perplexity_H ,3)
        STEM =  round((correctness + detectability + naturalness)/3,3)
        print("correctness:", correctness, "detectability:", detectability, "naturalness:", naturalness, "STEM:", STEM)
    
    except ZeroDivisionError:
        print("ZeroDivisionError: perplexity_H or F_no_wm is 0")
        return None
    except ValueError as e:
        print(f"ValueError: {e}")
        return None

datasets = {
    "mbppplus": {
        "KGW": {"pass_k": 0.000, "AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000},
        "EWD": {"pass_k": 0.000, "AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000},
        "SWEET": {"pass_k": 0.000,"AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000},
        "STONE": {"pass_k": 0.000,"AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000}
    },
    "humanevalplus": {
        "KGW": {"pass_k": 0.000, "AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000},
        "EWD": {"pass_k": 0.000, "AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000},
        "SWEET": {"pass_k": 0.000,"AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000},
        "STONE": {"pass_k": 0.000,"AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000}
    },
    "humanevalpack_cpp": {
        "KGW": {"pass_k": 0.000, "AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000},
        "EWD": {"pass_k": 0.000, "AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000},
        "SWEET": {"pass_k": 0.000,"AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000},
        "STONE": {"pass_k": 0.000,"AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000}
    },
    "humanevalpack_java": {
        "KGW": {"pass_k": 0.000, "AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000},
        "EWD": {"pass_k": 0.000, "AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000},
        "SWEET": {"pass_k": 0.000,"AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000},
        "STONE": {"pass_k": 0.000,"AUROC":0.000, "perplexity_H": 0.000, "perplexity_wm": 0.000}
    }
}


def run_experiment():
    for dataset, baselines in datasets.items():
        print(f"Dataset: {dataset}")
        for baseline, values in baselines.items():
            print("Baseline:", baseline)
            calculate_STEM(values["pass_k"], values["AUROC"], values["perplexity_H"],values["perplexity_wm"])
        print()
        
run_experiment() 