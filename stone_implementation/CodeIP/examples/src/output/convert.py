import json
import os

dataset_type = "humanevalpack"#mbpp, humanevalpack, humaneval
language = "java" #cpp, java 

results_json_path = f"./result_{language}_{dataset_type}.json"
output_jsonl_path = f"../../../../results/1samples/result_{language}_{dataset_type}.jsonl"
humanevalpack_output_path = f"../../../../results/1samples/humanevalpack_{language}_result.json"

# 디렉토리가 없으면 생성
os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
os.makedirs(os.path.dirname(humanevalpack_output_path), exist_ok=True)

if dataset_type == "mbpp":
    dataset = "MBPP"
elif dataset_type == "humaneval":
    dataset = "HumanEval"
elif dataset_type == "humanevalpack":
    dataset = "HumanEvalPack"

with open(results_json_path, "r") as results_file:
    results_data = json.load(results_file)
    prefix_and_output_text = results_data.get("prefix_and_output_text", [])
    task_ids = results_data.get("task_id", [])



# HumanEvalPack을 위한 추가 처리
if dataset_type == "humanevalpack":
    humanevalpack_data = [[text.strip()] for text in prefix_and_output_text]
    with open(humanevalpack_output_path, "w") as f:
        json.dump(humanevalpack_data, f, indent=4)
    print(f"HumanEvalPack result file is generated: {humanevalpack_output_path}")
else:
    jsonl_data = []
    for i, (completion, task_id) in enumerate(zip(prefix_and_output_text, task_ids)):
        if dataset == "HumanEval":
            entry = {
                "task_id": task_id,
                "completion": completion.strip()
            }
        else:
            entry = {
                "task_id": task_id,
                "solution": completion.strip()
            }
        jsonl_data.append(entry)

    with open(output_jsonl_path, "w") as output_file:
        for entry in jsonl_data:
            output_file.write(json.dumps(entry) + "\n")
    print(f"new jsonl file is generated: {output_jsonl_path}")