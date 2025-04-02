import json

results_json_path = "./codeip_results.json"
output_jsonl_path = "../../../results/1samples/codeip_results.jsonl"

with open(results_json_path, "r") as results_file:
    results_data = json.load(results_file)
    prefix_and_output_text = results_data.get("prefix_and_output_text", [])

jsonl_data = []
for i, completion in enumerate(prefix_and_output_text):
    entry = {
        "task_id": f"HumanEval/{i}",
        "completion": completion.strip()
    }
    jsonl_data.append(entry)

with open(output_jsonl_path, "w") as output_file:
    for entry in jsonl_data:
        output_file.write(json.dumps(entry) + "\n")

print(f"new jsonl file is generated: {output_jsonl_path}")