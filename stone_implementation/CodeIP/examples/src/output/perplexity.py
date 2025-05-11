from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

model_name = "bigcode/starcoder2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

jsonl_file_path = "../../../../results/1samples/result_python_humaneval-sanitized.jsonl"

with open(jsonl_file_path, "r") as file:
    lines = file.readlines()

perplexities = []
for line in lines:
    data = json.loads(line)
    if "completion" in data:
        text = data["completion"]
        perplexity = calculate_perplexity(text)
        perplexities.append(perplexity)
        print(f"Task ID: {data['task_id']}, Perplexity: {perplexity}")
    if "solution" in data:
        text = data["solution"]
        perplexity = calculate_perplexity(text)
        perplexities.append(perplexity)
        print(f"Task ID: {data['task_id']}, Perplexity: {perplexity}")

if perplexities:
    average_perplexity = sum(perplexities) / len(perplexities)
    print(f"Average Perplexity: {average_perplexity}")
else:
    print("No perplexities calculated.")