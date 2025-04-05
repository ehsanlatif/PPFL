import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from datasets import load_dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess_function(batch,tokenizer):
    inputs = ["Instruction: " + instruction + "\n" + "Input: " + input_text + "\n" + "Output:" for instruction, input_text in zip(batch["instruction"], batch["input"])]
    targets = batch["output"]
    tokenizer.pad_token = tokenizer.eos_token
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=256).input_ids
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
    model_inputs["labels"] = labels
    return model_inputs

def test_random_samples(model, tokenizer, dataset, num_samples=10):
    model.eval()

    for i in random.sample(range(len(dataset)), num_samples):
        tokenizer.pad_token = tokenizer.eos_token

        instruction = dataset[i]["instruction"]
        input_text = dataset[i]["input"]
        expected_output = dataset[i]["output"]

        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256)

        generated_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"=== Sample {i+1} ===")
        print(f"Instruction: {instruction}")
        print(f"Input: {input_text}")
        print(f"Expected Output: {expected_output}")
        print(f"Generated Output: {generated_output}")
        print("===================\n")

model_name = "fine_tuned_model_with_lora_2" #Change model name based on your saved model name.

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Example usage for testing
# input_text = "Instruction: Explain fine-tuning with LoRA.\nInput: None\nOutput:"
# inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")
# outputs = model.generate(**inputs)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

test_random_samples(model, tokenizer, load_dataset("garage-bAInd/Open-Platypus")["train"].select(range(10)))

