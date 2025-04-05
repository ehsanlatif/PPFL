import copy
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def preprocess_function(batch, tokenizer, max_length=1024):
    """
    Preprocess the dataset batch by combining instruction, input, and output into a single sequence,
    tokenizing them, and preparing input and label tensors for training.
    """
    # Combine instruction and input into a single prompt
    inputs = [
        "Instruction: " + instruction + "\n" + "Input: " + input_text + "\n" + "Output:" + output_text
        for instruction, input_text, output_text in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    # targets = batch["output"]

    # Ensure the tokenizer has a padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize inputs
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=max_length)

    # Tokenize targets and mask padding tokens
    # labels = tokenizer(targets, padding="max_length", truncation=True, max_length=max_length).input_ids
    # labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs

def load_partitioned_data(dataset_name, client_id, tokenizer, num_clients=3, split_size=8000):
    dataset = load_dataset(dataset_name, split="train").shuffle(seed=42)
    start = client_id * split_size
    end = (client_id + 1) * split_size
    train_val_dataset = dataset.select(range(start, min(end, len(dataset))))
    train_test_split = train_val_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    val_dataset = train_test_split["test"]
       # Preprocess datasets
    train_dataset = train_dataset.map(
    lambda batch: preprocess_function(batch, tokenizer=tokenizer),
    batched=True,
    )
    val_dataset = val_dataset.map(
    lambda batch: preprocess_function(batch, tokenizer=tokenizer),
    batched=True,
    )
    # train_dataset = train_dataset.map(preprocess_function, batched=True)
    # val_dataset = val_dataset.map(preprocess_function, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # return train_loader, val_loader
    return train_dataset, val_dataset

def train_local(model, tokenizer, dataset, epochs=1, lr=5e-5, batch_size=32, max_length=1024, device="cpu"):
    """
    Fine-tune the model on the local client dataset.
    
    Args:
        model: The transformer model to fine-tune.
        tokenizer: The tokenizer for the model.
        dataset: The dataset to fine-tune on.
        epochs: Number of epochs to train.
        lr: Learning rate for the optimizer.
        batch_size: Batch size for training.
        max_length: Maximum tokenized sequence length.
        device: Device to train the model on (e.g., "cpu" or "cuda").

    Returns:
        Updated model state dict after training.
    """

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up model and optimizer
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(data_loader) * epochs
    )
    total_loss = 0
    # Training loop
    for _ in range(epochs):
        model.train()
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

    # Return the updated model state
    return copy.deepcopy(model.state_dict()),  (total_loss / len(data_loader))


def evaluate_model(model, tokenizer, dataset, batch_size=4, max_length=512, device="cpu"):

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device).eval()

    total_loss = 0.0
    for batch in data_loader:
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    return total_loss / len(data_loader), len(dataset)


def apply_lora(model):
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1)
    return get_peft_model(model, lora_config)


def get_model_and_tokenizer(model_name="Maykeye/TinyLLama-v0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model, parameters):
    state_dict = model.state_dict()
    for (key, value), param in zip(state_dict.items(), parameters):
        state_dict[key] = torch.tensor(param, dtype=state_dict[key].dtype)
    model.load_state_dict(state_dict, strict=True)


def params_to_numpy(state_dict):
    return [val.cpu().numpy() for _, val in state_dict.items()]
