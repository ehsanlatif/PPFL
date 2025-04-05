import flwr as fl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from peft import PeftModel, LoraConfig, set_peft_model_state_dict

def save_global_model(parameters, model_name="Maykeye/TinyLLama-v0", save_path="global_model"):
    """
    Save the global model parameters (including LoRA adapter) to a local directory.
    """
    # Load the base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize LoRA configuration and wrap the base model
    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"
    )
    model = PeftModel(model, lora_config)

    # Convert Parameters object to PyTorch state_dict
    serialized_tensors = parameters.tensors
    lora_state_dict = model.state_dict()  # LoRA's state_dict

    for (key, value), tensor in zip(lora_state_dict.items(), serialized_tensors):
        tensor_array = np.frombuffer(tensor, dtype=np.float32)
        
        # Check for size mismatch and handle it
        if tensor_array.size > value.numel():
            print(f"Trimming extra elements for {key}: {tensor_array.size - value.numel()} elements")
            tensor_array = tensor_array[:value.numel()]  # Trim extra elements
        elif tensor_array.size < value.numel():
            raise ValueError(f"Size mismatch for {key}: Expected {value.numel()}, Got {tensor_array.size}")

        # Reshape to match the original tensor shape
        reshaped_tensor = torch.tensor(tensor_array.reshape(value.shape), dtype=value.dtype)
        lora_state_dict[key] = reshaped_tensor

    # Set the LoRA state_dict to the model
    set_peft_model_state_dict(model, lora_state_dict)

    # Save the base model and LoRA adapter
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Global model with LoRA adapter saved to {save_path}")




def main():
    global_model_parameters = None
    total_rounds = 3

    class CustomFedAvg(fl.server.strategy.FedAvg):
        def aggregate_evaluate(self, rnd, results, failures):
            # Optionally, process evaluation results here
            return super().aggregate_evaluate(rnd, results, failures)

        def aggregate_fit(self, rnd, results, failures):
            # Aggregate model parameters as usual
            aggregated_parameters = super().aggregate_fit(rnd, results, failures)
            if rnd == total_rounds:  # Save the global model at the final round
                nonlocal global_model_parameters
                global_model_parameters = aggregated_parameters[0]
            return aggregated_parameters

    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )
    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=total_rounds),
        strategy=strategy,
    )

    # Save the final global model after training rounds
    # final_parameters = strategy.get_parameters()
    if global_model_parameters:
        save_global_model(global_model_parameters)

if __name__ == "__main__":
    main()
