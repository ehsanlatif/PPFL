import argparse
import flwr as fl
import utils


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, model_name, dataset_name, num_clients, split_size, device):
        self.client_id = client_id
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.split_size = split_size
        self.device = device

        # Load tokenizer
        self.local_model, self.tokenizer = utils.get_model_and_tokenizer(self.model_name)

        # Load local partition of data
        self.local_train_data, self.local_val_data  = utils.load_partitioned_data(
            dataset_name=self.dataset_name,
            client_id=self.client_id,
            tokenizer = self.tokenizer, 
            num_clients=self.num_clients,
            split_size=self.split_size,
        )
        # self.local_data =  self.local_data.map(
        # lambda batch: utils.preprocess_function(batch, tokenizer=self.tokenizer),
        # batched=True,
        # remove_columns=self.local_data.column_names,
        # )
        # self.local_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        print("Client Initialized Successfully")

    def get_parameters(self, config):
        """
        Return the local model parameters.
        """
        # local_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.local_model = utils.apply_lora(self.local_model)
        return utils.get_params(self.local_model)

    def fit(self, parameters, config):
        """
        Train the model locally and return updated parameters.
        """
        # local_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.local_model = utils.apply_lora(self.local_model)
        utils.set_params(self.local_model, parameters)

        epochs = config.get("epochs", 1)
        lr = config.get("lr", 5e-5)
        batch_size = config.get("batch_size", 32)
        max_length = config.get("max_length", 1024)

        updated_state, train_loss = utils.train_local(
            model=self.local_model,
            tokenizer=self.tokenizer,
            dataset=self.local_train_data,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            max_length=max_length,
            device=self.device,
        )
        print("Client Trained with loss:", float(train_loss))
        return utils.params_to_numpy(updated_state), len(self.local_train_data), {"loss": float(train_loss)}

    def evaluate(self, parameters, config):
        """
        Evaluate the model on local data and return loss and number of examples.
        """
        # local_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.local_model = utils.apply_lora(self.local_model)
        utils.set_params(self.local_model, parameters)

        self.local_model.to(self.device)
        self.local_model.eval()

        loss, num_examples = utils.evaluate_model(
            model=self.local_model,
            tokenizer=self.tokenizer,
            dataset=self.local_val_data,
            batch_size=32,
            max_length=1024,
            device=self.device,
        )
        print("Client Validated with loss:", float(loss))
        return float(loss), num_examples, {"loss": float(loss)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="Maykeye/TinyLLama-v0")
    parser.add_argument("--dataset_name", type=str, default="garage-bAInd/Open-Platypus")
    parser.add_argument("--split_size", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    client = FlowerClient(
        client_id=args.client_id,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        num_clients=args.num_clients,
        split_size=args.split_size,
        device=args.device,
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
