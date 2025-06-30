import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'} for inference")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path).to(device)

    return model, tokenizer, device
