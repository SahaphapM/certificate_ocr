import os
from model_loader import load_model
from predictor import predict_entities
from utils import print_results, save_to_json
from data import sample_texts


def main():
    # เปลี่ยนเป็น path ของโมเดล
    model_path = "trained_model"
    model, tokenizer, device = load_model(model_path)

    all_results = []
    for i, text in enumerate(sample_texts, 1):
        print(f"\n\n{'#' * 30} EXAMPLE {i} {'#' * 30}")
        entities = predict_entities(text, model, tokenizer, device)
        print_results(text, entities)
        all_results.append({
            "text": text,
            "entities": entities
        })

    save_to_json(all_results, "predictions.json")
    print("\nPredictions saved to predictions.json")


if __name__ == "__main__":
    main()
