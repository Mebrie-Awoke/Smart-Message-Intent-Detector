# Smart Message Intent Detector
# This project is intentionally small and educational.

import argparse
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


INTENTS = [
    "application_status",
    "application_fee",
    "application_location",
]

# Each sentence is paired with its intent. The examples are deliberately small,
# so the model is learning from a tiny training set rather than a production corpus.
TRAINING_DATA = [
    ("What is the status of my application?", "application_status"),
    ("Can you tell me whether my application is approved?", "application_status"),
    ("My application is still under review.", "application_status"),
    ("How much is the application fee?", "application_fee"),
    ("What is the fee for submitting an application?", "application_fee"),
    ("Do I need to pay an application fee?", "application_fee"),
    ("Where should I send my application?", "application_location"),
    ("Which office accepts applications?", "application_location"),
    ("Can you provide the application location?", "application_location"),
]


class IntentDataset(Dataset):
    """Simple PyTorch dataset that stores text and its label."""

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

def build_dataset():
    """Split the examples into text and label values."""
    texts = [message for message, _ in TRAINING_DATA]
    labels = [INTENTS.index(intent) for _, intent in TRAINING_DATA]
    return IntentDataset(texts, labels)

def train_model(
    model_name="distilbert-base-uncased",
    epochs=3,
    batch_size=4,
    learning_rate=2e-5,  
): 
    """Fine-tune a pretrained Transformer for the three intents."""
    random.seed(42)
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(INTENTS),
    )

    dataset = IntentDataset(
        [message for message, _ in TRAINING_DATA],
        [INTENTS.index(intent) for _, intent in TRAINING_DATA],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # A small optimizer lets the pretrained model learn the new intent labels.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for _ in range(epochs):
        for messages, batch_labels in loader:
            encoded_messages = tokenizer(
                list(messages),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded_messages["input_ids"]
            attention_mask = encoded_messages["attention_mask"]

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
  
    model.eval()
    return model, tokenizer

def predict_intent(message, model, tokenizer):
    """Return the highest-probability intent for one message."""
    encoded = tokenizer(
        [message],
        padding=True, 
        truncation=True,
        return_tensors="pt",
    )  
    
    with torch.no_grad(): 
        predictions = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

    probabilities = torch.softmax(predictions.logits, dim=-1)[0] 
    predicted_index = probabilities.argmax().item()
    return INTENTS[predicted_index], probabilities[predicted_index].item()


def interactive_loop(model, tokenizer):  
    """Ask the user for messages and print each prediction."""
    print("\nSmart Message Intent Detector")
    print("Type 'quit' to exit.")

    while True:
        message = input("\nMessage: ").strip()
        if not message:
            print("Please enter a message.")
            continue
        if message.lower() == "quit":
            break

        intent, confidence = predict_intent(message, model, tokenizer)
        print(f"Predicted intent: {intent}")
        print(f"Confidence: {confidence:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a tiny Transformer model to detect intents."
    )
    parser.add_argument(
        "--message",
        help="Classify one message and exit instead of starting the interactive loop.",
    )
    args = parser.parse_args()

    # Download the pretrained model once if it is not already cached.
    model, tokenizer = train_model()    
     
    if args.message:  
        intent, confidence = predict_intent(args.message, model, tokenizer)
        print(f"Message: {args.message}")
        print(f"Predicted intent: {intent}")   
        print(f"Confidence: {confidence:.3f}")
    else:
        interactive_loop(model, tokenizer)

if __name__ == "__main__":
    main()
