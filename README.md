# Smart Message Intent Detector 

Smart Message Intent Detector is a very small Python project for learning how Hugging Face Transformers classify text.

It uses a pretrained DistilBERT model and fine-tunes it on a tiny dataset containing three application-related intents:

- `application_status`
- `application_fee`
- `application_location`

This project is for learning only. It does not provide a production service, database, web interface, RAG system, or deployment setup.

## What the project does

1. Loads a small pretrained Transformer model.
2. Trains the model using nine short example messages.
3. Predicts the most likely intent for a message you enter.
4. Prints the predicted intent and confidence score.

The code is intentionally small and includes comments explaining the main steps.

## Requirements

- Python 3.9 or newer
- PyTorch
- Hugging Face Transformers

## Install

From the project directory, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

The required packages are listed in [requirements.txt](requirements.txt).

## Run the project

Start the interactive detector:

```bash
python intent_detector.py
```

You will see a prompt like this:

```text
Message:
```

Enter a message, for example:

```text
What is the status of my application?
```

The program prints the prediction:

```text
Predicted intent: application_status
Confidence: 0.999
```

Type `quit` to stop.

You can also classify one message without starting the interactive loop:

```bash
python intent_detector.py --message "How much is the application fee?"
```

## Project files

- [intent_detector.py](intent_detector.py) contains the dataset, model training, prediction, and command-line interface.
- [requirements.txt](requirements.txt) contains the Python packages needed to run the project.
- [README.md](README.md) contains the setup and usage instructions.

## How the code works

- `TRAINING_DATA` stores short messages and their correct intent labels.
- `IntentDataset` prepares the data for a PyTorch training loop.
- `AutoTokenizer` converts messages into tokens that the Transformer can process.
- `AutoModelForSequenceClassification` adds a classifier for the three intents.
- `train_model` fine-tunes the pretrained model for a few training epochs.
- `predict_intent` calculates probabilities and returns the highest-scoring intent.

## Learning notes

- The model learns from the words and context in the examples.
- A sentence about the application status is classified as `application_status`, even when it uses different wording.
- The confidence score is a rough indication of how certain the model is about its prediction.
- The examples are intentionally small, so the model is easy to inspect and understand.
- The model is trained locally and uses the CPU by default.
- Model weights are not saved because this project is only for learning.

## Important

The first run downloads the pretrained DistilBERT model from Hugging Face. A network connection is required for the first run unless the model is already cached on your computer.

The project uses a tiny training set, so its predictions should not be treated as reliable for real-world application handling.
 