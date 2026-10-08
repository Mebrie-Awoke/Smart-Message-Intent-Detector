# Smart Message Intent Detector

This tiny project shows how a pretrained Hugging Face Transformer can classify a message into one of three application-related intents.

## What it learns

The project trains a small DistilBERT model on nine examples across these three intents:

- `application_status`
- `application_fee`
- `application_location`

The examples are intentionally short. The goal is to understand how Transformer-based text classification learns meaning, not to build a production service.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

The first run downloads the pretrained model and trains the classification head:

```bash
python intent_detector.py
```

Enter a message at the prompt. Type `quit` to stop.

To classify one message directly:

```bash
python intent_detector.py --message "Where should I send my application?"
```

The program prints the predicted intent and a confidence score.

## Project structure

- `intent_detector.py` contains the tiny dataset, training loop, prediction function, and command-line interface.
- `requirements.txt` lists the required Python packages.

## Notes

- No database, web server, RAG system, frontend, or deployment setup is included.
- The model is trained locally and uses the CPU by default.
- The model weights are not saved; this keeps the project small and easy to understand.
- Transformers and PyTorch require enough memory to load the pretrained model. A reasonable amount of free RAM is helpful.
 