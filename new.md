
```python
#!/usr/bin/env python3

import json
from pathlib import Path

import torch
from torch import nn


# ============================================================
# Configuration
# ============================================================

TEXT_FILE = Path("training.txt")

SEQ_LENGTH = 100
BATCH_SIZE = 64
EPOCHS = 20

EMBED_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2

LEARNING_RATE = 0.002

TEMPERATURE = 0.8
GENERATE_LENGTH = 500


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print(f"Device: {DEVICE}")


# ============================================================
# Model
# ============================================================

class TextModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        num_layers
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x, hidden=None):

        x = self.embedding(x)

        output, hidden = self.lstm(
            x,
            hidden
        )

        output = self.fc(output)

        return output, hidden


# ============================================================
# Load training text
# ============================================================

if not TEXT_FILE.exists():
    raise FileNotFoundError(
        "training.txt was not found."
    )

text = TEXT_FILE.read_text(
    encoding="utf-8"
)

if len(text) < SEQ_LENGTH + 1:
    raise ValueError(
        "training.txt is too short for the selected SEQ_LENGTH."
    )

print(f"Training characters: {len(text):,}")


# ============================================================
# Vocabulary
# ============================================================

chars = sorted(set(text))

char_to_id = {
    char: i
    for i, char in enumerate(chars)
}

id_to_char = {
    i: char
    for char, i in char_to_id.items()
}

encoded = torch.tensor(
    [char_to_id[c] for c in text],
    dtype=torch.long
)

vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")


# ============================================================
# Dataset
# ============================================================

class TextDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data,
        seq_length
    ):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):

        x = self.data[
            index:index + self.seq_length
        ]

        y = self.data[
            index + 1:index + self.seq_length + 1
        ]

        return x, y


dataset = TextDataset(
    encoded,
    SEQ_LENGTH
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


# ============================================================
# Create model
# ============================================================

model = TextModel(
    vocab_size=vocab_size,
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
).to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

print()
print("Starting training...")
print()

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        output, _ = model(x)

        loss = criterion(
            output.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(loader)
    )

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS:03d} "
        f"Loss: {average_loss:.4f}"
    )


# ============================================================
# Save trained model
# ============================================================

checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "embed_size": EMBED_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "seq_length": SEQ_LENGTH
}

torch.save(
    checkpoint,
    "model.pt"
)


# ============================================================
# Save vocabulary
# ============================================================

with open(
    "vocab.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "char_to_id": char_to_id,
            "id_to_char": {
                str(k): v
                for k, v in id_to_char.items()
            }
        },
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("Training complete.")
print("Saved model: model.pt")
print("Saved vocabulary: vocab.json")


# ============================================================
# Load saved model
# ============================================================

def load_model():

    with open(
        "vocab.json",
        "r",
        encoding="utf-8"
    ) as f:

        vocabulary = json.load(f)

    checkpoint = torch.load(
        "model.pt",
        map_location=DEVICE,
        weights_only=True
    )

    loaded_model = TextModel(
        vocab_size=checkpoint["vocab_size"],
        embed_size=checkpoint["embed_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"]
    ).to(DEVICE)

    loaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    loaded_model.eval()

    loaded_char_to_id = {
        key: int(value)
        for key, value in vocabulary["char_to_id"].items()
    }

    loaded_id_to_char = {
        int(key): value
        for key, value in vocabulary["id_to_char"].items()
    }

    return (
        loaded_model,
        loaded_char_to_id,
        loaded_id_to_char
    )


# ============================================================
# Text generation
# ============================================================

@torch.no_grad()
def generate_text(
    model,
    char_to_id,
    id_to_char,
    prompt,
    length=500,
    temperature=0.8
):

    model.eval()

    prompt = "".join(
        character
        for character in prompt
        if character in char_to_id
    )

    if not prompt:
        prompt = next(iter(char_to_id))

    input_ids = torch.tensor(
        [
            char_to_id[character]
            for character in prompt
        ],
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    hidden = None

    output_text = prompt

    # Process the entire prompt.
    output, hidden = model(
        input_ids,
        hidden
    )

    current = input_ids[:, -1:]

    # Generate one character at a time.
    for _ in range(length):

        output, hidden = model(
            current,
            hidden
        )

        logits = output[:, -1, :]

        logits = logits / temperature

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        next_id = torch.multinomial(
            probabilities,
            num_samples=1
        )

        character = id_to_char[
            next_id.item()
        ]

        output_text += character

        current = next_id

    return output_text


# ============================================================
# Load model from disk
# ============================================================

print()
print("Loading saved model...")

loaded_model, loaded_char_to_id, loaded_id_to_char = load_model()

print("Model loaded successfully.")


# ============================================================
# Generate text
# ============================================================

print()
print("=" * 60)
print("GENERATED TEXT")
print("=" * 60)
print()

generated = generate_text(
    loaded_model,
    loaded_char_to_id,
    loaded_id_to_char,
    prompt="",
    length=GENERATE_LENGTH,
    temperature=TEMPERATURE
)

print(generated)

print()
print("=" * 60)
```

Run it with:

```bash
python train_and_generate.py
```

It will produce:

```text
training.txt
train_and_generate.py
model.pt
vocab.json
```

The important workflow is:

```text
training.txt
     ↓
PyTorch LSTM
     ↓
training
     ↓
model.pt + vocab.json
     ↓
load model
     ↓
generate new text
```

You can change these at the top:

```python
EPOCHS = 20
HIDDEN_SIZE = 256
TEMPERATURE = 0.8
GENERATE_LENGTH = 500
```

For **more conservative text**, lower `TEMPERATURE` toward `0.5`. For **more varied text**, increase it toward `1.0–1.2`.
