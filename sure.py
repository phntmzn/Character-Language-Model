#!/usr/bin/env python3

import json
from pathlib import Path

import torch
from torch import nn


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

EPOCHS = 20
SEQUENCE_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.002

EMBEDDING_DIM = 128
HIDDEN_DIM = 256


# ============================================================
# Multiline training text
# ============================================================

TEXT = """

"""


# ============================================================
# Character Language Model
# ============================================================

class CharacterLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        x, _ = self.rnn(x)

        x = self.output(x)

        return x


# ============================================================
# Dataset
# ============================================================

def build_dataset(
    text,
    sequence_length=SEQUENCE_LENGTH
):

    if len(text) <= sequence_length + 1:
        raise ValueError(
            "Training text is too short. "
            f"Provide more than {sequence_length + 1} characters."
        )

    # Build character vocabulary

    chars = sorted(set(text))

    char_to_id = {
        char: i
        for i, char in enumerate(chars)
    }

    id_to_char = {
        i: char
        for char, i in char_to_id.items()
    }

    # Encode text

    encoded = torch.tensor(
        [
            char_to_id[character]
            for character in text
        ],
        dtype=torch.long
    )

    inputs = []
    targets = []

    # Create training sequences

    for i in range(
        0,
        len(encoded) - sequence_length,
        sequence_length
    ):

        input_sequence = encoded[
            i:i + sequence_length
        ]

        target_sequence = encoded[
            i + 1:i + sequence_length + 1
        ]

        if len(target_sequence) != sequence_length:
            continue

        inputs.append(input_sequence)
        targets.append(target_sequence)

    if not inputs:
        raise ValueError(
            "No training sequences were created."
        )

    inputs = torch.stack(inputs)

    targets = torch.stack(targets)

    return (
        inputs,
        targets,
        char_to_id,
        id_to_char
    )


# ============================================================
# Training
# ============================================================

def train_model(
    text,
    epochs=EPOCHS,
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE
):

    print()
    print("=" * 60)
    print("PHNTMZN PyTorch Character Language Model")
    print("=" * 60)

    print()
    print(f"Device: {DEVICE}")

    if DEVICE.type == "mps":
        print("MPS / Apple Silicon GPU enabled")
    else:
        print("Using CPU")

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        inputs,
        targets,
        char_to_id,
        id_to_char
    ) = build_dataset(
        text,
        sequence_length
    )

    vocab_size = len(char_to_id)

    print(f"Characters:       {len(text):,}")
    print(f"Vocabulary:       {vocab_size:,}")
    print(f"Sequences:        {len(inputs):,}")
    print(f"Sequence length:  {sequence_length}")
    print(f"Batch size:       {batch_size}")
    print(f"Epochs:           {epochs}")
    print(f"Learning rate:    {learning_rate}")

    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CharacterLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    print("Model:")
    print(model)

    print()

    # --------------------------------------------------------
    # Optimizer and loss
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    loss_function = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        permutation = torch.randperm(
            len(inputs)
        )

        batch_count = 0

        for start in range(
            0,
            len(inputs),
            batch_size
        ):

            indices = permutation[
                start:start + batch_size
            ]

            x = inputs[
                indices
            ].to(DEVICE)

            y = targets[
                indices
            ].to(DEVICE)

            # Clear gradients

            optimizer.zero_grad()

            # Forward pass

            predictions = model(x)

            # Cross entropy expects:
            #
            # predictions:
            # [batch, sequence, vocab]
            #
            # targets:
            # [batch, sequence]

            predictions = predictions.reshape(
                -1,
                vocab_size
            )

            y = y.reshape(-1)

            # Loss

            loss = loss_function(
                predictions,
                y
            )

            # Backpropagation

            loss.backward()

            # Gradient clipping

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # Update model

            optimizer.step()

            total_loss += loss.item()

            batch_count += 1

        average_loss = (
            total_loss /
            max(1, batch_count)
        )

        print(
            f"Epoch "
            f"{epoch + 1:03d}/{epochs:03d} "
            f"loss={average_loss:.6f}"
        )

    # ========================================================
    # Save model
    # ========================================================

    model_path = Path("model.pt")
    vocab_path = Path("vocab.json")
    config_path = Path("config.json")

    # --------------------------------------------------------
    # PyTorch model
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "sequence_length": sequence_length,
        },
        model_path
    )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    with open(
        vocab_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            char_to_id,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = {
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "sequence_length": sequence_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(DEVICE)
    }

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=2
        )

    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 60)
    print("Training complete")
    print("=" * 60)

    print()
    print(f"Model saved:        {model_path.resolve()}")
    print(f"Vocabulary saved:   {vocab_path.resolve()}")
    print(f"Configuration saved:{config_path.resolve()}")

    print()

    return (
        model,
        char_to_id,
        id_to_char
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    train_model(
        text=TEXT,
        epochs=EPOCHS,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
