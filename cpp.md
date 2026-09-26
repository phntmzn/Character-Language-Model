Absolutely. Here is a complete **PyTorch C++ code language model** with explicit **saving and loading**. It uses one `.pt` file containing the model weights, vocabulary, and architecture settings—so you don't need a separate vocabulary file.

### Project

```text
cpp_model/
├── train.py
├── generate.py
└── data/
    ├── main.cpp
    ├── example.cpp
    └── example.hpp
```

## `train.py`

```python
from pathlib import Path
import re

import torch
from torch import nn


DATA_DIR = Path("data")
MODEL_FILE = Path("cpp_model.pt")

SEQ_LEN = 128
EMBED_SIZE = 128
HIDDEN_SIZE = 256
LAYERS = 2

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.002


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print(f"Device: {DEVICE}")


def tokenize_cpp(text):
    pattern = r"""
        //.*?$ |
        /\*.*?\*/ |
        "(?:\\.|[^"\\])*" |
        '(?:\\.|[^'\\])*' |
        [A-Za-z_][A-Za-z0-9_]* |
        \d+(?:\.\d+)? |
        ==|!=|<=|>=|->|::|\+\+|--|&&|\|\||<<|>>|+=|-=|\*=|/= |
        [^\s]
    """

    return re.findall(
        pattern,
        text,
        flags=re.MULTILINE | re.DOTALL | re.VERBOSE,
    )


def load_cpp_files():
    files = []

    for extension in ("*.cpp", "*.cc", "*.cxx", "*.h", "*.hpp"):
        files.extend(DATA_DIR.rglob(extension))

    if not files:
        raise RuntimeError(
            f"No C++ files found in {DATA_DIR}"
        )

    tokens = []

    for path in files:
        print(f"Loading: {path}")

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        file_tokens = tokenize_cpp(text)

        tokens.extend(file_tokens)
        tokens.append("<FILE_END>")

    return tokens


tokens = load_cpp_files()

vocab = sorted(set(tokens))

stoi = {
    token: index
    for index, token in enumerate(vocab)
}

itos = {
    index: token
    for token, index in stoi.items()
}

encoded = torch.tensor(
    [stoi[token] for token in tokens],
    dtype=torch.long,
)


print()
print(f"Tokens: {len(tokens):,}")
print(f"Vocabulary: {len(vocab):,}")
print()


class CppLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        layers,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
        )

        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
        )

        self.output = nn.Linear(
            hidden_size,
            vocab_size,
        )

    def forward(self, x, hidden=None):

        x = self.embedding(x)

        x, hidden = self.rnn(
            x,
            hidden,
        )

        logits = self.output(x)

        return logits, hidden


model = CppLanguageModel(
    vocab_size=len(vocab),
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    layers=LAYERS,
).to(DEVICE)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

criterion = nn.CrossEntropyLoss()


def create_batches(data):

    inputs = []
    targets = []

    for start in range(
        0,
        len(data) - SEQ_LEN - 1,
        SEQ_LEN,
    ):
        x = data[
            start:
            start + SEQ_LEN
        ]

        y = data[
            start + 1:
            start + SEQ_LEN + 1
        ]

        inputs.append(x)
        targets.append(y)

    if not inputs:
        raise RuntimeError(
            "Not enough C++ code for the selected SEQ_LEN."
        )

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    for start in range(
        0,
        len(inputs),
        BATCH_SIZE,
    ):
        yield (
            inputs[start:start + BATCH_SIZE],
            targets[start:start + BATCH_SIZE],
        )


model.train()


for epoch in range(1, EPOCHS + 1):

    total_loss = 0.0
    batch_count = 0

    for x, y in create_batches(encoded):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits, _ = model(x)

        loss = criterion(
            logits.reshape(-1, len(vocab)),
            y.reshape(-1),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    average_loss = (
        total_loss / batch_count
    )

    print(
        f"Epoch {epoch:03d}/{EPOCHS} "
        f"Loss: {average_loss:.4f}"
    )


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

checkpoint = {
    "model_state_dict": model.state_dict(),

    "optimizer_state_dict": optimizer.state_dict(),

    "stoi": stoi,
    "itos": itos,

    "vocab_size": len(vocab),

    "seq_len": SEQ_LEN,
    "embed_size": EMBED_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "layers": LAYERS,

    "epochs": EPOCHS,
}


torch.save(
    checkpoint,
    MODEL_FILE,
)


print()
print("Training complete.")
print(f"Model saved to: {MODEL_FILE}")
```

## `generate.py`

This script **loads the `.pt` file** and generates C++ without retraining.

```python
from pathlib import Path
import torch
from torch import nn


MODEL_FILE = Path("cpp_model.pt")


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print(f"Device: {DEVICE}")


# --------------------------------------------------
# MODEL
# --------------------------------------------------

class CppLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        layers,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
        )

        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
        )

        self.output = nn.Linear(
            hidden_size,
            vocab_size,
        )

    def forward(self, x, hidden=None):

        x = self.embedding(x)

        x, hidden = self.rnn(
            x,
            hidden,
        )

        logits = self.output(x)

        return logits, hidden


# --------------------------------------------------
# LOAD
# --------------------------------------------------

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}"
    )


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False,
)


stoi = checkpoint["stoi"]
itos = checkpoint["itos"]


model = CppLanguageModel(
    vocab_size=checkpoint["vocab_size"],
    embed_size=checkpoint["embed_size"],
    hidden_size=checkpoint["hidden_size"],
    layers=checkpoint["layers"],
).to(DEVICE)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print("Model loaded.")
print(f"Vocabulary: {len(stoi):,}")
print()


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

def tokenize_cpp(text):

    import re

    pattern = r"""
        //.*?$ |
        /\*.*?\*/ |
        "(?:\\.|[^"\\])*" |
        '(?:\\.|[^'\\])*' |
        [A-Za-z_][A-Za-z0-9_]* |
        \d+(?:\.\d+)? |
        ==|!=|<=|>=|->|::|\+\+|--|&&|\|\||<<|>>|+=|-=|\*=|/= |
        [^\s]
    """

    return re.findall(
        pattern,
        text,
        flags=re.MULTILINE | re.DOTALL | re.VERBOSE,
    )


# --------------------------------------------------
# SAMPLING
# --------------------------------------------------

def sample_token(
    logits,
    temperature=0.8,
):

    logits = logits / temperature

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    token_id = torch.multinomial(
        probabilities,
        num_samples=1,
    )

    return token_id.item()


# --------------------------------------------------
# GENERATION
# --------------------------------------------------

def generate(
    prompt,
    length=200,
    temperature=0.8,
):

    prompt_tokens = tokenize_cpp(prompt)

    known_tokens = [
        token
        for token in prompt_tokens
        if token in stoi
    ]

    if not known_tokens:

        if "int" in stoi:
            known_tokens = ["int"]
        else:
            known_tokens = [
                next(iter(stoi))
            ]

    ids = [
        stoi[token]
        for token in known_tokens
    ]

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE,
    )

    generated = list(known_tokens)

    hidden = None

    with torch.no_grad():

        logits, hidden = model(
            x,
            hidden,
        )

        for _ in range(length):

            next_id = sample_token(
                logits[:, -1, :].squeeze(0),
                temperature,
            )

            next_token = itos[next_id]

            generated.append(
                next_token
            )

            next_input = torch.tensor(
                [[next_id]],
                dtype=torch.long,
                device=DEVICE,
            )

            logits, hidden = model(
                next_input,
                hidden,
            )

    return generated


# --------------------------------------------------
# BASIC C++ FORMATTING
# --------------------------------------------------

def format_cpp(tokens):

    output = ""

    for token in tokens:

        if token == "{":

            output += " {\n"

        elif token == "}":

            output += "\n}\n"

        elif token == ";":

            output += ";\n"

        elif token == "#":

            output += "#"

        elif token in {
            "(",
            "[",
        }:

            output += token

        elif token in {
            ")",
            "]",
            ",",
        }:

            output += token

        else:

            if output and not output.endswith(
                (
                    " ",
                    "\n",
                    "(",
                    "[",
                    "#",
                )
            ):
                output += " "

            output += token

    return output


# --------------------------------------------------
# GENERATE
# --------------------------------------------------

prompt = """
#include <iostream>

int main() {
"""

tokens = generate(
    prompt=prompt,
    length=200,
    temperature=0.75,
)


cpp_code = format_cpp(tokens)


print(cpp_code)
```

### Install PyTorch

```bash
pip install torch
```

### Train

Put your C++ source in `data/`:

```text
data/
├── main.cpp
├── engine.cpp
├── engine.hpp
├── player.cpp
└── player.hpp
```

Then:

```bash
python train.py
```

You'll get:

```text
cpp_model.pt
```

That single file contains:

```text
cpp_model.pt
├── neural-network weights
├── vocabulary
├── token mappings
├── model architecture
├── optimizer state
└── training information
```

### Load and generate

After training, you can completely close Python and later run:

```bash
python generate.py
```

It loads:

```python
checkpoint = torch.load(
    "cpp_model.pt",
    map_location=DEVICE,
    weights_only=False,
)
```

and restores the neural network with:

```python
model.load_state_dict(
    checkpoint["model_state_dict"]
)
```

So **`train.py` trains + saves**, while **`generate.py` loads + generates**.

If you eventually want this to become a real **C++ code-generation model**, the next step would be replacing the GRU with a small **PyTorch Transformer**, while keeping this same single-file save/load approach.
