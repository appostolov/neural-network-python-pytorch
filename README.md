# Email Syntax Validation Neural Network

A high-performance, character-level **1D Convolutional Neural Network (1D CNN)** built natively in PyTorch to classify email address structures as valid or invalid. The system achieves **99.xx% validation accuracy** by moving past simple structural shortcuts to learn robust sequential text dependencies.

---

## 📁 Project Architecture

The project is structured as a fully decoupled, portable Python package with a centralized control dashboard:

```text
my_email_project/
├── data/
│   └── emails.csv          # Balanced dataset (10k valid, 10k invalid edge cases)
├── src/
│   └── neural_network/
│       ├── __init__.py     # Package initialization marker
│       ├── __main__.py     # Parameterized entry point (sys.argv execution router)
│       ├── dataset.py      # Vectorized pre-tokenization RAM caching dataset
│       ├── models.py       # EmailCNNClassifier architecture definition
│       └── vocab/
│           ├── map.json   # 85-token core vocabulary configuration array
│           └── util.py    # Functional text-to-tensor tokenization logic
├── config.py               # Central control dashboard (Hyperparameters & paths)
├── pyproject.toml          # Modern setuptools build system declaration file
└── README.md
```

---

## 🚀 Setup and Installation

Follow these steps to initialize an isolated environment and install the package requirements:

### 1. Clone and Navigate
```bash
git clone https://github.com/appostolov/neural-network-python-pytorch.git
cd neural-network-python-pytorch
```

### 2. Activate Virtual Environment
* **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* **Windows (CMD / PowerShell):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate
  ```

### 3. Install in Editable Development Mode
Execute the modern build installation directly on the root directory. This downloads `torch` and `pandas` and hooks up a live shortcut to your source files:
```bash
pip install -e .
```

---

## 🕹️ Package Execution Interface

The package handles execution routing dynamically based on text parameters passed directly from the terminal line.

### ➔ Train the Model (80/20 Validation Split)
Loads your CSV dataset, caches numeric array variations entirely into RAM, partitions data via an 80/20 split, applies dropout regularization to prevent memorization, and exports optimized weights to disk:
```bash
python -m src.neural_network train
```

### ➔ Test Specific Strings (Real-Time Inference)
Loads the learned weight parameters into a frozen architecture (`model.eval()`) and returns an explicit boolean validation result along with a decimal confidence score:
```bash
python -m src.neural_network run robert.global.testing123@yahoo.com
```

---

## 📊 Core Optimizations and Safeguards

* **Vectorized Pre-Tokenization:** Eliminates slow string manipulation loops during active steps by converting the entire text database into sequential integer tensor blocks up-front inside `__init__`. Training data streaming latency drops to near zero.
* **Catastrophic Memorization Defense:** Includes an `nn.Dropout(p=0.3)` layer combined with scaled-down architectural filter limits (`EMBEDDING_DIM = 16`) and weight decay constraints. This forces the sliding filters to analyze global structural syntax (like domain boundaries) rather than memorizing training names.
* **Neutral Padding Allocation:** The vocabulary utilizes a dedicated `<PAD>` token (Index 1) that is forced to remain locked at absolute zero values via `padding_idx=1` in the embedding layer, preventing dead alignment space from introducing mathematical gradient bias.