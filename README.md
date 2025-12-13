# Tiny Gemma Model Training

This project trains a small Gemma-based language model (~150M parameters) on the TinyStories dataset. The model is optimized for local training on AMD GPUs with ROCm support.

## Hardware Requirements

- AMD GPU with ROCm support (e.g., 8060s)
- 64GB dedicated GPU memory
- MAX 395+ AI chip or similar

## Software Requirements

- Python 3.8+
- PyTorch 2.4+ with ROCm support
- ROCm drivers and toolkit

## Environment Setup with pyenv

1. Install pyenv (if not already installed):
   ```bash
   curl https://pyenv.run | bash
   ```
   Add to your shell profile (e.g., ~/.bashrc or ~/.zshrc):
   ```bash
   export PATH="$HOME/.pyenv/bin:$PATH"
   eval "$(pyenv init --path)"
   eval "$(pyenv virtualenv-init -)"
   ```

2. Install Python 3.14.0 (or latest 3.8+):
   ```bash
   pyenv install 3.14.0
   ```

3. Set local Python version:
   ```bash
   pyenv local 3.14.0
   ```

4. Create and activate a virtual environment:
   ```bash
   pyenv virtualenv 3.14.0 tiny-story
   pyenv activate tiny-story
   ```

## Installation

1. Install dependencies (includes PyTorch with ROCm):
    ```bash
    pip install -r requirements.txt
    ```

3. Download the base Gemma-3-1B model and place it in `./gemma-3-1b/`

## Training

Run the training script:
```bash
python 1_train_english.py
```

The script will train for 5 epochs on the full TinyStories dataset and save the model to `./tiny-story/`.

## Inference

After training, use the model for text generation:
```python
from transformers import pipeline

pipe = pipeline('text-generation', model='./tiny-story', device=0)
print(pipe("Once upon a time"))
```

## Converting Chat Model to Ollama

To use the fine-tuned chat model (`tiny-story-chat/`) with Ollama, convert it to GGUF format first.

### Prerequisites
- Install Ollama from https://ollama.ai/
- Install sentencepiece: `pip install sentencepiece`

### Conversion Steps

1. Clone and build llama.cpp:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make
   ```

2. Convert the Hugging Face model to GGUF:
   ```bash
   python ../../cpp/llama.cpp/convert_hf_to_gguf.py ./tiny-story-chat --outtype f16 --outfile ../../ollama/tiny_story_chat.gguf
   ```

3. Create a Modelfile in the project root:
   ```
   FROM tiny-story-chat.gguf
   ```

4. Create the Ollama model:
   ```bash
   ollama create tiny-story-chat -f Modelfile
   ```

5. Run the model:
   ```bash
   ollama run tiny-story-chat
   ```
