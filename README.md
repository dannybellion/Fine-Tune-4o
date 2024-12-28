# Fine-Tune 4o: Custom LLM Training Pipeline

A robust Python-based pipeline for fine-tuning OpenAI models using Supervised Fine Tuning (SFT) and Direct Preference Optimization (DPO). This project provides a streamlined workflow for training, evaluating, and deploying custom language models.

## 🌟 Key Features

- Automated fine-tuning pipeline with OpenAI's API
- Support for Direct Preference Optimization (DPO)
- Real-time training metrics and visualization
- Flexible model evaluation and comparison

## 🏗️ Project Structure

```
.
├── src/                    # Source code
│   ├── fine_tune.py       # Core fine-tuning logic
│   ├── models.py          # Model interfaces and chat functionality
│   └── main.py           # Entry point and CLI
├── data/                  # Training and validation datasets
│   ├── fine_tune.jsonl    # Primary training data
│   ├── fine_tune_val.jsonl # Validation dataset
│   └── fine_tune_dpo.jsonl # DPO training pairs
├── docs/                  # Documentation
└── notebook.ipynb         # Interactive development and examples
```

## 💻 Core Components

### Fine-Tuning Engine (`src/fine_tune.py`)
The `FineTuner` class handles all fine-tuning operations:
- File upload and token counting
- Training job creation and management
- Real-time status monitoring
- Training metrics visualization
- Job cancellation and cleanup

### Model Interface (`src/models.py`)
The `Models` class provides:
- Chat completion interface
- Multi-response generation
- DPO preference collection
- Custom system prompts
- Temperature and sampling controls

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   ```

## 📊 Data Format

Training data should be in JSONL format with the following structure:

```json
{
    "messages": [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant response"}
    ]
}
```

## 🔧 Usage

Basic fine-tuning workflow:

```python
from src.fine_tune import FineTuner

# Initialize the fine-tuner
tuner = FineTuner()

# Upload training data
file_id = tuner.upload_file("data/fine_tune.jsonl")

# Create and monitor fine-tuning job
job_id = tuner.create_job(
    training_file=file_id,
    model="gpt-4o-mini-2024-07-18"
)

# Monitor progress and visualize metrics
tuner.plot_training_metrics(job_id)
```

## 📝 License

MIT License - Feel free to use and modify as needed!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
