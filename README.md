# LLM Experiment Framework

A comprehensive framework for experimenting with LLM fine-tuning and agentic workflows, designed for Apple Silicon and free API endpoints.

## Features

### Training & Fine-Tuning
- **PEFT Methods**: LoRA, QLoRA, Prefix Tuning, and more
- **Advanced Training**: Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO)
- **Apple Silicon Optimized**: Native MPS backend support for Mac
- **MLflow Integration**: Track experiments, metrics, and model artifacts

### Model Support
- **Local Models**: HuggingFace transformers with quantization (4-bit, 8-bit)
- **API Providers**: Groq, HuggingFace Inference API, Ollama
- **Model Families**: Llama, Mistral, Qwen, Phi, Gemma
- **Easy Switching**: Change models via configuration files

### Agentic Workflows
- **ReAct Agent**: Reasoning + Acting with tool use
- **RAG Pipeline**: Retrieval-Augmented Generation with vector stores
- **Multi-Agent Systems**: Sequential, hierarchical, and collaborative orchestration

### Data Processing
- **CSV/TSV Support**: Load and preprocess tabular data
- **Flexible Formats**: Instruction tuning, chat format, preference pairs
- **Custom Preprocessing**: Text cleaning, filtering, format conversion

## Installation

### Prerequisites
- Python 3.10+
- Apple Silicon Mac (for local training) or any platform (for API usage)

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd claude-code-experiment
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or .venv\Scripts\activate  # On Windows
```

3. **Install with uv (recommended)**
```bash
pip install uv
uv pip install -e .
```

Or with pip:
```bash
pip install -e .
```

4. **Set up API keys** (optional, for API usage)
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Quick Start

### 1. Fine-Tuning a Model

Train a model using LoRA on your custom dataset:

**With uv:**
```bash
uv run scripts/train.py \
  model=qwen2_7b \
  training=lora \
  data.paths.train_file=data/raw/your_data.csv
```

**With Python:**
```bash
python scripts/train.py \
  model=qwen2_7b \
  training=lora \
  data.paths.train_file=data/raw/your_data.csv
```

**Example with Groq API:**
```bash
# With uv
uv run scripts/train.py model=api_groq training=lora data.paths.train_file=data/raw/sample_instructions.csv

# With Python
python scripts/train.py model=api_groq training=lora data.paths.train_file=data/raw/sample_instructions.csv
```

### 2. Running Inference

Generate text with a trained model:

**With uv:**
```bash
uv run scripts/inference.py \
  model=qwen2_7b \
  checkpoint_path=outputs/checkpoints/final \
  prompt="Explain quantum computing"
```

**With Python:**
```bash
python scripts/inference.py \
  model=qwen2_7b \
  checkpoint_path=outputs/checkpoints/final \
  prompt="Explain quantum computing"
```

### 3. Using Agents

**ReAct Agent with Tools:**
```bash
# With uv
uv run scripts/run_agent.py agent=react model=api_groq +query="What is 123 multiplied by 456?"

# With Python
python scripts/run_agent.py agent=react model=api_groq +query="What is 123 multiplied by 456?"
```

**RAG Pipeline:**
```bash
# With uv
uv run scripts/run_agent.py agent=rag model=api_groq +query="What is machine learning?"

# With Python
python scripts/run_agent.py agent=rag model=api_groq +query="What is machine learning?"
```

**Multi-Agent System:**
```bash
# With uv
uv run scripts/run_agent.py agent=multi_agent model=api_groq +query="Write a blog post about AI"

# With Python
python scripts/run_agent.py agent=multi_agent model=api_groq +query="Write a blog post about AI"
```

## Configuration

The framework uses [Hydra](https://hydra.cc/) for configuration management. All configs are in `configs/`.

### Directory Structure
```
configs/
├── config.yaml           # Main config with defaults
├── model/                # Model configurations
│   ├── qwen2_7b.yaml
│   ├── llama3_8b.yaml
│   ├── api_groq.yaml
│   └── ollama.yaml
├── training/             # Training method configs
│   ├── lora.yaml
│   ├── qlora.yaml
│   ├── dpo.yaml
│   └── sft.yaml
├── data/                 # Data processing configs
│   ├── instruction.yaml
│   └── preference.yaml
└── agent/                # Agent workflow configs
    ├── react.yaml
    ├── rag.yaml
    └── multi_agent.yaml
```

### Customizing Configurations

Create your own config or override parameters:

```bash
# Override specific parameters (with uv)
uv run scripts/train.py \
  model=qwen2_7b \
  training=lora \
  training.lora.r=16 \
  training.lora.alpha=32 \
  training.num_epochs=5

# Override specific parameters (with Python)
python scripts/train.py \
  model=qwen2_7b \
  training=lora \
  training.lora.r=16 \
  training.lora.alpha=32 \
  training.num_epochs=5

# Use multiple config files
uv run scripts/train.py model=llama3_8b training=qlora data=instruction
# or
python scripts/train.py model=llama3_8b training=qlora data=instruction
```

## Data Format

### Instruction Tuning Format (CSV)

Your CSV should have these columns:
- `instruction`: The task description
- `input`: Optional context or input (can be empty)
- `output`: The expected output

Example:
```csv
instruction,input,output
"Translate to French","Hello, how are you?","Bonjour, comment allez-vous?"
"Summarize this text","Long article text...","Brief summary..."
```

### Preference Pairs for DPO (CSV)

For DPO training:
- `instruction`: The task/question
- `chosen`: Preferred response
- `rejected`: Less preferred response

Example:
```csv
instruction,chosen,rejected
"What is AI?","AI is artificial intelligence...","AI is computers..."
```

## Project Structure

```
.
├── configs/              # Hydra configuration files
├── data/
│   ├── raw/             # Your raw CSV/TSV data
│   └── processed/       # Preprocessed datasets
├── src/
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model abstractions (local + API)
│   ├── training/        # Training implementations
│   ├── agents/          # Agentic workflows
│   └── utils/           # MLflow and utilities
├── scripts/             # Entry point scripts
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation script
│   ├── inference.py     # Inference script
│   └── run_agent.py     # Agent runner
├── outputs/             # Training outputs (checkpoints, logs)
├── mlruns/              # MLflow tracking data
└── pyproject.toml       # Project dependencies
```

## Advanced Usage

### Custom Model Configuration

Create a new model config in `configs/model/my_model.yaml`:

```yaml
name: my_custom_model
family: llama
source: local
model_path: meta-llama/Llama-2-7b-hf

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16

generation:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
```

### Custom Training Configuration

Create `configs/training/my_training.yaml`:

```yaml
method: lora
num_epochs: 5
batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4

lora:
  r: 32
  alpha: 64
  dropout: 0.1
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
```

### MLflow Tracking

View your experiments:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### Using Different API Providers

**Groq API:**
```bash
export GROQ_API_KEY=your_key_here

# With uv
uv run scripts/run_agent.py agent=react model=api_groq

# With Python
python scripts/run_agent.py agent=react model=api_groq
```

**HuggingFace Inference API:**
```bash
export HF_API_KEY=your_key_here

# With uv
uv run scripts/train.py model=api_huggingface training=lora

# With Python
python scripts/train.py model=api_huggingface training=lora
```

**Ollama (local):**
```bash
# Make sure Ollama is running
ollama serve

# With uv
uv run scripts/inference.py model=ollama

# With Python
python scripts/inference.py model=ollama
```

## Examples

### Example 1: QLoRA Fine-Tuning

Fine-tune a 7B model with 4-bit quantization:

```bash
# With uv
uv run scripts/train.py \
  model=qwen2_7b \
  training=qlora \
  data.paths.train_file=data/raw/instructions.csv \
  training.num_epochs=3 \
  training.batch_size=4

# With Python
python scripts/train.py \
  model=qwen2_7b \
  training=qlora \
  data.paths.train_file=data/raw/instructions.csv \
  training.num_epochs=3 \
  training.batch_size=4
```

### Example 2: DPO Training

Train with preference pairs:

```bash
# With uv
uv run scripts/train.py \
  model=llama3_8b \
  training=dpo \
  data=preference \
  data.paths.train_file=data/raw/preferences.csv

# With Python
python scripts/train.py \
  model=llama3_8b \
  training=dpo \
  data=preference \
  data.paths.train_file=data/raw/preferences.csv
```

### Example 3: RAG with Custom Documents

```python
from src.agents import RAGPipeline
from langchain_groq import ChatGroq
import os

# Set up RAG
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
cfg = OmegaConf.load("configs/agent/rag.yaml")

rag = RAGPipeline(cfg=cfg, llm=llm)
rag.setup()

# Ingest documents
rag.ingest_documents("data/raw/documents/", file_type="txt")

# Query
result = rag.query("What is the main topic of the documents?")
print(result["answer"])
```

### Example 4: Multi-Agent Workflow

```python
from src.agents import MultiAgentOrchestrator
from langchain_groq import ChatGroq
from omegaconf import OmegaConf

llm = ChatGroq(model="llama-3.3-70b-versatile")
cfg = OmegaConf.load("configs/agent/multi_agent.yaml")

orchestrator = MultiAgentOrchestrator(cfg=cfg, llm=llm)
orchestrator.setup()

result = orchestrator.run(
    task="Research and write about renewable energy",
    context={"focus": "solar power"}
)
print(result["output"])
```

## Troubleshooting

### Mac MPS Issues

If you encounter MPS errors with local models:
1. Use API providers (Groq, Ollama) instead
2. Or install Ollama and run models locally with full compatibility

### Memory Issues

For 32GB Mac:
- 7B models: Use `load_in_4bit: true`
- 13B models: Use `load_in_4bit: true` with smaller batch sizes
- 70B+ models: Use API providers (Groq, HuggingFace)

### Import Errors

If you see import errors, reinstall:
```bash
uv pip install -e .
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with:
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)
- [LangChain](https://www.langchain.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Hydra](https://hydra.cc/)
- [MLflow](https://mlflow.org/)
