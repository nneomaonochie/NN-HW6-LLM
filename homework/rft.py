from .base_llm import BaseLLM
from .sft import test_model


class RFTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        RFT models are trained on raw questions without chat templates.
        Return the question as-is.
        """
        raise question

def load() -> RFTModel:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = RFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


# Claude Sonnet 4.5
def format_rft_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
    """
    Format RFT training example.
    The reasoning already contains the answer, so we just use it as-is.
    """
    return {
        "question": question,
        "answer": reasoning  # Reasoning includes the full chain-of-thought + answer
    }

# Claude Sonnet 4.5
class RFTDataset:
    """
    Dataset for RFT training.
    Loads from JSON file with [question, answer, reasoning] tuples.
    """
    def __init__(self, tokenizer, json_file: str):
        import json
        from pathlib import Path
        
        self.tokenizer = tokenizer
        
        # Load RFT data
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} RFT examples from {json_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        question, answer, reasoning = self.data[idx]
        formatted = format_rft_example(question, answer, reasoning)
        return tokenize(self.tokenizer, **formatted)



# Claude Sonnet 4.5
def train_model(
    output_dir: str = "./homework/rft_model",
    rft_data_file: str = "data/rft.json",
    num_train_epochs: int = 8,
    learning_rate: float = 5e-4,
    per_device_train_batch_size: int = 8,
    **kwargs,
):
    """
    Train RFT model using generated reasoning data.
    Similar to SFT but with chain-of-thought reasoning.
    """
    from pathlib import Path
    from transformers import TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    from .data import Dataset
    
    print("="*60)
    print("Training RFT Model with Chain-of-Thought Reasoning")
    print("="*60)
    
    # Check if RFT data exists
    if not Path(rft_data_file).exists():
        print(f"\n❌ ERROR: RFT dataset not found at {rft_data_file}")
        print("Please generate it first with:")
        print(f"  python -m homework.datagen --output_file {rft_data_file}")
        return
    
    # 1. Load base model
    print("\n1. Loading base model...")
    llm = RFTModel()
    
    # 2. Configure LoRA (slightly larger for reasoning)
    print("\n2. Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=24,                         # Larger rank for chain-of-thought
        lora_alpha=48,                # 2x rank
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 3. Apply LoRA
    print("\n3. Applying LoRA adapter...")
    llm.model = get_peft_model(llm.model, lora_config)
    
    if str(llm.device) == "cuda":
        llm.model.enable_input_require_grads()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in llm.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in llm.model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"Trainable: {100 * trainable_params / total_params:.2f}%")
    print(f"Model size: {trainable_params * 4 / 1024 / 1024:.2f} MB")
    
    # 4. Load datasets
    print("\n4. Loading datasets...")
    train_dataset = RFTDataset(llm.tokenizer, rft_data_file)
    eval_dataset = RFTDataset(llm.tokenizer, rft_data_file)  # Use same for eval
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # 5. Training arguments
    print("\n5. Configuring training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        
        # Training hyperparameters
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        
        # Memory optimization
        gradient_checkpointing=False,
        gradient_accumulation_steps=4,
        
        # Logging and evaluation
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        
        # Other
        warmup_steps=50,
        weight_decay=0.01,
        fp16=False,
        max_grad_norm=1.0,
    )
    
    # 6. Create trainer
    print("\n6. Creating trainer...")
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 7. Train
    print("\n7. Starting training...")
    print("="*60)
    trainer.train()
    
    # 8. Save model
    print("\n8. Saving model...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    
    # 9. Test
    print("\n9. Testing RFT model...")
    print("="*60)
    test_rft_model(output_dir)


# Claude Sonnet 4.5
def test_rft_model(ckpt_path: str):
    """
    Test the trained RFT model.
    """
    from .data import Dataset, benchmark
    
    print(f"\nLoading RFT model from {ckpt_path}...")
    testset = Dataset("valid")
    llm = RFTModel()
    
    from peft import PeftModel
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()
    
    # Test on sample questions
    print("\nTesting on sample questions:")
    print("-"*60)
    test_questions = [
        "Convert 5 meters to feet.",
        "How many yards are in 10 meters?",
    ]
    
    for q in test_questions:
        response = llm.generate(q)
        answer = llm.parse_answer(response)
        print(f"Q: {q}")
        print(f"A: {response}")
        print(f"Parsed: {answer}\n")
    
    # Run benchmark
    print("Running full benchmark...")
    print("-"*60)
    benchmark_result = benchmark(llm, testset, 100)
    print(f"\nResults:")
    print(f"  Accuracy: {benchmark_result.accuracy:.3f}")
    print(f"  Answer Rate: {benchmark_result.answer_rate:.3f}")





if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
