from .base_llm import BaseLLM
from .data import Dataset, benchmark


class SFTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        SFT models are trained on raw questions without chat templates.
        Return the question as-is.
        """
        return question


def load() -> SFTModel:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = SFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # Claude Sonnet 4.5
    rounded_answer = round(answer, 2)
    
    return {
        "question": prompt,
        "answer": f"<answer>{rounded_answer}</answer>"
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str = "./homework/sft_model",
    num_train_epochs: int = 1,#5,
    learning_rate: float = 5e-4,  # Increased from 1e-4
    per_device_train_batch_size: int = 8,  # Reduced from 32 for stability
    **kwargs,
):

    # Claude Sonnet 4.5
    
    """
    Train a LoRA-adapted SFT model for unit conversion.
    """
    from pathlib import Path
    from transformers import TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    
    print("="*60)
    print("Setting up SFT Training with LoRA")
    print("="*60)
    
    # 1. Load base model
    print("\n1. Loading base model...")
    llm = SFTModel()
    
    # 2. Configure LoRA
    print("\n2. Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=16,                         # Increased rank for better capacity
        lora_alpha=32,                # 2x rank
        target_modules="all-linear",  # Apply to all linear layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 3. Apply LoRA to model
    print("\n3. Applying LoRA adapter to model...")
    llm.model = get_peft_model(llm.model, lora_config)
    
    # Enable gradient checkpointing for GPU memory savings
    if str(llm.device) == "cuda":
        llm.model.enable_input_require_grads()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in llm.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in llm.model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"Trainable: {100 * trainable_params / total_params:.2f}%")
    print(f"Model size: {trainable_params * 4 / 1024 / 1024:.2f} MB")
    
    # 4. Load and tokenize datasets
    print("\n4. Loading datasets...")
    train_dataset = TokenizedDataset(llm.tokenizer, Dataset("train"), format_example)
    eval_dataset = TokenizedDataset(llm.tokenizer, Dataset("valid"), format_example)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # 5. Set up training arguments
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
        gradient_checkpointing=False,  # Disable - causes NaN issues
        gradient_accumulation_steps=4,  # Accumulate to simulate larger batch
        
        # Logging and evaluation
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        
        # Other
        warmup_steps=50,  # Reduced warmup
        weight_decay=0.01,
        fp16=False,  # Disable FP16 - can cause NaN
        max_grad_norm=1.0,  # Clip gradients
    )
    
    # 6. Create trainer
    print("\n6. Creating trainer...")
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 7. Train!
    print("\n7. Starting training...")
    print("="*60)
    trainer.train()
    
    # 8. Save final model
    print("\n8. Saving model...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    
    # 9. Test the model
    print("\n9. Testing trained model...")
    print("="*60)
    test_model(output_dir)

'''
def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = SFTModel()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")
'''


# Claude Sonnet 4.5
def test_model(ckpt_path: str):
    """
    Test the trained SFT model on validation set.
    """
    print(f"\nLoading model from {ckpt_path}...")
    testset = Dataset("valid")
    llm = SFTModel()

    # Load the model with LoRA adapters
    from peft import PeftModel
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()
    
    # Test on a few examples first
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
    
    # Run full benchmark
    print("Running full benchmark...")
    print("-"*60)
    benchmark_result = benchmark(llm, testset, 100)
    print(f"\nResults:")
    print(f"  Accuracy: {benchmark_result.accuracy:.3f}")
    print(f"  Answer Rate: {benchmark_result.answer_rate:.3f}")

if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
