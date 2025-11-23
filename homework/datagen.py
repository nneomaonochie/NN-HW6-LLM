# Claude Sonnet 4.5

"""
Generate RFT dataset using rejection sampling.
"""
import json
from pathlib import Path
from tqdm import tqdm


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    """
    Generate RFT dataset using rejection sampling.
    
    For each question:
    1. Generate multiple completions with CoT reasoning
    2. Parse answers and check correctness
    3. Keep the first correct completion
    4. Save to JSON file
    
    Args:
        output_json: Path to save the generated dataset (e.g., "data/rft.json")
        oversample: Number of completions to generate per question
        temperature: Sampling temperature (higher = more diverse)
    """
    from .cot import CoTModel
    from .data import Dataset
    
    print("="*60)
    print("Generating RFT Dataset with Rejection Sampling")
    print("="*60)
    print(f"Oversample: {oversample} completions per question")
    print(f"Temperature: {temperature}")
    print()
    
    # Load CoT model (use larger 1.7B model for better results)
    print("Loading model (SmolLM2-1.7B-Instruct for better reasoning)...")
    try:
        model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
        print("✓ Using 1.7B model")
    except Exception as e:
        print(f"⚠ Could not load 1.7B model, using default 360M: {e}")
        model = CoTModel()
    
    # Load training dataset
    print("Loading train dataset...")
    dataset = Dataset("train")
    
    # Limit to first 500 samples for faster generation
    max_samples = 500
    dataset = dataset[:max_samples]
    
    print(f"Using {len(dataset)} samples (limited from full dataset)")
    
    # Generate dataset
    rft_data = []
    success_count = 0
    
    print(f"\nGenerating {len(dataset)} examples...")
    print("-"*60)
    
    for i, (question, correct_answer) in enumerate(tqdm(dataset, desc="Generating")):
        # Generate multiple completions
        completions = model.batched_generate(
            [question],
            num_return_sequences=oversample,
            temperature=temperature
        )[0]  # Get list of completions for this question
        
        # Find first correct completion
        found_correct = False
        for completion in completions:
            # Parse answer from completion
            parsed_answer = model.parse_answer(completion)
            
            # Check if answer is correct (within tolerance)
            if not (parsed_answer != parsed_answer):  # Not NaN
                error = abs(parsed_answer - correct_answer)
                relative_error = error / (abs(correct_answer) + 1e-8)
                
                # Accept if within 1% relative error or 0.01 absolute error
                if relative_error < 0.01 or error < 0.01:
                    # Found a correct answer! Save it
                    rft_data.append([question, correct_answer, completion])
                    success_count += 1
                    found_correct = True
                    break
        
        # Progress update every 100 examples
        if (i + 1) % 100 == 0:
            success_rate = success_count / (i + 1)
            print(f"\nProgress: {i+1}/{len(dataset)} | Success rate: {success_rate:.1%}")
            
            # SAVE CHECKPOINT every 100 samples
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(rft_data, f, indent=2)
            print(f"✓ Checkpoint saved to {output_path}")
    
    # Final statistics
    print("\n" + "="*60)
    print(f"Generation complete!")
    print(f"Total questions: {len(dataset)}")
    print(f"Successful generations: {success_count}")
    print(f"Success rate: {success_count / len(dataset):.1%}")
    print("="*60)
    
    # Save dataset
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(rft_data, f, indent=2)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"Dataset size: {len(rft_data)} examples")
    
    # Show a few examples
    if rft_data:
        print("\n" + "="*60)
        print("Sample entries:")
        print("="*60)
        for i, (q, a, r) in enumerate(rft_data[:3]):
            print(f"\nExample {i+1}:")
            print(f"Question: {q}")
            print(f"Answer: {a}")
            print(f"Reasoning: {r[:150]}...")  # Truncate long reasoning
    
    return rft_data


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)