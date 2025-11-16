from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """


        # Claude Sonnet 4.5
        """
        Create a chat template with instructions and in-context examples for unit conversion.
        
        Strategy:
        1. System message: Brief instructions
        2. User example: Example question
        3. Assistant example: Show chain-of-thought reasoning with answer tags
        4. User message: Actual question to answer
        """
        
        messages = [
            # System message: Clear, concise instructions
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that converts units. "
                    "Show your reasoning step-by-step, then provide the final answer "
                    "in <answer>NUMBER</answer> tags. Be concise."
                )
            },
            
            # In-context example: User question
            {
                "role": "user",
                "content": "Convert 10 meters to feet."
            },
            
            # In-context example: Assistant response with chain-of-thought
            {
                "role": "assistant",
                "content": (
                    "To convert meters to feet:\n"
                    "1 meter = 3.28084 feet\n"
                    "10 meters = 10 × 3.28084 = 32.8084 feet\n"
                    "<answer>32.8084</answer>"
                )
            },
            
            # Actual question to answer
            {
                "role": "user",
                "content": question
            }
        ]
        
        # Convert to chat template format
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt


def load() -> CoTModel:
    return CoTModel()

'''
def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")
'''

# Claude Sonnet 4.5
def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    
    # Test on a few examples first
    print("\n" + "="*60)
    print("Testing individual examples:")
    print("="*60)
    
    test_questions = [
        "Convert 5 meters to yards.",
        "How many feet are in 20 meters?",
        "Convert 100 yards to meters."
    ]
    
    for q in test_questions:
        print(f"\nQuestion: {q}")
        response = model.generate(q)
        print(f"Response: {response}")
        answer = model.parse_answer(response)
        print(f"Parsed answer: {answer}")
    
    # Run full benchmark
    print("\n" + "="*60)
    print("Running benchmark on validation set:")
    print("="*60)
    benchmark_result = benchmark(model, testset, 100)
    print(f"\nResults:")
    print(f"  Accuracy: {benchmark_result.accuracy:.3f}")
    print(f"  Answer Rate: {benchmark_result.answer_rate:.3f}")
    print(f"\nTarget: accuracy ≥ 0.5, answer_rate ≥ 0.85")

if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
