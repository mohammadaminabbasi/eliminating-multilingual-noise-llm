# Eliminating Multilingual Noise in AI with Real-Time Language Filtering in vLLM

This project demonstrates a setup for deploying the Meta-Llama-3.1-8B-Instruct model with a real-time character filtering solution to reduce multilingual noise, specifically in low-resource languages. By integrating a Python-only development environment, we enable rapid development and testing for multilingual LLMs with custom filtering.

## Features
- **Character Filtering**: Filters out characters from unintended languages (e.g., Chinese, Hebrew, Korean) to maintain language coherence in responses.
- **Real-Time Processing**: Handles multilingual noise filtering in real-time, preserving relevant tokens for output.
- **Python-Only Development Mode**: Enables live code changes without recompilation, streamlining development.

## Quick Setup

1. **Install vLLM**:
   ```bash
   pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
   ```

2. **Set Up Local Development Environment**:
   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   python python_only_dev.py
   ```

3. **Run the vLLM Server**:
   ```bash
  vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto --api-key token-abc123
   ```

## Implementing Character Filtering

In the `step` function of the `LLMEngine` class, a character filter detects and suppresses tokens from unintended languages by adjusting their log probabilities. See [llm_engine.py](https://github.com/mohammadaminabbasi/eliminating-multilingual-noise-llm/blob/main/llm_engine.py) for the core code implementation.

```python
import re
def contains_specific_language(self, text):
    # Regex pattern to match Chinese, Korean, and Hebrew Unicode ranges
    pattern = r'[\u4E00-\u9FFF\uAC00-\uD7AF\u0590-\u05FF]'
    return bool(re.search(pattern, text))
outputs = self.model_executor.execute_model(
    execute_model_req=execute_model_req
)
print("Original output:", outputs)
logprobs = outputs[0].outputs[0].samples[0].logprobs
# Iterate through tokens and filter based on language-specific characters
for token_id in logprobs.keys():
    decoded_token = self.tokenizer_llama.decode(token_id)
    if self.contains_specific_language(decoded_token):
        logprobs[token_id].logprob = float('-inf')  # Set probability to near-zero
# Sorting log probabilities by updated values and re-ranking tokens
sorted_logprobs = sorted(logprobs.items(), key=lambda item: item[1].logprob, reverse=True)
sorted_logprobs_with_new_ranks = {
    item[0]: Logprob(logprob=item[1].logprob, rank=index + 1, decoded_token=item[1].decoded_token)
    for index, item in enumerate(sorted_logprobs)
}
# Update the output token with the highest-ranked token post-filtering
outputs[0].outputs[0].samples[0].output_token = list(sorted_logprobs.keys())[0]
print("Filtered output:", outputs)
```

## Exiting Development Mode

To exit the Python-only development environment:

```bash
python python_only_dev.py --quit-dev
```

## Key Takeaways

1. **Efficient Multilingual Support**: The character filter significantly improves output coherence without requiring model retraining.
2. **Scalability**: This setup supports large-scale deployments of multilingual LLMs, offering cleaner, language-specific outputs.
3. **Adaptability**: Ideal for use cases where maintaining language purity is essential, particularly in educational and inclusive AI applications.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
