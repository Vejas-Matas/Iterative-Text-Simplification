from vllm import LLM, SamplingParams

# Initialize LLaMa model with vLLM
llm = LLM(model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")  # Use your specific model

# Define sampling parameters
sampling_params = SamplingParams(max_tokens=100)  # Adjust as needed

# Example list of chat-based tasks
tasks = [
    [{"role": "user", "content": "Summarize the following text..."}],
    [{"role": "user", "content": "Translate this sentence into Spanish..."}],
    [{"role": "user", "content": "Explain the concept of entropy..."}]
]

token_counts = []
cumulative_tokens = 0  # Initialize cumulative token counter

for idx, task in enumerate(tasks):
    # Run inference
    outputs = llm.chat(task, sampling_params)

    # Extract and update token counts
    for output in outputs:
        num_prompt_tokens = len(output.prompt_token_ids)  # Input tokens
        num_generated_tokens = sum(len(o.token_ids) for o in output.outputs)  # Output tokens
        num_tokens = num_prompt_tokens + num_generated_tokens  # Total tokens

        token_counts.append(num_tokens)
        cumulative_tokens += num_tokens  # Update cumulative count

        print(f"Task {idx+1}: {task[0]['content']}")
        print(f"Prompt Tokens: {num_prompt_tokens}")
        print(f"Generated Tokens: {num_generated_tokens}")
        print(f"Total Processed Tokens: {num_tokens}")
        print(f"Cumulative Tokens: {cumulative_tokens}\n")

# Final token usage summary
print("Token usage per task:", token_counts)
print("Total cumulative tokens:", cumulative_tokens)
