from vllm import LLM, SamplingParams
import torch

# Initialize LLaMa model with vLLM
model_name = "meta-llama/Llama-3.1-8B-Instruct"
llm = LLM(model=model_name, max_model_len=30)  # Use your specific model
# model_name = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
# llm = LLM(model=model_name, max_model_len=8192, dtype=torch.float16, quantization="awq", tensor_parallel_size=1, max_num_seqs=1)  # Use your specific model

# Define sampling parameters
sampling_params = SamplingParams(max_tokens=10)  # Adjust as needed

# # Example list of chat-based tasks
# tasks = [
#     [{"role": "user", "content": "Summarize the following text..."}],
#     [{"role": "user", "content": "Translate this sentence into Spanish..."}],
#     [{"role": "user", "content": "Explain the concept of entropy..."}]
# ]

# token_counts = []
# cumulative_tokens = 0  # Initialize cumulative token counter

# for idx, task in enumerate(tasks):
#     # Run inference
#     outputs = llm.chat(task, sampling_params)

#     # Extract and update token counts
#     for output in outputs:
#         num_prompt_tokens = len(output.prompt_token_ids)  # Input tokens
#         num_generated_tokens = sum(len(o.token_ids) for o in output.outputs)  # Output tokens
#         num_tokens = num_prompt_tokens + num_generated_tokens  # Total tokens

#         token_counts.append(num_tokens)
#         cumulative_tokens += num_tokens  # Update cumulative count

#         print(f"Task {idx+1}: {task[0]['content']}")
#         print(f"Prompt Tokens: {num_prompt_tokens}")
#         print(f"Generated Tokens: {num_generated_tokens}")
#         print(f"Total Processed Tokens: {num_tokens}")
#         print(f"Cumulative Tokens: {cumulative_tokens}\n")

# # Final token usage summary
# print("Token usage per task:", token_counts)
# print("Total cumulative tokens:", cumulative_tokens)

# passage = "A significative percentage of the human population suffer from impairments in their capacity to distinguish or even see colours. For them, everyday tasks like navigating through a train or metro network map becomes demanding. We present a novel technique for extracting colour information from everyday natural stimuli and presenting it to visually impaired users as pleasant, non-invasive sound. This technique was implemented inside a Personal Digital Assistant (PDA) portable device. In this implementation, colour information is extracted from the input image and categorised according to how human observers segment the colour space. This information is subsequently converted into sound and sent to the user via speakers or headphones. In the original implementation, it is possible for the user to send its feedback to reconfigure the system, however several features such as these were not implemented because the current technology is limited.We are confident that the full implementation will be possible in the near future as PDA technology improves."
dc = "university student"
# Examples taken form https://readable.com/readability/flesch-reading-ease-flesch-kincaid-grade-level/
prompts = [
    "You are a helpful assistant that completes language tasks and provides output that is very strictly formatted and easy to parse for other machines",
    """Here are references for FKGL score ranges (score range : school level : example book):
0 - 3 : Kindergarten / Elementary : Hooray for Fish!
3 - 6 : Elementary : The Gruffalo
6 - 9 : Middle School : Harry Potter
9 - 12 : High School : Jurassic Park
12 - 15 : College : A Brief History of Time
17 - 20 : Post-grad : Academic Papers""",
    f'What is an equivalent FKGL score for the following type of reader: <{dc}>. Only provide a floating point number with two decimal points such as "12.34", or "None" input is not appropriate. Do not include anything else in your answer',
]
response = llm.chat(prompts, sampling_params) #[0].outputs[0].text
print(response)
# print("\n\n\n[ " + response + "]\n\n\n")
