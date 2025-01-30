from vllm import LLM, SamplingParams

MODEL_NAME = 'meta-llama/Llama-3.1-8B-Instruct'

llm = LLM(MODEL_NAME, max_model_len=8192) # you probably want to adjust max_model_len
generation_args = SamplingParams(max_tokens=8192) # a bunch of things to adjust here too

PROMPT = '''
What is 2 + 2?
'''

result = llm.generate([PROMPT], generation_args)

print(result[0].outputs[0].text)

# result[0].outputs[0].text
