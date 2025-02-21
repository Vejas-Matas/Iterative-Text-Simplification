import openai
import vllm
import datetime
import json
import torch

class VllmChatBot:
    def __init__(self, model_name):
        self.model = vllm.LLM(model_name, max_model_len=8192, dtype=torch.float16, quantization="awq", tensor_parallel_size=1, max_num_seqs=1) # Make this nicer !!!
        self.chat_log = []
        self.token_counts = []

    def add_system_prompt(self, prompt):
        self.chat_log.append({"role": "system", "content": prompt})
    
    def send_prompt(self, prompt):
        self.chat_log.append({"role": "user", "content": prompt}) 
        response = self.model.chat(
            messages=self.chat_log[:context_message_count],
            sampling_params=vllm.SamplingParams(temperature=0.5, max_tokens=1024), # Make this nicer !!!
        )
        self.chat_log.append({"role": "assistant", "content": response[0].outputs[0].text})

        num_prompt_tokens = len(response[0].prompt_token_ids)
        num_generated_tokens = len(response[0].outputs[0].token_ids) # sum(len(o.token_ids) for o in response[0].outputs)
        self.token_counts.append({"in": num_prompt_tokens, "out": num_generated_tokens})
        # self.token_counts.append({"in": len(response[0].prompt_token_ids), "out": len(response[0].usage.generated_tokens)})

    def send_limited_context_prompt(self, prompt, context_message_count=None):
        self.chat_log.append({"role": "user", "content": prompt}) 
        
        if context_message_count is not None:
            context_message_count = -context_message_count

        # There are 3 initial messages – task desciption, passage, and parameters
        initial_message_count = 3
        intial_messages = self.chat_log[:initial_message_count]
        last_messages = self.chat_log[initial_message_count:][context_message_count:]
        context = intial_messages + last_messages

        response = self.model.chat(
            messages=context,
            sampling_params=vllm.SamplingParams(temperature=0.5, max_tokens=1024), # Make this nicer !!!
        )
        self.chat_log.append({"role": "assistant", "content": response[0].outputs[0].text})

        num_prompt_tokens = len(response[0].prompt_token_ids)
        num_generated_tokens = len(response[0].outputs[0].token_ids) # sum(len(o.token_ids) for o in response[0].outputs)
        self.token_counts.append({"in": num_prompt_tokens, "out": num_generated_tokens})
        # self.token_counts.append({"in": len(response[0].prompt_token_ids), "out": len(response[0].usage.generated_tokens)})

    def get_last_response(self):
        return self.chat_log[-1]["content"]

    def get_token_usage(self):
        return {
            "in": sum([counts["in"] for counts in self.token_counts]),
            "out": sum([counts["out"] for counts in self.token_counts])
        }
    
    def print_chat(self):
        for message in self.chat_log:
            role = message["role"].upper()
            content = message["content"]
            print(f"{role}: {content}", end="\n\n")

    def print_token_usage_log(self):
        for entry in self.token_counts:
            print("IN = " + str(entry["in"]) + ", OUT = " + str(entry["out"]))

    def save_chat(self):
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        file_name = f"./runs/chat-log_{current_datetime}.json"
        with open(file_name, "w") as file:
            json.dump(self.chat_log, file, indent=2)

    def clear(self):
        self.chat_log = []
        self.token_counts = []


class OpenAIChatBot:
    def __init__(self, model_name, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.chat_log = []

    def add_system_prompt(self, prompt):
        self.chat_log.append({"role": "system", "content": prompt})
    
    def send_prompt(self, prompt):
        self.chat_log.append({"role": "user", "content": prompt}) 
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.chat_log
        )
        self.chat_log.append({"role": "assistant", "content": response.choices[0].message.content})

    def get_last_response(self):
        return self.chat_log[-1]["content"]
    
    def print_chat(self):
        for message in self.chat_log:
            role = message["role"].upper()
            content = message["content"]
            print(f"{role}: {content}", end="\n\n")

    def save_chat(self):
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        file_name = f"./runs/chat-log_{current_datetime}.json"
        with open(file_name, "w") as file:
            json.dump(self.chat_log, file, indent=2)

    def clear(self):
        self.chat_log = []
