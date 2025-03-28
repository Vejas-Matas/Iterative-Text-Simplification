import openai
import vllm
import datetime
import json
import torch
import easse.sari
import easse.bleu
import easse.fkgl

class VllmChatBot:
    ### Setup
    def __init__(self, model_name):
        self.model = vllm.LLM(model_name, max_model_len=8192, dtype=torch.float16, quantization="awq", tensor_parallel_size=1, max_num_seqs=1) # Make this nicer !!!
        self.sampling_parameters = vllm.SamplingParams(temperature=0.5, max_tokens=1024), # Make this nicer !!!
        self.mini_task_sampling_parameters = ingParams(temperature=0.1, max_tokens=1024), # Make this nicer !!!
        self.clear()
    
    def clear(self):
        self.chat_log = []
        self.token_count_log = []
        self.total_token_counts = {"in": 0, "out": 0}
        self.iteration_results = []
        self.sources = []
        self.references = []


    ### Chat
    def add_system_prompt(self, prompt):
        self.chat_log.append({"role": "system", "content": prompt})
    
    def send_prompt(self, prompt):
        self.chat_log.append({"role": "user", "content": prompt}) 
        response = self.model.chat(
            messages=self.chat_log,
            sampling_params=self.sampling_parameters,
        )
        self.chat_log.append({"role": "assistant", "content": response[0].outputs[0].text})
        self.increment_token_usage(response)

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
            sampling_params=self.sampling_parameters,
        )
        self.chat_log.append({"role": "assistant", "content": response[0].outputs[0].text})
        self.increment_token_usage(response)

    def send_no_context_prompts(self, prompts):
        response = self.model.chat(
            messages=prompts,
            sampling_params=self.mini_task_sampling_parameters,
        )
        return response[0].outputs[0].text

    ### Internal variable updates
    def increment_token_usage(self, response):
        num_prompt_tokens = len(response[0].prompt_token_ids)
        num_generated_tokens = len(response[0].outputs[0].token_ids) # sum(len(o.token_ids) for o in response[0].outputs)
        self.token_count_log.append({"in": num_prompt_tokens, "out": num_generated_tokens})
        # self.token_count_log.append({"in": len(response[0].prompt_token_ids), "out": len(response[0].usage.generated_tokens)})
        self.total_token_counts["in"] += num_prompt_tokens
        self.total_token_counts["out"] += num_generated_tokens

    def add_iteration_results(self):
        prediction = self.get_last_response()
        results = {
            "prediction": prediction,
            "metrics": {
                "sari": easse.sari.corpus_sari(sys_sents=[prediction], refs_sents=[self.references], orig_sents=self.sources),
                "bleu": easse.bleu.corpus_bleu(sys_sents=[prediction], refs_sents=[self.references]),
                "fkgl": easse.fkgl.corpus_fkgl(sentences=[prediction]),
                "in_tokens": self.total_token_counts["in"],
                "out_tokens": self.total_token_counts["out"],
            }
        }

        self.iteration_results.append(results)

    ### Getters
    def get_last_response(self):
        return self.chat_log[-1]["content"]

    def get_total_token_usage(self):
        return self.total_token_counts.copy()
    
    def get_iteration_results(self):
        return self.iteration_results.copy()

    def get_latest_fkgl(self):
        fkgl = self.iteration_results[-1]["metrics"]["fkgl"]
        return fkgl # f"{fkgl.2f}"

    ### Saving / displaying results
    def print_chat(self):
        for message in self.chat_log:
            role = message["role"].upper()
            content = message["content"]
            print(f"{role}: {content}", end="\n\n")

    def print_token_usage_log(self):
        for entry in self.token_count_log:
            print("IN = " + str(entry["in"]) + ", OUT = " + str(entry["out"]))

    def save_chat(self):
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        file_name = f"./runs/chat-log_{current_datetime}.json"
        with open(file_name, "w") as file:
            json.dump(self.chat_log, file, indent=2)



# # Outdated
# class OpenAIChatBot:
#     def __init__(self, model_name, api_key):
#         self.client = openai.OpenAI(api_key=api_key)
#         self.model_name = model_name
#         self.chat_log = []

#     def add_system_prompt(self, prompt):
#         self.chat_log.append({"role": "system", "content": prompt})
    
#     def send_prompt(self, prompt):
#         self.chat_log.append({"role": "user", "content": prompt}) 
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=self.chat_log
#         )
#         self.chat_log.append({"role": "assistant", "content": response.choices[0].message.content})

#     def get_last_response(self):
#         return self.chat_log[-1]["content"]
    
#     def print_chat(self):
#         for message in self.chat_log:
#             role = message["role"].upper()
#             content = message["content"]
#             print(f"{role}: {content}", end="\n\n")

#     def save_chat(self):
#         current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
#         file_name = f"./runs/chat-log_{current_datetime}.json"
#         with open(file_name, "w") as file:
#             json.dump(self.chat_log, file, indent=2)

#     def clear(self):
#         self.chat_log = []
