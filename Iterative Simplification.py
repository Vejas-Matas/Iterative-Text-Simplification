# ## ChatBot setup

# %pip install openai
# %pip install evaluate
# %pip install py-readability-metrics
# %pip install sacrebleu
# %pip install sacremoses
# %pip install nltk
# %pip install textstat

import openai
import vllm
import torch
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import pprint

# NLP packages
import evaluate
import readability
import nltk
import textstat

# My own files
import file_io_utils

nltk.download("punkt_tab")


openai_api_key = ""
openai_model = "gpt-4o-mini"
# vllm_model = "meta-llama/Llama-3.1-8B-Instruct"
vllm_model = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"


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
            messages=self.chat_log,
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


algorithm_results = {}


# ## Evaluation

# #### Dataset

# Rewrite to class?

dataset = {}

for data_type in ["train", "test"]:
    dataset[data_type] = {}
    for passage_length in ["abs", "snt"]:
        dataset[data_type][passage_length] = {}

for data_type in ["train", "test"]:
    for passage_length in ["abs", "snt"]:
        for passage_type in ["source", "reference"]:
            if data_type == "test" and passage_type == "reference":
                continue
                
            folder = f"./dataset/{data_type}/"
            file_name = f"simpletext_task3_2024_{data_type}_{passage_length}_{passage_type}.json"
            path = folder + file_name

            print("Reading: " + path)
            with open(path, "r", encoding="utf8") as file:
                data = json.load(file)
            
            dataset[data_type][passage_length][passage_type] = data


# #### Metrics

# Mssing metrics:
# - count
# - Compression ratio
# - Sentence splits
# - Levenshtein similarity
# - Exact copies
# - Additions proportion
# - Deletions proportion
# - Lexical complexity score


bleu = evaluate.load("bleu")
sari = evaluate.load("sari")


def compute_metrics(sources, predictions, references):
    # sources – original passages, predictions – simplified passages, references – target simplifications
    sari_nested_references = [[reference] for reference in references]
    
    results = {}
    results["SARI"] = sari.compute(sources=sources, predictions=predictions, references=sari_nested_references)["sari"]
    results["BLEU"] = bleu.compute(predictions=predictions, references=references)["bleu"]
    # results["FKGL"] = readability.Readability(simplified_passage).flesch_kincaid() # Should I use score or grade.level?
    results["FKGL"] = np.mean([textstat.flesch_kincaid_grade(passage) for passage in predictions])

    return results


algorithm_parameters = {
    "DC": "University student",
    "ILT": "Medium"
}

system_prompt = """
You are a reader assistant. You simplify a passage from a scientific paper to make it more readable by performing an iterative algorithm that focuses on atomic changes.

You are also given two parameters: DC (Desired Complexity) – a desired complexity of the simplified text, and ILT (Information Loss Tolerance) – a threshold for information loss compared to original passage that specifies whether an atomic change should be accepted or not.

The algorithm is as follows:
1. Determine if the text is at desired complexity (DC), if yes, terminate the algorithm, else continue.
2. Identify a section of the text whose complexity is above DC.
3. Propose a simplification of the identified section.
4. Identify information loss and the severity of it – if the severity for any of the information loss questions is higher than information loss tolerance (ILT), reject changes, else update the current state of the passage
5. Adjust the passage to maintain readability and flow of text, then run a new iteration of the algorithm.

When the algorithm terminates, you print the final simplified passage

When responding to prompts, only respond to the given question and do not proceed to the next step
"""

non_iterative_system_prompt = """
You are a reader assistant. You simplify a passage from a scientific paper to make it more readable.

You are also given two parameters: DC (Desired Complexity) – a desired complexity of the simplified text, and ILT (Information Loss Tolerance) – a threshold for information loss compared to original passage.

Only print the simplified passage
"""

aiir_mistral_system_prompt = """
You are a skilled editor, known for your ability to simplify complex text while preserving it. You explain the technical terms, defining what they are (e.g., terms like Blockchain, Cryptojacking, all abbreviations), without removing sentences or summarizing them.
"""

aiir_llama_run_1_system_prompt = """
Simplify this text for English speaking science students in college. Maximize the use of simple words and short sentences, but include keywords from the original text. Optimize the output ROUGE, SARI, and BLEU scores
"""


def simplify_passage_iteratively(chat_bot, system_prompt, parameters, passage, max_iter=20):
    chat_bot.clear()
    
    chat_bot.add_system_prompt(system_prompt)
    chat_bot.add_system_prompt(f"The passage:\n{passage}")

    if parameters is not None and parameters != {}:
        chat_bot.add_system_prompt("\n".join(f"{parameter}: {value}" for parameter, value in parameters.items()))

    for _ in range(max_iter):
        chat_bot.send_prompt("Identify which parts of the text are the most complex, then the complexity level of the passage. Limit your answer to a maximum of 5 sentences")
        chat_bot.send_prompt(f'Is determined complexity higher than DC ({algorithm_parameters["DC"]})? Answer "Yes" or "No"')
        if "NO" in chat_bot.get_last_response().upper():
            break
        chat_bot.send_prompt(f'Identify a single complicated section of the passage. Remember to respect the ILT ({algorithm_parameters["ILT"]}) contraint. Only provide the identified section')
        chat_bot.send_prompt("Simplify this section. Only provide the proposed simplification")
        chat_bot.send_prompt("Reincorporate the simplified section into the passage")
        chat_bot.send_prompt("Identify information loss and its severity in the updated passage compared to the original. Comparison must be between the originally provided (the very first) passage and the current simplified version. Limit your answer to a maximum of 5 sentences")
        chat_bot.send_prompt(f'What is the highest severity level identified in your last answer? Is it higher than ILT ({algorithm_parameters["ILT"]})? Provide the highest severity level, followed by an answer to the ILT question as "Yes" or "No"')
        if "YES" in chat_bot.get_last_response().upper():
            chat_bot.send_prompt("Revert the last proposed change. In further iterations you may still attempt to simplify this section in other ways")
        # else:
        #     chat_bot.send_prompt("If needed, adjust the passage to maintain readabily and flow of text")

    chat_bot.send_prompt("Print the final version of the simplified passage, include only the text of the passage with no comments or additional punctuation, and do not provide the original passage")
    # chat_bot.print_chat()
    # chat_bot.save_chat()
    # chat_bot.print_token_usage_log()

    return chat_bot.get_last_response()


def get_sources_and_references(passage_type_abbreviation, n):    
    sources = [entry[f"source_{passage_type_abbreviation}"] for entry in dataset["train"][passage_type_abbreviation]["source"][:n] ]
    references = [entry[f"simplified_{passage_type_abbreviation}"] for entry in dataset["train"][passage_type_abbreviation]["reference"][:n] ]

    return (sources, references)



def simplify_passages(algorithm_name, algorithm_fn, system_prompt, parameters, passage_type, max_iter, n=None):
    if passage_type == "abstract":
        sources, references = get_sources_and_references("abs", n)
    elif passage_type == "sentence":
        sources, references = get_sources_and_references("snt", n)
    else:
        raise ValueError('Passage type should be "abstract" or "sentence"')

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    parameter_string = ("dc=" + parameters["DC"] + "_" + "ilt=" + parameters["DC"]).lower().replace(" ", "_")
    results_file_name = f"algorithm={algorithm_name}_type={passage_type}_{parameter_string}_i={max_iter}_timestamp={timestamp}"

    predictions = []
    results = []
    token_usage = []

    # chat_bot = OpenAIChatBot(
    #     model=openai_model,
    #     api_key=api_key,
    # )
    chat_bot = VllmChatBot(
        model_name=vllm_model,
    )
    
    for i in range(len(sources)):

        prediction = algorithm_fn(chat_bot, system_prompt, parameters, sources[i], max_iter)
        # metrics = compute_metrics([sources[i]], [prediction], [references[i]])
        token_usage.append(chat_bot.get_token_usage())

        predictions.append(prediction)
        results.append({
            "source": sources[i],
            "prediction": prediction,
            "reference": references[i],
            # "metrics": metrics,
        })

        file_io_utils.append_to_txt(f"predictions/{results_file_name}", prediction)

        chat_bot.clear()

    # overall_metrics = compute_metrics(sources, predictions, references)
    # return (overall_metrics, results)


    overall_metrics = {
        "in_tokens": sum([counts["in"] for counts in token_usage]),
        "out_tokens": sum([counts["out"] for counts in token_usage])
    }

    file_io_utils.convert_dict_to_json(f"metrics/{results_file_name}_metrics.json", overall_metrics)

    return (results, overall_metrics)


passages_to_simplify = 50
passage_type_to_simplify = "sentence"

# simplify_passages("iterative", simplify_passage_iteratively, system_prompt, algorithm_parameters, passage_type_to_simplify, 20, passages_to_simplify)
simplify_passages("non_iterative", simplify_passage_iteratively, non_iterative_system_prompt, algorithm_parameters, passage_type_to_simplify, 0, passages_to_simplify)

# algorithm_results["iterative"] = simplify_passages(simplify_passage_iteratively, system_prompt, algorithm_parameters, passage_type_to_simplify, 20, passages_to_simplify)
# algorithm_results["non_iterative"] = simplify_passages(simplify_passage_iteratively, non_iterative_system_prompt, algorithm_parameters, passage_type_to_simplify, 0, passages_to_simplify)
# algorithm_results["aiir_mistral_prompt"] = simplify_passages(simplify_passage_iteratively, aiir_mistral_system_prompt, {}, passage_type_to_simplify, 0, passages_to_simplify)
# algorithm_results["aiir_llama_run_1_prompt"] = simplify_passages(simplify_passage_iteratively, aiir_llama_run_1_system_prompt, {}, passage_type_to_simplify, 0, passages_to_simplify)
