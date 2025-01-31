#!/usr/bin/env python
# coding: utf-8

# ## Convert **.ipynb** to **.py**

# In[2]:


# !jupyter nbconvert --to python "Iterative Simplification.ipynb"


# ## ChatBot setup

# In[1]:


# %pip install openai
# %pip install evaluate
# %pip install py-readability-metrics
# %pip install sacrebleu
# %pip install sacremoses
# %pip install nltk
# %pip install textstat


# In[3]:


import openai
import vllm
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


# In[ ]:


nltk.download("punkt_tab")


# In[4]:


openai_api_key = ""
openai_model = "gpt-4o-mini"
vllm_model = "meta-llama/Llama-3.1-8B-Instruct"


# In[5]:


class OpenAIChatBot:
    def __init__(self, model, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.chat_log = []

    def add_system_prompt(self, prompt):
        self.chat_log.append({"role": "system", "content": prompt})
    
    def send_prompt(self, prompt):
        self.chat_log.append({"role": "user", "content": prompt}) 
        response = self.client.chat.completions.create(
            model=self.model,
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
            json.dump(self.chat_log, file)

    def clear_chat(self):
        self.chat_log = []


# In[ ]:


class VllmChatBot:
    def __init__(self, model_name):
        self.model = vllm.LLM(model_name, max_model_len=8192) # Make this nicer !!!
        self.chat_log = []

    def add_system_prompt(self, prompt):
        self.chat_log.append({"role": "system", "content": prompt})
    
    def send_prompt(self, prompt):
        self.chat_log.append({"role": "user", "content": prompt}) 
        response = self.model.chat(
            messages=self.chat_log,
            sampling_params=SamplingParams(max_tokens=8192), # Make this nicer !!!
        )
        self.chat_log.append({"role": "assistant", "content": response[0].outputs[0].text})

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
            json.dump(self.chat_log, file)

    def clear_chat(self):
        self.chat_log = []


# In[6]:


# # chat_bot = ChatBot(
# #     model="gpt-4o-mini",
# #     api_key=api_key,
# # )

# temp_bot = ChatBot(
#     model="gpt-4o-mini",
#     api_key=api_key,
# )
# temp_bot.add_system_prompt("You solve math equations")
# temp_bot.send_prompt("What is 2+3?")
# temp_bot.print_chat()


# In[7]:


algorithm_results = {}


# ## Evaluation

# #### Dataset

# In[ ]:


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


# In[ ]:





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

# In[9]:


bleu = evaluate.load("bleu")
sari = evaluate.load("sari")


# In[10]:


def compute_metrics(sources, predictions, references):
    # sources – original passages, predictions – simplified passages, references – target simplifications
    sari_nested_references = [[reference] for reference in references]
    
    results = {}
    results["SARI"] = sari.compute(sources=sources, predictions=predictions, references=sari_nested_references)["sari"]
    results["BLEU"] = bleu.compute(predictions=predictions, references=references)["bleu"]
    # results["FKGL"] = readability.Readability(simplified_passage).flesch_kincaid() # Should I use score or grade.level?
    results["FKGL"] = np.mean([textstat.flesch_kincaid_grade(passage) for passage in predictions])

    return results


# In[11]:


# >>> predictions = ["hello there general kenobi", "foo bar foobar"]
# >>> references = [
# ...     ["hello there general kenobi", "hello there !"],
# ...     ["foo bar foobar"]
# ... ]
# >>> bleu = evaluate.load("bleu")
# >>> results = bleu.compute(predictions=predictions, references=references)


# In[12]:


# sources=["About 95 species are currently accepted."]
# predictions=["About 95 you now get in."]
# references=[["About 95 species are currently known.","About 95 species are now accepted.","95 species are now accepted."]]
# sari_score = sari.compute(sources=sources, predictions=predictions, references=references)


# ## Algorithm

# In[13]:


algorithm_parameters = {
    "DC": "University student",
    "ILT": "Medium"
}


# In[14]:


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
"""


# In[15]:


non_iterative_system_prompt = """
You are a reader assistant. You simplify a passage from a scientific paper to make it more readable.

You are also given two parameters: DC (Desired Complexity) – a desired complexity of the simplified text, and ILT (Information Loss Tolerance) – a threshold for information loss compared to original passage.

Only print the simplified passage
"""


# In[16]:


aiir_mistral_system_prompt = """
You are a skilled editor, known for your ability to simplify complex text while preserving it. You explain the technical terms, defining what they are (e.g., terms like Blockchain, Cryptojacking, all abbreviations), without removing sentences or summarizing them.
"""

aiir_llama_run_1_system_prompt = """
Simplify this text for English speaking science students in college. Maximize the use of simple words and short sentences, but include keywords from the original text. Optimize the output ROUGE, SARI, and BLEU scores
"""


# In[ ]:





# In[29]:


def simplify_passage_iteratively(chat_bot, system_prompt, parameters, passage, max_iter=20):
    chat_bot.clear_chat()
    
    chat_bot.add_system_prompt(system_prompt)
    chat_bot.add_system_prompt(f"The passage:\n{passage}")

    if parameters is not None and parameters != {}:
        chat_bot.add_system_prompt("\n".join(f"{parameter}: {value}" for parameter, value in parameters.items()))

    for _ in range(max_iter):
        chat_bot.send_prompt("Identify which parts of the text are the most complex, then the complexity level of the passage")
        chat_bot.send_prompt(f'Is determined complexity higher than DC ({algorithm_parameters["DC"]})? Answer "Yes" or "No"')
        if "NO" in chat_bot.get_last_response().upper():
            break
        chat_bot.send_prompt(f'Identify a single complicated section of the passage. Remember to respect the ILT ({algorithm_parameters["ILT"]}) contraint. Only provide the identified section')
        chat_bot.send_prompt("Simplify this section. Only provide the proposed simplification")
        chat_bot.send_prompt("Reincorporate the simplified section into the passage")
        chat_bot.send_prompt("Identify information loss and its severity in the updated passage compared to the original. Comparison must be between the originally provided (the very first) passage and the current simplified version")
        chat_bot.send_prompt(f'What is the highest severity level identified in your last answer? Is it higher than ILT ({algorithm_parameters["ILT"]})? Provide the highest severity level, followed by an answer to the ILT question as "Yes" or "No"')
        if "YES" in chat_bot.get_last_response().upper():
            chat_bot.send_prompt("Revert the last proposed change. In further iterations you may still attempt to simplify this section in other ways")
        # else:
        #     chat_bot.send_prompt("If needed, adjust the passage to maintain readabily and flow of text")

    chat_bot.send_prompt("Print the final version of the simplified passage, include only the text of the passage with no comments or additional punctuation, and do not provide the original passage")
    # chat_bot.print_chat()
    chat_bot.save_chat()

    return chat_bot.get_last_response()


# In[30]:


def get_sources_and_references(passage_type_abbreviation, n):    
    sources = [entry[f"source_{passage_type_abbreviation}"] for entry in dataset["train"][passage_type_abbreviation]["source"][:n] ]
    references = [entry[f"simplified_{passage_type_abbreviation}"] for entry in dataset["train"][passage_type_abbreviation]["reference"][:n] ]

    return (sources, references)


# In[31]:


def simplify_passages(algorithm_fn, system_prompt, parameters, passage_type, max_iter, n=None):
    if passage_type == "abstract":
        sources, references = get_sources_and_references("abs", n)
    elif passage_type == "sentence":
        sources, references = get_sources_and_references("snt", n)
    else:
        raise ValueError('Passage type should be "abstract" or "sentence"')

    predictions = []
    results = []
    
    for i in range(len(sources)):
        # chat_bot = OpenAIChatBot(
        #     model=openai_model,
        #     api_key=api_key,
        # )
        chat_bot = VllmChatBot(
            model=vllm_model,
        )

        prediction = algorithm_fn(chat_bot, system_prompt, parameters, sources[i], max_iter)
        # metrics = compute_metrics([sources[i]], [prediction], [references[i]])

        predictions.append(prediction)
        results.append({
            "source": sources[i],
            "prediction": prediction,
            "reference": references[i],
            # "metrics": metrics,
        })

    overall_metrics = compute_metrics(sources, predictions, references)
    return (overall_metrics, results)


# In[32]:


passages_to_simplify = 1
passage_type_to_simplify = "sentence"

algorithm_results["iterative"] = simplify_passages(simplify_passage_iteratively, system_prompt, algorithm_parameters, passage_type_to_simplify, 20, passages_to_simplify)
# algorithm_results["non_iterative"] = simplify_passages(simplify_passage_iteratively, non_iterative_system_prompt, algorithm_parameters, passage_type_to_simplify, 0, passages_to_simplify)
# algorithm_results["aiir_mistral_prompt"] = simplify_passages(simplify_passage_iteratively, aiir_mistral_system_prompt, {}, passage_type_to_simplify, 0, passages_to_simplify)
# algorithm_results["aiir_llama_run_1_prompt"] = simplify_passages(simplify_passage_iteratively, aiir_llama_run_1_system_prompt, {}, passage_type_to_simplify, 0, passages_to_simplify)


# In[ ]:


print("METRICS:")
for algorithm, results in algorithm_results.items():
    print(f"{algorithm.upper()}: {results[0]}")


# In[ ]:


# print("SIMPLIFICATION EXAMPLES:")
# print(200*"–")
# for i in range(5):
#     print(f"{i}:")
#     print(f"SOURCE: {algorithm_results["iterative"][1][i]["source"]}")
#     print(f"REFERENCE: {algorithm_results["iterative"][1][i]["reference"]}")
#     for algorithm, results in algorithm_results.items():
#         print(f"{algorithm.upper()}:")
#         print(results[1][i]["prediction"])
#     print(200*"–")
#     print()


# In[32]:


# current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
# with open(f"./evaluations/{passage_type_to_simplify}s_university_medium_max20_{current_datetime}", "w") as file:
#     json.dump(algorithm_results, file)


# ## Playground

# In[54]:


# with open('evaluations/abstracts_university_medium_max20_2024-12-12_21-37-41.106794') as f:
#     old_simplifications = json.load(f)


# In[ ]:


# old_fkgl_scores = np.array([(textstat.flesch_kincaid_grade(passage_set["source"]), textstat.flesch_kincaid_grade(passage_set["prediction"])) for passage_set in old_simplifications["iterative"][1]])
# # old_fkgl_scores = np.array([(readability.Readability(passage_set["source"]).flesch_kincaid().score, readability.Readability(passage_set["prediction"]).flesch_kincaid().score) for passage_set in old_simplifications["iterative"][1]])
# print(old_fkgl_scores)

# # for passage_set in old_simplifications["iterative"][1][:20]:
# #     print(textstat.flesch_kincaid_grade(passage_set["source"]), end="")
# #     print(" –> ", end="")
# #     print(textstat.flesch_kincaid_grade(passage_set["prediction"]))


# In[ ]:


# diff = old_fkgl_scores[:, 0] - old_fkgl_scores[:, 1]
# dir = diff >= 0

# plt.scatter(old_fkgl_scores[:, 0], old_fkgl_scores[:, 1], abs(diff), np.where(dir, "b", "r"))
# plt.title("Abstract complexity change")
# plt.xlabel("Original passage FKGL")
# plt.ylabel("Simplified passage FKGL")
# plt.xscale("log")
# plt.yscale("log")
# plt.show()


# In[76]:


# outliers = []
# for passage_set, scores in zip(old_simplifications["iterative"][1], old_fkgl_scores):
#     if scores[0] - scores[1] < -20:
#         outliers.append((passage_set, scores))


# In[ ]:


# pprint.pprint(outliers)


# ## Graphs

# In[ ]:


# def my_exp(x, k, a):
#     return a * (np.exp(-k * (x-0.8)) - 1)

# # Define the range of x values
# x = np.linspace(0, 1, 500)

# # Define the functions
# exp_1 = my_exp(x, k=1, a=0.816)
# exp_2 = my_exp(x, k=2, a=0.253)
# exp_3 = my_exp(x, k=10, a=0.00033559)
# linear = 1 - 1.25 * x

# nullifier = np.where(x > 0.8, 0, 1)

# # Horizontal and vertical constants
# horizontal_constant = 0.25
# vertical_constant = 0.5

# plt.figure(figsize=(10, 6))

# # Functions
# plt.plot(x, linear * nullifier, color="red")
# plt.plot(x, exp_1 * nullifier, color="red")
# plt.plot(x, exp_2 * nullifier, color="red")
# plt.plot(x, exp_3 * nullifier, color="red")

# # Parameters
# plt.axhline(y=horizontal_constant, color="blue", linestyle="--")
# plt.axvline(x=vertical_constant, color="blue", linestyle="--")
# plt.fill_between(x, horizontal_constant, 1, color="blue", alpha=0.4)

# # Customize the plot
# plt.title("Complexity and Information Loss Trade-Off with Parameters")
# plt.xlabel("Complexity")
# plt.ylabel("Information loss")
# plt.axhline(0, color="black", linewidth=0.5, linestyle="-")  # x-axis
# plt.axvline(0, color="black", linewidth=0.5, linestyle="-")  # y-axis
# plt.grid(True, linestyle="--", alpha=0.7)


# plt.show()
# # plt.savefig("graphs/trade_off_parameters.pdf")


# In[ ]:





# In[ ]:




