import glob
import re
import json
import numpy as np
import pandas as pd

import file_io_utils
import parameters
import chat_bots


def compare_information_units(chat_bot, source, prediction):
    fact_extraction_prompts = [
        {"role": "system", "content": "You are a diligigent and attentive text evaluator. You extract factual information from passages. Each fact must be an atomic information unit, expressed as a clause. Provide these units as a numbered list, do not include any other text besides the list"},
        {"role": "system", "content": f"The first pair of examples of information extraction (desired output are the numbered lists):\n\n{parameters.information_extraction_example_1}"},
        {"role": "system", "content": f"The second pair of examples of information extraction (desired output are the numbered lists):\n\n{parameters.information_extraction_example_2}"},
        {"role": "user",   "content": "Extract atomic information units from the following passage. Only provide the list"},
    ]

    source_facts = chat_bot.send_no_context_prompts(fact_extraction_prompts + [{"role": "user",   "content": source}])
    prediction_facts = chat_bot.send_no_context_prompts(fact_extraction_prompts + [{"role": "user",   "content": prediction}])

    fact_comparison_prompts = [
        {"role": "system",      "content": parameters.information_comparison_instructions},
        {"role": "system",      "content": f"The first example of information extraction (will already be provided):\n\n{parameters.information_extraction_example_1}"},
        {"role": "system",      "content": f"The first example of information comparison (desired output):\n\n{parameters.information_comparison_example_1}"},
        {"role": "system",      "content": f"The second example of information extraction (will already be provided):\n\n{parameters.information_extraction_example_2}"},
        {"role": "system",      "content": f"The second example of information comparison (desired output):\n\n{parameters.information_comparison_example_2}"},
        {"role": "user",        "content": "Extract information units from the following passage (original)"},
        {"role": "user",        "content": source},
        {"role": "assistant",   "content": source_facts},
        {"role": "user",        "content": "Extract information units from the following passage (simplified)"},
        {"role": "user",        "content": prediction},
        {"role": "assistant",   "content": prediction_facts},
        {"role": "user",        "content": "Analyse the data and provide the four lists: PRESERVATIONS, OVERSIMPLIFICATIONS, DELETIONS, HALLACINATIONS. One fact pair should be present in only one list most of the time, a fact might be in multiple pairs if appropriate"},
    ]

    fact_comparison = chat_bot.send_no_context_prompts(fact_comparison_prompts)

    return fact_comparison

def convert_fact_string_to_dict(text):
    sections = ["PRESERVATIONS", "OVERSIMPLIFICATIONS", "DELETIONS", "HALLUCINATIONS"]
    pattern = "|".join(f"(?={s}:)" for s in sections)
    parts = re.split(pattern, text.strip())

    result = {}
    for part in parts:
        if not part.strip():
            continue

        title_match = re.match(r"([A-Z]+):", part.strip())
        if not title_match:
            continue

        title = title_match.group(1).lower()
        content = part.strip()[len(title) + 1:].strip()

        if content.startswith("0. "):
            result[title] = []
        else:
            items = re.findall(r"\d+\.\s+(.*?)(?=\n\d+\.|\Z)", content, re.DOTALL)
            result[title] = [item.strip() for item in items]

    return result

def compare_run_information_units(run_name, chat_bot):
    comparisons = []

    with open(f"evaluations/metrics/{run_name}.json", encoding="utf8") as file:
        simplification_results = json.load(file)

    for simplification_result in simplification_results:
        source = simplification_result[0]["prediction"]
        prediction = simplification_result[-1]["prediction"]

        fact_comparison_string = compare_information_units(chat_bot, source, prediction)
        fact_comparison_categories = convert_fact_string_to_dict(fact_comparison_string)
        dict_entry = {
            "source": source,
            "prediction": prediction,
            "comparison": fact_comparison_categories
        }
        comparisons.append(dict_entry)

    file_io_utils.convert_dict_to_json(f"evaluations/information_comparison/{run_name}.json", comparisons)