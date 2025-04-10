import glob
import re
import json
import numpy as np
import pandas as pd

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

    # # VERSION 1: NO PAST MEMORY
    # fact_comparison_prompts = [
    #     {"role": "system", "content": "You take lists of information, and treat them as mathematical sets. Then you provide three lists: elements present only in the first set (\"LOST: \"), only in the second one (\"ADDED: \"), and in the intesection (\"KEPT: \")"},
    #     {"role": "user",   "content": f"First list:\n{source_facts}"},
    #     {"role": "user",   "content": f"Second list:\n{prediction_facts}"},
    # ]

    # VERSION 2: PAST MEMORY
    fact_comparison_prompts = [
        {"role": "system",      "content": information_comparison_instructions},
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

    # # VERSION 3: DIRECT COMPARISON
    # fact_comparison_prompts = [
    #     {"role": "system",      "content": """You take two passages – an original text and its simplified version. Then you extract atomic knowledge units from both, and compare them as mathematical sets. Provide three lists: "ADDED" (present only in the first list), "KEPT" (present in both, in the set intersection), and "DELETED" (present only in the second list)."""},
    #     {"role": "user",        "content": source},
    #     {"role": "user",        "content": prediction},
    #     {"role": "user",        "content": "Analyse the data and provide the three lists: ADDED, KEPT, DELETED"},
    # ]
    # source_facts = ""
    # prediction_facts = ""

    fact_comparison = chat_bot.send_no_context_prompts(fact_comparison_prompts)



    # file_io_utils.append_to_txt(fact_comparison_path, 100*"#")
    # file_io_utils.append_to_txt(fact_comparison_path, 100*"–")
    # file_io_utils.append_to_txt(fact_comparison_path, source)
    # file_io_utils.append_to_txt(fact_comparison_path, "")
    # file_io_utils.append_to_txt(fact_comparison_path, source_facts)
    # file_io_utils.append_to_txt(fact_comparison_path, 100*"–")
    # file_io_utils.append_to_txt(fact_comparison_path, prediction)
    # file_io_utils.append_to_txt(fact_comparison_path, "")
    # file_io_utils.append_to_txt(fact_comparison_path, prediction_facts)
    # file_io_utils.append_to_txt(fact_comparison_path, 100*"/")
    # file_io_utils.append_to_txt(fact_comparison_path, fact_comparison)

    print(100*"#")
    print(100*"–")
    print(source)
    # print()
    # print(source_facts)
    print(100*"–")
    print(prediction)
    # print()
    # print(prediction_facts)
    # print(100*"/")
    # print(fact_comparison)

    return fac_comparison


chat_bot = chat_bots.VllmChatBot(
    model_name=parameters.vllm_model,
)

s = "In the modern era of automation and robotics, autonomous vehicles are currently the focus of academic and industrial research."
p = "In the modern technology, autonomous vehicles are currently the focus of academic and industrial research."

print(compare_information_units(chat_bot, s, p))