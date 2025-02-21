# Libraries
import datetime

# My own files
import file_io_utils
import dataset_utils
import chat_bots
import parameters
import plotting

# Rewrite to class?
dataset = dataset_utils.read_dataset()


def simplify_passage_iteratively(chat_bot, system_prompt, algorithm_parameters, passage, max_iter=20):
    chat_bot.clear()
    
    chat_bot.add_system_prompt(system_prompt)
    chat_bot.add_system_prompt(f"The passage:\n{passage}")

    if algorithm_parameters is not None and algorithm_parameters != {}:
        chat_bot.add_system_prompt("\n".join(f"{parameter}: {value}" for parameter, value in algorithm_parameters.items()))

    for _ in range(max_iter):
        chat_bot.send_prompt("Identify which parts of the text are the most complex, then the complexity level of the passage. Limit your answer to a maximum of 5 sentences")
        chat_bot.send_prompt(f'Is determined complexity higher than DC ({parameters.algorithm_parameters["DC"]})? Answer "Yes" or "No"')
        if "NO" in chat_bot.get_last_response().upper():
            break
        chat_bot.send_prompt(f'Identify a single complicated section of the passage. Remember to respect the ILT ({parameters.algorithm_parameters["ILT"]}) contraint. Only provide the identified section')
        chat_bot.send_prompt("Simplify this section. Only provide the proposed simplification")
        chat_bot.send_prompt("Reincorporate the simplified section into the passage")
        chat_bot.send_prompt("Identify information loss and its severity in the updated passage compared to the original. Comparison must be between the originally provided (the very first) passage and the current simplified version. Limit your answer to a maximum of 5 sentences")
        chat_bot.send_prompt(f'What is the highest severity level identified in your last answer? Is it higher than ILT ({parameters.algorithm_parameters["ILT"]})? Provide the highest severity level, followed by an answer to the ILT question as "Yes" or "No"')
        if "YES" in chat_bot.get_last_response().upper():
            chat_bot.send_prompt("Revert the last proposed change. In further iterations you may still attempt to simplify this section in other ways")
        # else:
        #     chat_bot.send_prompt("If needed, adjust the passage to maintain readabily and flow of text")

    chat_bot.send_prompt("Print the final version of the simplified passage, include only the text of the passage with no comments or additional punctuation, and do not provide the original passage")
    # chat_bot.print_chat()
    chat_bot.save_chat()
    # chat_bot.print_token_usage_log()

    return chat_bot.get_last_response()

def simplify_passage_iteratively_condensed(chat_bot, system_prompt, algorithm_parameters, passage, max_iter=20):
    chat_bot.clear()
    
    chat_bot.add_system_prompt(system_prompt)
    chat_bot.add_system_prompt(f"The passage:\n{passage}")

    if algorithm_parameters is not None and algorithm_parameters != {}:
        chat_bot.add_system_prompt("\n".join(f"{parameter}: {value}" for parameter, value in algorithm_parameters.items()))

    for _ in range(max_iter):
        chat_bot.send_prompt(f'Identify which parts of the text are the most complex, then the complexity. Is determined complexity higher than DC ({parameters.algorithm_parameters["DC"]})? Answer "Yes" or "No"', 1)
        if "NO" in chat_bot.get_last_response().upper():
            break
        chat_bot.send_prompt(f'Identify a single complicated section of the passage. Remember to respect the ILT ({parameters.algorithm_parameters["ILT"]}) contraint. Simplify this section. Reincorporate the simplified section into the passage. Only provide the reincorporated version', 2)
        chat_bot.send_prompt(f'Identify information loss and its severity in the updated passage compared to the original. Comparison must be between the originally provided (the very first) passage and the current simplified version. Limit your answer to a maximum of 5 sentences. What is the highest severity level identified in your last answer? Is it higher than ILT ({parameters.algorithm_parameters["ILT"]})? Provide the highest severity level, followed by an answer to the ILT question as "Yes" or "No"', 3)
        if "YES" in chat_bot.get_last_response().upper():
            chat_bot.send_prompt("Revert the last proposed change. In further iterations you may still attempt to simplify this section in other ways. Print the reverted passage.", 4)
        else:
            chat_bot.send_prompt("Accept the proposed simplification. Print the updated version of the passage", 4)
        #     chat_bot.send_prompt("If needed, adjust the passage to maintain readabily and flow of text")

    chat_bot.send_prompt("Print the final version of the simplified passage, include only the text of the passage with no comments or additional punctuation, and do not provide the original passage", 1)
    # chat_bot.print_chat()
    # chat_bot.save_chat()
    # chat_bot.print_token_usage_log()

    return chat_bot.get_last_response()



def simplify_passages(algorithm_name, algorithm_fn, system_prompt, algorithm_parameters, passage_type, max_iter, n=None):
    if passage_type == "abstract":
        sources, references = dataset_utils.get_sources_and_references("abs", n)
    elif passage_type == "sentence":
        sources, references = dataset_utils.get_sources_and_references("snt", n)
    else:
        raise ValueError('Passage type should be "abstract" or "sentence"')

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    parameter_string = ("dc=" + algorithm_parameters["DC"] + "_" + "ilt=" + algorithm_parameters["DC"]).lower().replace(" ", "_")
    results_file_name = f"algorithm={algorithm_name}_type={passage_type}_{parameter_string}_i={max_iter}_n={n}_timestamp={timestamp}"

    predictions = []
    results = []
    token_usage = []

    # chat_bot = chat_bots.OpenAIChatBot(
    #     model=parameters.openai_model,
    #     api_key=parameters.openai_api_key,
    # )
    chat_bot = chat_bots.VllmChatBot(
        model_name=parameters.vllm_model,
    )
    
    for i in range(len(sources)):

        prediction = algorithm_fn(chat_bot, system_prompt, algorithm_parameters, sources[i], max_iter)
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
    # file_io_utils.convert_dict_to_json(f"dummy_metrics.json", overall_metrics)

    return (results, overall_metrics)


passages_to_simplify = 10
passage_type_to_simplify = "sentence"

simplify_passages("condensed_iterative", simplify_passage_iteratively_condensed, parameters.system_prompt, parameters.algorithm_parameters, passage_type_to_simplify, 20, passages_to_simplify)
# simplify_passages("iterative", simplify_passage_iteratively, parameters.system_prompt, parameters.algorithm_parameters, passage_type_to_simplify, 20, passages_to_simplify)
# simplify_passages("non_iterative", simplify_passage_iteratively, parameters.non_iterative_system_prompt, parameters.algorithm_parameters, passage_type_to_simplify, 0, passages_to_simplify)

# plotting.make_token_usage_graphs(datetime.timedelta(minutes=30))
