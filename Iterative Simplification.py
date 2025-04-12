# Libraries
import datetime

# My own files
import information_units
import file_io_utils
import dataset_utils
import chat_bots
import parameters
import plotting

# def simplify_passage_iteratively(chat_bot, system_prompt, algorithm_parameters, passage, max_iter=20):
def simplify_passage_iteratively(chat_bot, algorithm_parameters, max_iter=20):
    for _ in range(max_iter):
        # chat_bot.send_prompt("Identify which parts of the passage are the most complex, then the complexity level of the passage. Limit your answer to a maximum of 5 sentences")
        # chat_bot.send_prompt(f'Is determined complexity higher than DC ({algorithm_parameters["DC"]})?  Answer "Yes" or "No"')
        
        # # Add prompting to convert DC to DC (FKGL), alternatively, add it to algorithm_parameters
        chat_bot.send_prompt("Identify which parts of the passage are the most complex, then the complexity level of the passage. Limit your answer to a maximum of 5 sentences")
        chat_bot.send_prompt(f'FKGL score of the latest passage version is {chat_bot.get_latest_fkgl()}. Is determined complexity higher than DC ({algorithm_parameters["DC"]})?  Answer "Yes" or "No"')
        if "NO" in chat_bot.get_last_response().upper()[-5:]:
            break
        chat_bot.send_prompt(f'Identify a single complicated section of the passage. Remember to respect the ILT ({algorithm_parameters["ILT"]}) contraint. Only provide the identified section')
        chat_bot.send_prompt("Simplify this section. Only provide the proposed simplification")
        chat_bot.send_prompt("Reincorporate the simplified section into the passage")
        chat_bot.send_prompt("Identify information loss and its severity in the updated passage compared to the original. Comparison must be between the originally provided (the very first) passage and the current simplified version. Limit your answer to a maximum of 5 sentences")
        chat_bot.send_prompt(f'What is the highest severity level identified in your last answer? Is it higher than ILT ({algorithm_parameters["ILT"]})? Provide the highest severity level, followed by an answer to the ILT question as "Yes" or "No"')
        if "NO" in chat_bot.get_last_response().upper()[-5:]:
            chat_bot.send_limited_context_prompt("Accept the proposed simplification. Print the updated version of the passage, do not print anything else")
            # chat_bot.send_prompt("If needed, adjust the passage to maintain readabily and flow of text")
        else:
            chat_bot.send_prompt("Revert the last proposed change. In further iterations you may still attempt to simplify this section in other ways. Print the reverted passage, do not print anything else")
            

        # Check if the model ran out of tokens, if so, ignore this iteration and terminate
        if chat_bot.get_last_response() == "":
            return
        
        chat_bot.add_iteration_results()

def simplify_passage_iteratively_unaware(chat_bot, algorithm_parameters, max_iter=20):
    for _ in range(max_iter):
        # chat_bot.send_prompt("Identify which parts of the passage are the most complex, then the complexity level of the passage. Limit your answer to a maximum of 5 sentences")
        # chat_bot.send_prompt(f'Is determined complexity higher than DC ({algorithm_parameters["DC"]})?  Answer "Yes" or "No"')
        
        # # Add prompting to convert DC to DC (FKGL), alternatively, add it to algorithm_parameters
        latest_passage_version = chat_bot.get_iteration_results()[-1]["prediction"]
        chat_bot.add_system_prompt("The latest version of the simplified passage is provided in the following prompt")
        chat_bot.add_system_prompt(latest_passage_version)
        chat_bot.send_limited_context_prompt("Identify which parts of the simplified passage are the most complex, then the complexity level of the passage. Limit your answer to a maximum of 5 sentences", 2)
        chat_bot.send_limited_context_prompt(f'FKGL score of the latest passage version is {chat_bot.get_latest_fkgl()}. Is determined complexity higher than DC ({algorithm_parameters["DC"]})?  Answer "Yes" or "No"', 4)
        # chat_bot.send_limited_context_prompt(f'Is determined complexity higher than DC ({algorithm_parameters["DC"]})?  Answer "Yes" or "No"', 4)
        if "NO" in chat_bot.get_last_response().upper()[-5:]:
            break
        chat_bot.send_limited_context_prompt(f'Identify a single complicated section of the simplified passage. Remember to respect the ILT ({algorithm_parameters["ILT"]}) contraint. Only provide the identified section', 6)
        chat_bot.send_limited_context_prompt("Simplify this section. Only provide the proposed simplification", 8)
        chat_bot.send_limited_context_prompt("Reincorporate the simplified section into the simplified passage", 10)
        chat_bot.send_limited_context_prompt("Identify information loss and its severity in the updated simplified passage compared to the original passage. Comparison must be between the originally provided (the very first) passage and the current simplified version with the proposed change. Limit your answer to a maximum of 5 sentences", 12)
        chat_bot.send_limited_context_prompt(f'What is the highest severity level identified in your last answer? Is it higher than ILT ({algorithm_parameters["ILT"]})? Provide the highest severity level, followed by an answer to the ILT question as "Yes" or "No"', 14)
        if "NO" in chat_bot.get_last_response().upper()[-5:]:
            chat_bot.send_limited_context_prompt("Accept the proposed simplification. Print the updated version of the simplified passage, do not print anything else", 16)
            # chat_bot.send_limited_context_prompt("If needed, adjust the passage to maintain readabily and flow of text", 18)
        else:
            chat_bot.send_limited_context_prompt("Revert the last proposed change. Print the simplified passage with the proposed change reverted, do not print anything else", 16)
            

        # Check if the model ran out of tokens, if so, ignore this iteration and terminate
        if chat_bot.get_last_response() == "":
            return
        
        chat_bot.add_iteration_results()

# def simplify_passage_iteratively_condensed(chat_bot, system_prompt, algorithm_parameters, passage, max_iter=20):
def simplify_passage_iteratively_condensed(chat_bot, algorithm_parameters, max_iter=20):
    for _ in range(max_iter):
        chat_bot.send_limited_context_prompt(f'FKGL score of the latest simplified passage version is {chat_bot.get_latest_fkgl()}. Identify which parts of the simplified passage are the most complex. Based on this information, identify the complexity of the simplified passage. Is determined complexity higher than DC ({algorithm_parameters["DC"]})? Answer "Yes" or "No"', 3)
        if "NO" in chat_bot.get_last_response().upper()[-5:]:
            break
        chat_bot.send_limited_context_prompt(f'Identify a single complicated section of the simplified passage. Simplify this section, and remember to respect the ILT ({algorithm_parameters["ILT"]}) contraint. Reincorporate the simplified section into the simplified passage. Only provide the reincorporated version', 5)
        chat_bot.send_limited_context_prompt(f'Identify information loss and its severity in the updated simplified passage compared to the original passage. Comparison must be between the originally provided (the very first) passage and the current simplified version with the proposed change. Limit your answer to a maximum of 5 sentences. What is the highest severity level identified in your last answer? Is it higher than ILT ({algorithm_parameters["ILT"]})? Provide the highest severity level, followed by an answer to the ILT question as "Yes" or "No"', 7)
        if "NO" in chat_bot.get_last_response().upper()[-5:]:
            chat_bot.send_limited_context_prompt("Accept the proposed simplification. Print the updated version of the simplified passage, do not print anything else", 9)
        #     chat_bot.send_limited_context_prompt("If needed, adjust the passage to maintain readabily and flow of text")
        else:
            chat_bot.send_limited_context_prompt("Revert the last proposed change. Print the simplified passage with the proposed change reverted, do not print anything else", 9)

        # Check if the model ran out of tokens, if so, ignore this iteration and terminate
        if chat_bot.get_last_response() == "":
            return
        
        chat_bot.add_iteration_results()

def simplify_passage_non_iteratively(chat_bot, algorithm_parameters, max_iter=0):
    chat_bot.send_prompt("Simplify the provided passage. Print the final version of the simplified passage, include only the text of the passage with no comments or additional punctuation, and do not provide the original passage")



def simplify_passages(algorithm_name, algorithm_fn, system_prompt, algorithm_parameters, passage_type, max_iter, n=None):
    ### Setup
    if passage_type == "abstract":
        sources, references = dataset_utils.get_sources_and_references("abs", n)
    elif passage_type == "sentence":
        sources, references = dataset_utils.get_sources_and_references("snt", n)
    else:
        raise ValueError('Passage type should be "abstract" or "sentence"')

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    parameter_string = ("dc=" + algorithm_parameters["DC"] + "_" + "ilt=" + algorithm_parameters["ILT"]).lower().replace(" ", "_")
    run_name = f"timestamp={timestamp}_algorithm={algorithm_name}_type={passage_type}_{parameter_string}_i={max_iter}_n={n}" # WILL NEED TRAIN / TSET DISTINCTION, AND IN EVALUATION FILE AS WELL

    results = []
    
    ### Simplifying passages one-by-one
    for i in range(len(sources)):
        ### Initialise
        chat_bot.clear()
        chat_bot.add_system_prompt(system_prompt)

        if algorithm_parameters is not None and algorithm_parameters != {}:
            algorithm_string = "\n".join(f"{parameter}: {value}" for parameter, value in algorithm_parameters.items())
            chat_bot.add_system_prompt(f"Algorithm parameters are as follows.\n{algorithm_string}\n\nThe original passage is provided in the following message")
        else:
            chat_bot.add_system_prompt(f"The original passage is provided in the following message")

        source = sources[i]
        chat_bot.sources.append(source)
        chat_bot.references.append(references[i])

        chat_bot.add_system_prompt(source)
        chat_bot.add_iteration_results()
        
        ### Simplify
        try:
            algorithm_fn(chat_bot, algorithm_parameters, max_iter)
        except ValueError:
            # Model ran out tokens, so just use the last iteration
            pass

        ### Collect intermediary results
        iteration_results = chat_bot.get_iteration_results()
        prediction = iteration_results[-1]["prediction"]
        results.append(iteration_results)

        ### Save / display results
        # chat_bot.print_chat()
        # chat_bot.save_chat()
        # chat_bot.print_token_usage_log()

        file_io_utils.append_to_txt(f"predictions/{run_name}", prediction)

    file_io_utils.convert_dict_to_json(f"evaluations/metrics/{run_name}.json", results)
    latest_run_names.append(run_name)

    return results



chat_bot = chat_bots.VllmChatBot(
    model_name=parameters.vllm_model,
)

passages_to_simplify = 10
passage_type_to_simplify = "sentence"
latest_run_names = []

# simplify_passages("iterative", simplify_passage_iteratively, parameters.system_prompt, parameters.algorithm_parameters, passage_type_to_simplify, 20, passages_to_simplify)
simplify_passages("unaware_iterative", simplify_passage_iteratively_unaware, parameters.system_prompt, parameters.algorithm_parameters, passage_type_to_simplify, 20, passages_to_simplify)
simplify_passages("condensed_iterative", simplify_passage_iteratively_condensed, parameters.system_prompt, parameters.algorithm_parameters, passage_type_to_simplify, 20, passages_to_simplify)
# simplify_passages("non_iterative", simplify_passage_non_iteratively, parameters.non_iterative_system_prompt, parameters.algorithm_parameters, passage_type_to_simplify, 0, passages_to_simplify)

for passage_type_to_simplify in ["sentence", "abstract"]:
    # simplify_passages("iterative", simplify_passage_iteratively, parameters.system_prompt, parameters.algorithm_parameters, passage_type_to_simplify, 20, passages_to_simplify)
    # simplify_passages("unaware_iterative", simplify_passage_iteratively_unaware, parameters.system_prompt, parameters.algorithm_parameters, passage_type_to_simplify, 20, passages_to_simplify)
    # simplify_passages("condensed_iterative", simplify_passage_iteratively_condensed, parameters.system_prompt, parameters.algorithm_parameters, passage_type_to_simplify, 20, passages_to_simplify)
    # simplify_passages("non_iterative", simplify_passage_non_iteratively, parameters.non_iterative_system_prompt, parameters.algorithm_parameters, passage_type_to_simplify, 0, passages_to_simplify)
    pass

print("––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")
print("––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")
print("––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")
print("–––––––––––––––––SIMPLIFICATION COMPLETE––––––––––––––––––––")
print("––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")
print("––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")
print("––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")
print("––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")



for run_name in latest_run_names:
    information_units.compare_run_information_units(run_name, chat_bot)

file_io_utils.convert_dict_to_json("./latest_run_names.json", latest_run_names)