openai_api_key = ""
openai_model = "gpt-4o-mini"
# vllm_model = "meta-llama/Llama-3.1-8B-Instruct"
vllm_model = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
# vllm_model = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

algorithm_parameters = {
    "DC": "Average person",
    "ILT": "Medium-High"
}

### ALGORITHM INSTRUCTION PROMPTS
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

### ATOMIC INFORMATION UNIT OPERATION PROMPT AND EXAMPLES
information_comparison_instructions = """You are a diligigent and attentive text evaluator. You take two lists – extracted information from an original text and its simplified version. Then you compare them for information discrepancies. Then you provide four lists:
1. PRESERVATIONS: information in the original and the simplification are the same, equivallent (e.g. synonyms) or it can be clearly implied to be the same;
2. OVERSIMPLIFICATIONS: core meaning is preserved, but some information was lost – e.g., vagueness is introduced, multiple meanings can be implied or original meaning is lost;
3. DELETIONS: information that was completely lost after simplification;
4. HALLUCINATIONS: information that was added after simplification, but is not present and could not be implied from the original.

Either of the lists is allowed to be empty. 
"""

# Information extraction and comparison examples are generated using ChatGPT and then refined
information_extraction_example_1 = """
Original passage:
"Mitochondria, often referred to as the powerhouse of the cell, generate adenosine triphosphate (ATP) through oxidative phosphorylation, a process that involves the electron transport chain and chemiosmosis to drive ATP synthesis."

Extracted atomic information units:
1. Mitochondria are called the powerhouse of the cell.
2. Mitochondria generate ATP.
3. ATP stands for adenosine triphosphate.
4. The process used is oxidative phosphorylation.
5. Oxidative phosphorylation involves the electron transport chain.
6. Oxidative phosphorylation involves chemiosmosis.
7. Chemiosmosis helps drive ATP synthesis.

Simplified passage:
"Mitochondria make energy for the cell using oxygen in a process called oxidative phosphorylation."

Extracted atomic information units:
1. Mitochondria make energy for the cell.
2. The process is called oxidative phosphorylation.
3. Oxygen is used in this process.
"""

information_comparison_example_1 = """
PRESERVATIONS:
1. Mitochondria generate energy ("ATP" is replaced with "energy" in the simplified version, but meaning is clear in the context of cells).
2. The process is called oxidative phosphorylation.
3. Oxygen is used in the process (implied in the complex version via "oxidative phosphorylation" but explicitly stated in the simplified version).

OVERSIMPLIFICATIONS:
0. None

DELETIONS:
1. Mitochondria are called the powerhouse of the cell.
2. ATP stands for adenosine triphosphate.
3. Oxidative phosphorylation involves the electron transport chain.
4. Oxidative phosphorylation involves chemiosmosis.
5. Chemiosmosis helps drive ATP synthesis.

HALLUCINATIONS:
0. None
"""

information_extraction_example_2 = """
Original passage:
"A neural network is a computational model inspired by the structure of biological neural networks, consisting of layers of interconnected artificial neurons that process input data using weighted connections and activation functions to learn patterns and make predictions."

Extracted atomic information units:
1. A neural network is a computational model.
2. A neural network is inspired by biological neural networks.
3. It consists of layers of interconnected artificial neurons.
4. It processes input data.
5. It uses weighted connections to process data.
6. It uses activation functions to process data.
7. The purpose of a neural network is to learn patterns.
8. A neural network is used to make predictions.

Simplified passage:
"A neural network is a computer program that learns patterns from data to make predictions and create images."

Extracted atomic information units:
1. A neural network is a computer program.
2. A neural network learns patterns.
3. A neural network processes data.
4. A neural network is used to make predictions.
5. A neural network is used to create images.
"""

information_comparison_example_2 = """
PRESERVATIONS:
1. A neural network learns patterns.
2. A neural network processes data.
3. A neural network is used to make predictions.

OVERSIMPLIFICATIONS:
1. A neural network is a computer program ("A computational model" changed to "a computer program", which is partially true but omits non-software implementations).

DELETIONS:
1. A neural network is inspired by biological neural networks (link to biology).
2. It consists of layers of interconnected artificial neurons (structure).
3. It uses weighted connections to process data (learning mechanism).
4. It uses activation functions to process data (learning mechanism).

HALLUCINATIONS:
1. A neural network is used to create images (statement is true, but is not mentioned or implied in the original).
"""
