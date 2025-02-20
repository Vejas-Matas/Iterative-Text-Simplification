import json

# Refers to old results structure
def extract_predictions_from_json(path):
    with open(path, encoding="utf8") as file:
        contents = json.load(file)
        return [outputs["prediction"] for outputs in contents["iterative"][1]]

def convert_list_to_txt(path, lines_list):
    with open(path, "w", encoding="utf8") as file:
        for line in lines_list:
            # .replace("\n", "\\n") preserves one entry-one line structure
            preserved_line = line.replace("\n", "\\n")
            file.write(f"{preserved_line}\n")

def append_to_txt(path, line):
    with open(path, "a", encoding="utf8") as file:
        # .replace("\n", "\\n") preserves one entry-one line structure
        preserved_line = line.replace("\n", "\\n")
        file.write(f"{preserved_line}\n")

def convert_dict_to_json(path, dictionary):
    with open(path, "w", encoding="utf8") as file:
        json.dump(dictionary, file, indent=2)