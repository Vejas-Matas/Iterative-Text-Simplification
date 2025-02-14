import json

def extract_predictions_from_json(path):
    with open(path, encoding="utf8") as f:
        contents = json.load(f)
        return [outputs["prediction"] for outputs in contents["iterative"][1]]

def convert_list_to_txt(path, lines_list):
    with open(path, "w") as f:
        for line in lines_list:
            # .replace("\n", "\\n") preserves one entry – one line structure 
            f.write(f"{line.replace("\n", "\\n")}\n") 

# preds = extract_predictions_from_json("evaluations/abstracts_university_medium_max20_2024-12-12_21-37-41.106794")[:10]
# convert_list_to_txt("dummy_test_pred.txt", preds)