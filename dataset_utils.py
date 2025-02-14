import json
import file_io_utils

def read_dataset():
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

    return dataset


def get_sources_and_references(passage_type_abbreviation, n):
    dataset = read_dataset()

    sources = [entry[f"source_{passage_type_abbreviation}"] for entry in dataset["train"][passage_type_abbreviation]["source"][:n] ]
    references = [entry[f"simplified_{passage_type_abbreviation}"] for entry in dataset["train"][passage_type_abbreviation]["reference"][:n] ]

    return (sources, references)

def create_dataset_line_files():
    snts = get_sources_and_references("snt", None)
    abss = get_sources_and_references("abs", None)

    file_io_utils.convert_list_to_txt("./dataset/simpletext_lines/sentence_train_sources.txt", snts[0])
    file_io_utils.convert_list_to_txt("./dataset/simpletext_lines/sentence_train_references.txt", snts[1])
    file_io_utils.convert_list_to_txt("./dataset/simpletext_lines/abstract_train_sources.txt", abss[0])
    file_io_utils.convert_list_to_txt("./dataset/simpletext_lines/abstract_train_references.txt", abss[1])

# create_dataset_line_files()