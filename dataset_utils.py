import json
import sklearn.model_selection

import file_io_utils
import parameters

def read_dataset():
    dataset = {}

    for data_type in ["train", "test"]:
        dataset[data_type] = {}
        for passage_length in ["abs", "snt"]:
            dataset[data_type][passage_length] = {}

    # for data_type in ["train", "test"]:
    #     for passage_length in ["abs", "snt"]:
    #         for passage_type in ["source", "reference"]:
    #             if data_type == "test" and passage_type == "reference":
    #                 continue
                    
    #             folder = f"./dataset/{data_type}/"
    #             file_name = f"simpletext_task3_2024_{data_type}_{passage_length}_{passage_type}.json"
    #             path = folder + file_name

    #             # print("Reading: " + path)
    #             with open(path, "r", encoding="utf8") as file:
    #                 data = json.load(file)
                
    #             dataset[data_type][passage_length][passage_type] = data

    for passage_length in ["abs", "snt"]:
        original_train_data = {}
        data_type = "train"

        for passage_type in ["source", "reference"]:
            folder = f"./dataset/{data_type}/"
            file_name = f"simpletext_task3_2024_{data_type}_{passage_length}_{passage_type}.json"
            path = folder + file_name

            with open(path, "r", encoding="utf8") as file:
                data = json.load(file)

            original_train_data[passage_type] = data
            
        sources_train, sources_test, references_train, references_test = sklearn.model_selection.train_test_split(
            original_train_data["source"],
            original_train_data["reference"],
            test_size=parameters.test_set_proportion,
            random_state=parameters.splitting_seed,
        )

        dataset["train"][passage_length]["source"] = sources_train
        dataset["test"][passage_length]["source"] = sources_test
        dataset["train"][passage_length]["reference"] = references_train
        dataset["test"][passage_length]["reference"] = references_test

    return dataset


def get_sources_and_references(passage_type_abbreviation, data_type, n):
    dataset = read_dataset()

    sources = [entry[f"source_{passage_type_abbreviation}"] for entry in dataset[data_type][passage_type_abbreviation]["source"][:n] ]
    references = [entry[f"simplified_{passage_type_abbreviation}"] for entry in dataset[data_type][passage_type_abbreviation]["reference"][:n] ]

    return (sources, references)

def create_dataset_line_files():
    snt_train = get_sources_and_references("snt", "train", None)
    abs_train = get_sources_and_references("abs", "train", None)
    snt_test = get_sources_and_references("snt", "test", None)
    abs_test = get_sources_and_references("abs", "test", None)

    file_io_utils.convert_list_to_txt("./dataset/simpletext_lines/sentence_train_sources.txt", snt_train[0])
    file_io_utils.convert_list_to_txt("./dataset/simpletext_lines/sentence_train_references.txt", snt_train[1])
    file_io_utils.convert_list_to_txt("./dataset/simpletext_lines/abstract_train_sources.txt", abs_train[0])
    file_io_utils.convert_list_to_txt("./dataset/simpletext_lines/abstract_train_references.txt", abs_train[1])

    file_io_utils.convert_list_to_txt("./dataset/simpletext_lines/sentence_test_sources.txt", snt_test[0])
    file_io_utils.convert_list_to_txt("./dataset/simpletext_lines/sentence_test_references.txt", snt_test[1])
    file_io_utils.convert_list_to_txt("./dataset/simpletext_lines/abstract_test_sources.txt", abs_test[0])
    file_io_utils.convert_list_to_txt("./dataset/simpletext_lines/abstract_test_references.txt", abs_test[1])