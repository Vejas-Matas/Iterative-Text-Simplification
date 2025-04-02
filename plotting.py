import matplotlib.pyplot as plt
import numpy as np
import datetime
import json
import glob
import re
import textwrap

def make_token_usage_graphs(timedelta=datetime.timedelta(minutes=1)):
    token_usage_data = {}
    metrics_files = get_recent_metrics_files(timedelta)

    for file_name in metrics_files:
        # re_match = re.search(r"algorithm=(\w+)_type=", file_name)
        # re_match = re.search(r"([\w\-=_.]+)_metrics.json", file_name)
        # algorithm_name = re_match.group(1)

        with open(file_name, encoding="utf8") as file:
            contents = json.load(file)
            token_usage_data[file_name] = contents
            # token_usage_data[algorithm_name] = contents # This overwrites things if there are multiple files with the same algorithm name

    algorithms, token_counts = restructure_token_data(token_usage_data)
    graph_token_usage(algorithms, token_counts)

def get_recent_metrics_files(timedelta=datetime.timedelta(minutes=1), full_dataset_only=False):
    timestamp_from = datetime.datetime.now() - timedelta
    files = glob.glob("./evaluations/*.json")
    files_to_keep = []

    for file_name in files:
        # re_match = re.search(r"timestamp=([\d\-_.]+)_metrics.json", file_name)
        re_match = re.search(r"timestamp=([\d\-_.]+)_algorithm", file_name)
        timestamp_string = re_match.group(1)
        timestamp = datetime.datetime.strptime(timestamp_string, "%Y-%m-%d_%H-%M-%S.%f")

        if timestamp < timestamp_from:
            continue

        if full_dataset_only:
            re_match = re.search(r"_n=None", file_name)
            if re_match is None:
                continue

        files_to_keep.append(file_name)

    return files_to_keep

def restructure_token_data(data):
    algorithms = data.keys()
    in_counts = []
    out_counts = []

    for algorithm in algorithms:
        in_counts.append(data[algorithm]["in_tokens"])
        out_counts.append(data[algorithm]["out_tokens"])

    token_counts = {
        "Prompt (in) tokens": np.array(in_counts),
        "Generated (out) tokens": np.array(out_counts),
    }

    return (algorithms, token_counts)

def graph_token_usage(algorithms, token_counts, bar_width=0.5):
    fig, ax = plt.subplots(figsize=(20,12))
    bar_bottom = np.zeros(len(algorithms))
    wrapped_x_labels = [textwrap.fill(label, width=10) for label in algorithms] # Instead, I could extract common parameters and move them to the title, shortening the column names

    for type_label, token_count in token_counts.items():
        ax.bar(wrapped_x_labels, token_count, bar_width, label=type_label, bottom=bar_bottom)
        # ax.bar(algorithms, token_count, bar_width, label=type_label, bottom=bar_bottom)
        bar_bottom += token_count

    ax.set_title("Token usage by algorithm")
    ax.legend(loc="upper right")
    ax.tick_params("x", rotation=80)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    plt.savefig(f"graphs/token_usage_{timestamp}.png")

# make_token_usage_graphs(datetime.timedelta(days=30))