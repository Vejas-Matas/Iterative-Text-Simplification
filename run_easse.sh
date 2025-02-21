#!/bin/bash

PREDICTION_FILE_NAME="type=sentence_dc=university_student_ilt=university_student_i=20_timestamp=2025-02-14_08-13-22.551088"

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <evaluate/report> <number of files to read>"
    exit 1
fi

# Assign arguments to variables
COMMAND="$1"
FILE_COUNT="$2"

# Validate that the second argument is an integer
if ! [[ ${FILE_COUNT} =~ ^[0-9]+$ ]]; then
    echo "Error: The second parameter must be an integer."
    exit 1
fi

# Get the list of n most recent file paths in the directory
FILES=$(ls "./predictions" -t | head -n ${FILE_COUNT})

# Execute the command on each file path
for FILE in $FILES; do
    echo "FILE: ${FILE}"
    easse ${COMMAND} -q -t custom --refs_sents_paths "dataset/simpletext_lines/sentence_train_references_50.txt" --orig_sents_path "dataset/simpletext_lines/sentence_train_sources_50.txt" --sys_sents_path "./predictions/${FILE}"
done


# easse evaluate -q -t custom --refs_sents_paths "dataset/simpletext_lines/sentence_train_references_50.txt" --orig_sents_path "dataset/simpletext_lines/sentence_train_sources_50.txt" --sys_sents_path "predictions/${PREDICTION_FILE_NAME}"
