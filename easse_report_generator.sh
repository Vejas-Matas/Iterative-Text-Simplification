#!/bin/bash

DATA_TYPE="test"

for PASSAGE_TYPE in "sentence" "abstract"
do
    echo "Printing ${PASSAGE_TYPE}s..."

    for FILE_PATH in ./predictions/*type=${PASSAGE_TYPE}_set=${DATA_TYPE}*
    # for FILE_PATH in ./predictions/*

    do
        if [[ -e "$FILE_PATH" ]]; then
	    FILE_NAME=$(basename "${FILE_PATH}")
	    echo "$FILE_NAME"
            easse report -p "./graphs/reports/${FILE_NAME}.html" -t custom --refs_sents_paths "dataset/simpletext_lines/${PASSAGE_TYPE}_${DATA_TYPE}_references.txt" --orig_sents_path "dataset/simpletext_lines/${PASSAGE_TYPE}_${DATA_TYPE}_sources.txt" --sys_sents_path "${FILE_PATH}"
        fi
    done
done
