#!/bin/bash

DATA_TYPE="test"

for PASSAGE_TYPE in "sentence" "abstract"
do
    echo "Printing ${PASSAGE_TYPE}s..."

    for FILE in ./predictions/*type=${PASSAGE_TYPE}_set=${DATA_TYPE}*
    # for FILE in ./predictions/*

    do
        if [[ -e "$FILE" ]]; then
            echo "$FILE"
            easse report -p "${FILE}.html" -t custom --refs_sents_paths "dataset/simpletext_lines/${PASSAGE_TYPE}_${DATA_TYPE}_references.txt" --orig_sents_path "dataset/simpletext_lines/${PASSAGE_TYPE}_${DATA_TYPE}_sources.txt" --sys_sents_path "${FILE}"
        fi
    done
done
