#!/bin/bash

DATA_TYPE="test"

for PASSAGE_TYPE in "sentence" "abstract"
do
    # Loop over files in current directory that contain 'type=<type>' in the filename
    for FILE in ./predictions/*type=${type}*
    do
        if [[ -e "$FILE" ]]; then
            echo "$FILE"
            # easse report -p "graphs/reports/${FILE}.html" -t custom --refs_sents_paths "dataset/simpletext_lines/${PASSAGE_TYPE}_${DATA_TYPE}_references.txt" --orig_sents_path "dataset/simpletext_lines/${PASSAGE_TYPE}_${DATA_TYPE}_sources.txt" --sys_sents_path "./predictions/${FILE}"
        fi
    done
done
