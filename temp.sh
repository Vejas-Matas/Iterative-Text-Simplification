#!/bin/bash

F1="algorithm=condensed_iterative_type=sentence_dc=average_person_ilt=average_person_i=20_n=None_timestamp=2025-02-21_03-18-38.160475"
F2="algorithm=iterative_type=sentence_dc=average_person_ilt=average_person_i=20_n=None_timestamp=2025-02-21_04-51-25.252243"
F3="algorithm=non_iterative_type=sentence_dc=average_person_ilt=average_person_i=0_n=None_timestamp=2025-02-21_07-24-03.190053"

COMMAND="evaluate"
PASSAGE_TYPE="sentence"
PASSAGE_COUNT_SUFFIX=""

# FILES=$(ls "./predictions" -t | head -n ${FILE_COUNT})
FILES=("${F1}" "${F2}" "${F3}")

# Execute the command on each file path
for FILE in $FILES; do
    echo "FILE: ${FILE}"
    easse ${COMMAND} -q -t custom --refs_sents_paths "dataset/simpletext_lines/${PASSAGE_TYPE}_train_references${PASSAGE_COUNT_SUFFIX}.txt" --orig_sents_path "dataset/simpletext_lines/${PASSAGE_TYPE}_train_sources${PASSAGE_COUNT_SUFFIX}.txt" --sys_sents_path "./predictions/${FILE}"
done


# easse evaluate -q -t custom --refs_sents_paths "dataset/simpletext_lines/sentence_train_references_50.txt" --orig_sents_path "dataset/simpletext_lines/sentence_train_sources_50.txt" --sys_sents_path "predictions/${PREDICTION_FILE_NAME}"
