#!/bin/bash

F1="timestamp=2025-02-28_03-10-47.231854_algorithm=iterative_type=sentence_dc=average_person_ilt=average_person_i=20_n=None"
F2="timestamp=2025-02-28_05-11-53.791848_algorithm=condensed_iterative_type=sentence_dc=average_person_ilt=average_person_i=20_n=None"
F3="timestamp=2025-02-28_06-37-00.168099_algorithm=non_iterative_type=sentence_dc=average_person_ilt=average_person_i=0_n=None"

COMMAND="report"
PASSAGE_TYPE="sentence"
PASSAGE_COUNT_SUFFIX=""

# FILES=$(ls "./predictions" -t | head -n ${FILE_COUNT})
FILES=("${F1}" "${F2}" "${F3}")

# Execute the command on each file path
# for FILE in $FILES; do
#     echo "FILE: ${FILE}"
#     easse ${COMMAND} -q -t custom --refs_sents_paths "dataset/simpletext_lines/${PASSAGE_TYPE}_train_references${PASSAGE_COUNT_SUFFIX}.txt" --orig_sents_path "dataset/simpletext_lines/${PASSAGE_TYPE}_train_sources${PASSAGE_COUNT_SUFFIX}.txt" --sys_sents_path "./predictions/${FILE}"
# done

echo ${F1}
easse ${COMMAND} -p "graphs/reports/${F1}" -t custom --refs_sents_paths "dataset/simpletext_lines/${PASSAGE_TYPE}_train_references${PASSAGE_COUNT_SUFFIX}.txt" --orig_sents_path "dataset/simpletext_lines/${PASSAGE_TYPE}_train_sources${PASSAGE_COUNT_SUFFIX}.txt" --sys_sents_path "./predictions/${F1}"
echo ${F2}
easse ${COMMAND} -p "graphs/reports/${F2}" -t custom --refs_sents_paths "dataset/simpletext_lines/${PASSAGE_TYPE}_train_references${PASSAGE_COUNT_SUFFIX}.txt" --orig_sents_path "dataset/simpletext_lines/${PASSAGE_TYPE}_train_sources${PASSAGE_COUNT_SUFFIX}.txt" --sys_sents_path "./predictions/${F2}"
echo ${F3}
easse ${COMMAND} -p "graphs/reports/${F3}" -t custom --refs_sents_paths "dataset/simpletext_lines/${PASSAGE_TYPE}_train_references${PASSAGE_COUNT_SUFFIX}.txt" --orig_sents_path "dataset/simpletext_lines/${PASSAGE_TYPE}_train_sources${PASSAGE_COUNT_SUFFIX}.txt" --sys_sents_path "./predictions/${F3}"



F1="timestamp=2025-02-28_06-39-46.295374_algorithm=iterative_type=abstract_dc=average_person_ilt=average_person_i=20_n=None"
F2="timestamp=2025-02-28_07-48-53.983910_algorithm=condensed_iterative_type=abstract_dc=average_person_ilt=average_person_i=20_n=None"
F3="timestamp=2025-02-28_08-16-14.631471_algorithm=non_iterative_type=abstract_dc=average_person_ilt=average_person_i=0_n=None"

PASSAGE_TYPE="abstract"

# FILES=$(ls "./predictions" -t | head -n ${FILE_COUNT})
FILES=("${F1}" "${F2}" "${F3}")

# Execute the command on each file path
echo ${F1}
easse ${COMMAND} -p "graphs/reports/${F1}" -t custom --refs_sents_paths "dataset/simpletext_lines/${PASSAGE_TYPE}_train_references${PASSAGE_COUNT_SUFFIX}.txt" --orig_sents_path "dataset/simpletext_lines/${PASSAGE_TYPE}_train_sources${PASSAGE_COUNT_SUFFIX}.txt" --sys_sents_path "./predictions/${F1}"
echo ${F2}
easse ${COMMAND} -p "graphs/reports/${F2}" -t custom --refs_sents_paths "dataset/simpletext_lines/${PASSAGE_TYPE}_train_references${PASSAGE_COUNT_SUFFIX}.txt" --orig_sents_path "dataset/simpletext_lines/${PASSAGE_TYPE}_train_sources${PASSAGE_COUNT_SUFFIX}.txt" --sys_sents_path "./predictions/${F2}"
echo ${F3}
easse ${COMMAND} -p "graphs/reports/${F3}" -t custom --refs_sents_paths "dataset/simpletext_lines/${PASSAGE_TYPE}_train_references${PASSAGE_COUNT_SUFFIX}.txt" --orig_sents_path "dataset/simpletext_lines/${PASSAGE_TYPE}_train_sources${PASSAGE_COUNT_SUFFIX}.txt" --sys_sents_path "./predictions/${F3}"


# easse evaluate -q -t custom --refs_sents_paths "dataset/simpletext_lines/sentence_train_references_50.txt" --orig_sents_path "dataset/simpletext_lines/sentence_train_sources_50.txt" --sys_sents_path "predictions/${PREDICTION_FILE_NAME}"
