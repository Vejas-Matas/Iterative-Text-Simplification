#!/bin/bash

PREDICTION_FILE_NAME="type=sentence_dc=university_student_ilt=university_student_i=20_timestamp=2025-02-14_08-13-22.551088"

easse evaluate -q -t custom --refs_sents_paths "dataset/simpletext_lines/sentence_train_references_50.txt" --orig_sents_path "dataset/simpletext_lines/sentence_train_sources_50.txt" --sys_sents_path "predictions/${PREDICTION_FILE_NAME}"