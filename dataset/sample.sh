#!/bin/bash

train_input="translation2019zh_train.json"
valid_input="translation2019zh_valid.json"
train_output="translation2019zh_train5k.json"
valid_output="translation2019zh_valid1k.json"

head -n 5000 "$train_input" > "$train_output"
head -n 1000 "$valid_input" > "$valid_output"

echo "Saved 5000 lines to $train_output, and 1000 lines to $valid_output"