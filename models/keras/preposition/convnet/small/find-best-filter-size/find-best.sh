#!/bin/bash

# Get a unique list of the feature names.
for input in $(for file in */model.log; do echo $(dirname $file) | sed 's,-[0-9]$,,'; done | sort | uniq)
do
    # For each feature, find the one filter width that yielded the lowest
    # validation loss.
    for file in ${input}*/model.log
    do
        echo $input $(dirname $file) $(grep val_acc $file | cat -n | sort -n -r -k17 | tail -1)
    done | sort -n -r -k17 | tail -1
done | sort -n -r -k 17
