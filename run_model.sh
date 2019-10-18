#!/bin/bash
ARCHIVE_FILE="https://ropes-dataset.s3-us-west-2.amazonaws.com/roberta_baseline_model/roberta_ropes_model.tar.gz"
python predict.py --archive_file $ARCHIVE_FILE --input_file /ropes/nolabels.json  --output_file /results/predictions.json

