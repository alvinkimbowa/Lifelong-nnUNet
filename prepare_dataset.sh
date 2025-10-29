#!/bin/bash

# 1) Activate conda env and export nnU-Net environment variables
eval "$(conda shell.bash hook)"
conda activate nnunet_mednca

export nnUNet_raw_data_base="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export RESULTS_FOLDER="data/nnUNet_results"
export EVALUATION_FOLDER="data/nnUNet_eval"
export PARAM_SEARCH_FOLDER="data/nnUNet_param_search"

echo "Environment variables:"
echo "  nnUNet_raw_data_base=$nnUNet_raw_data_base"
echo "  nnUNet_preprocessed=$nnUNet_preprocessed"
echo "  RESULTS_FOLDER=$RESULTS_FOLDER"
echo "  EVALUATION_FOLDER=$EVALUATION_FOLDER"
echo "  PARAM_SEARCH_FOLDER=$PARAM_SEARCH_FOLDER"

TASK_ID="Task078_KneeUS_OtherDevices"
SPLIT="Tr"
DATASET_DIR="$nnUNet_raw_data_base/nnUNet_raw_data/$TASK_ID"

# 2) Convert PNG -> NIfTI
python convert_png_to_nii.py --task_name "$TASK_ID" --split "$SPLIT"

DATASET_JSON="$DATASET_DIR/dataset.json"
if [ ! -f "$DATASET_JSON" ]; then
    echo "ERROR: dataset.json not found at $DATASET_JSON" >&2
    exit 1
fi

# 4) Run nnUNet preprocessing
# Extract dataset id from TASK_ID (assumes TASK_ID like "Task078_...")
DATASET_ID=$(echo $TASK_ID | grep -oP '(?<=Task)\d+')
echo "DATASET_ID: $DATASET_ID"
nnUNet_plan_and_preprocess -t $DATASET_ID -pl3d None

# 5) Print success message
echo "Done. Preprocessed data at: $nnUNet_preprocessed/$TASK_ID"


