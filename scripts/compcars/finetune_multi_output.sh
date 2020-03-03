export PYTHONPATH='.'

export NETWORK=resnet152

python3 wsl_survey/compcars/experiments/classification/multi_output_classifier.py \
    --image_size 224 \
    --finetune True \
    --num_workers 8 \
    --batch_size 128 \
    --epochs 25 \
    --dataset_dir ./data/compcars/arxiv_data/train_test_split/classification \
    --image_dir ./data/compcars/ \
    --network_name $NETWORK \
    --model_file models/multi_output/$NETWORK/finetune/model
