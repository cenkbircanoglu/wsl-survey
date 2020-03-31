export PYTHONPATH='.'

export NETWORK=resnet152
export CATEGORY=year

python3 wsl_survey/compcars/experiments/classification/category_classifier.py \
    --image_size 224 \
    --finetune True \
    --num_workers 8 \
    --batch_size 128 \
    --epochs 25 \
    --dataset_dir ./data/compcars/arxiv_data/train_test_split/classification \
    --image_dir ./data/compcars/ \
    --network_name $NETWORK \
    --category_name $CATEGORY \
    --model_file models/$CATEGORY/$NETWORK/finetune/model
