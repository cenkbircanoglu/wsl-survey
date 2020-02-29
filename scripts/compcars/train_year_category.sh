export PYTHONPATH='.'

export NETWORK=resnet18

python3 wsl_survey/compcars/experiments/classification/category_classifier.py \
    --image_size 224 \
    --num_workers 32 \
    --batch_size 128 \
    --epochs 25 \
    --dataset_dir ./data/compcars/arxiv_data/train_test_split/classification \
    --image_dir ./data/compcars/ \
    --network_name $NETWORK \
    --category_name year \
    --model_file models/year/$NETWORK/model
