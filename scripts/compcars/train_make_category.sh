

python3 wsl_survey/compcars/experiments/classification/category_classifier.py \
    --image_size 224 \
    --num_workers 32 \
    --batch_size 128 \
    --epochs 1 \
    --dataset_dir /Users/cenk.bircanoglu/workspace/icpr/arxiv_data/train_test_split/classification \
    --image_dir /Users/cenk.bircanoglu/workspace/icpr/ \
    --network_name resnet18 \
    --category_name make_id \
    --model_file models/make_id/resnet18/model
