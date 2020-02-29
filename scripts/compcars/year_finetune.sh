

python3 wsl_survey/compcars/experiments/classification/category_finetuning.py \
    --image_size 224 \
    --num_workers 32 \
    --batch_size 128 \
    --epochs 1 \
    --dataset_dir /Users/cenk.bircanoglu/workspace/icpr/arxiv_data/train_test_split/classification \
    --image_dir /Users/cenk.bircanoglu/workspace/icpr/ \
    --network_name resnet18 \
    --category_name year \
    --model_file models/year/resnet18/model
