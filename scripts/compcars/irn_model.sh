export PYTHONPATH='.'

export MODEL=resnet50
export NETWORK=ResNet50
export OUTPUT_FOLDER=./outputs/compcars/model_id/results/$MODEL/

python3 wsl_survey/segmentation/irn_compcars/main.py \
    --voc12_root=./data/compcars/data/image/ \
    --category_name=model_id \
    --train_list=./data/compcars/arxiv_data/train_test_split/classification_train.txt \
    --val_list=./data/compcars/arxiv_data/train_test_split/classification_test.txt \
    --infer_list=./data/compcars/arxiv_data/train_test_split/classification_train.txt \
    --cam_weights_name=$OUTPUT_FOLDER/sess/cam.pth \
    --irn_weights_name=$OUTPUT_FOLDER/sess/irn.pth \
    --cam_out_dir=$OUTPUT_FOLDER/cam \
    --sem_seg_out_dir=$OUTPUT_FOLDER/sem_seg \
    --ins_seg_out_dir=$OUTPUT_FOLDER/ins_seg \
    --ir_label_out_dir=$OUTPUT_FOLDER/irn_label \
    --cam_network=$NETWORK \
    --irn_network=$NETWORK \
    --log_name=$OUTPUT_FOLDER/logs \
    --train_cam_pass=True \
    --make_cam_pass=True \
    --cam_to_ir_label_pass=True \
    --train_irn_pass=True \
    --make_sem_seg_pass=True \
    --num_workers=64 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16



python3 wsl_survey/segmentation/irn_compcars/main.py \
    --voc12_root=./data/compcars/data/image/ \
    --category_name=model_id \
    --train_list=./data/compcars/arxiv_data/train_test_split/classification_test.txt \
    --val_list=./data/compcars/arxiv_data/train_test_split/classification_test.txt \
    --infer_list=./data/compcars/arxiv_data/train_test_split/classification_test.txt \
    --cam_weights_name=$OUTPUT_FOLDER/sess/cam.pth \
    --irn_weights_name=$OUTPUT_FOLDER/sess/irn.pth \
    --cam_out_dir=$OUTPUT_FOLDER/cam \
    --sem_seg_out_dir=$OUTPUT_FOLDER/sem_seg \
    --ins_seg_out_dir=$OUTPUT_FOLDER/ins_seg \
    --ir_label_out_dir=$OUTPUT_FOLDER/irn_label \
    --cam_network=$NETWORK \
    --irn_network=$NETWORK \
    --log_name=$OUTPUT_FOLDER/logs \
    --make_cam_pass=True \
    --make_sem_seg_pass=True \
    --num_workers=64 \
    --cam_network_module=wsl_survey.segmentation.irn.net.resnet_cam \
    --irn_network_module=wsl_survey.segmentation.irn.net.resnet_irn \
    --cam_batch_size=16 \
    --irn_batch_size=16
