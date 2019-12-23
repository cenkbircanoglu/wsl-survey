cd weights/
sh download_weights.sh

cd data/
sh get_coco_dataset.sh

python3 test.py --weights_path weights/yolov3.weights

python3 detect.py --image_folder data/samples/

python3 train.py --data_config config/coco.data  --pretrained_weights weights/darknet53.conv.74


cd config/
sh create_custom_model.sh <num-classes>


python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
--pretrained_weights weights/darknet53.conv.74
