export PYTHONPATH='.'

mkdir networks/orj/

python3 wsl_survey/utils/network_summary.py \
    --network_name=ResNet152 \
    --network_module=wsl_survey.segmentation.irn.net.resnet_cam > networks/orj/CAMResNet152.txt

python3 wsl_survey/utils/network_summary.py \
    --network_name=ResNet152 \
    --network_module=wsl_survey.segmentation.irn.net.resnet_cam > networks/orj/CAMResNet152.txt

python3 wsl_survey/utils/network_summary.py \
    --network_name=ResNet152CAM \
    --network_module=wsl_survey.segmentation.irn.net.resnet_cam > networks/orj/CAMResNet152CAM.txt

python3 wsl_survey/utils/network_summary.py \
    --network_name=ResNet152 \
    --network_module=wsl_survey.segmentation.irn.net.resnet_irn > networks/orj/IRNResNet152.txt

python3 wsl_survey/utils/network_summary.py \
    --network_name=ResNet152AffinityDisplacementLoss \
    --network_module=wsl_survey.segmentation.irn.net.resnet_irn > networks/orj/IRNResNet152AffinityDisplacementLoss.txt

python3 wsl_survey/utils/network_summary.py \
    --network_name=ResNet152EdgeDisplacement \
    --network_module=wsl_survey.segmentation.irn.net.resnet_irn > networks/orj/IRNResNet152EdgeDisplacement.txt


#### DISTILLED

mkdir networks/distilled/
python3 wsl_survey/utils/network_summary.py \
    --network_name=ResNet152 \
    --network_module=wsl_survey.segmentation.irn.net.distilled.resnet_cam > networks/distilled/CAMResNet152.txt

python3 wsl_survey/utils/network_summary.py \
    --network_name=ResNet152 \
    --network_module=wsl_survey.segmentation.irn.net.distilled.resnet_cam > networks/distilled/CAMResNet152.txt

python3 wsl_survey/utils/network_summary.py \
    --network_name=ResNet152CAM \
    --network_module=wsl_survey.segmentation.irn.net.distilled.resnet_cam > networks/distilled/CAMResNet152CAM.txt

python3 wsl_survey/utils/network_summary.py \
    --network_name=ResNet152 \
    --network_module=wsl_survey.segmentation.irn.net.distilled.resnet_irn > networks/distilled/IRNResNet152.txt

python3 wsl_survey/utils/network_summary.py \
    --network_name=ResNet152AffinityDisplacementLoss \
    --network_module=wsl_survey.segmentation.irn.net.distilled.resnet_irn > networks/distilled/IRNResNet152AffinityDisplacementLoss.txt

python3 wsl_survey/utils/network_summary.py \
    --network_name=ResNet152EdgeDisplacement \
    --network_module=wsl_survey.segmentation.irn.net.distilled.resnet_irn > networks/distilled/IRNResNet152EdgeDisplacement.txt

python3 networks/clear_networks.py
