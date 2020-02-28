import numpy as np
from pycocotools.coco import COCO

coco = COCO(
    '/Users/cenk.bircanoglu/wsl/wsl_survey/datasets/coco2014/annotations/instances_train2014.json'
)
img_ids = coco.getImgIds()
for img_id in img_ids:
    img = coco.loadImgs(img_id)[0]
    anns_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    anns_img = np.zeros((img['height'], img['width']))
    coco.showAnns(anns)
    for ann in anns:
        anns_img = np.maximum(anns_img,
                              coco.annToMask(ann) * ann['category_id'])
