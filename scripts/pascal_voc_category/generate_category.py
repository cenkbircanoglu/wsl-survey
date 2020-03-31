import numpy as np

from wsl_survey.segmentation.irn.voc12.dataloader import decode_int_filename

tr_aug_path = './data/voc12/train_aug.txt'
tr_path = './data/voc12/train.txt'
val_path = './data/voc12/val.txt'
test_path = './data/voc12/test.txt'

class_labels_path = './data/voc12/cls_labels.npy'

cls_labels_dict = np.load(class_labels_path, allow_pickle=True).item()

tr_aug, tr, val, test = [], [], [], []

with open(tr_aug_path, mode='r') as f:
    for i in f.readlines():
        tr_aug.append(
            {
                'img_name': decode_int_filename(i.strip()),
                'img_path': i.strip()
            }
        )

with open(tr_path, mode='r') as f:
    for i in f.readlines():
        tr.append(
            {
                'img_name': decode_int_filename(i.strip()),
                'img_path': i.strip()
            }
        )

with open(val_path, mode='r') as f:
    for i in f.readlines():
        val.append(
            {
                'img_name': decode_int_filename(i.strip()),
                'img_path': i.strip()
            }
        )

with open(test_path, mode='r') as f:
    for i in f.readlines():
        test.append(
            {
                'img_name': decode_int_filename(i.strip()),
                'img_path': i.strip()
            }
        )

print(cls_labels_dict)
print(tr_aug)
