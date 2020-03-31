import os
import shutil

import numpy as np

d = np.load('./data/disfa/cls_labels_conf.npy', allow_pickle=True).item()

orj_path = '/Users/cenk.bircanoglu/wsl/wsl_survey/outputs/disfa/conf/results/resnet50/irn_label'
grouped_path = '/Users/cenk.bircanoglu/wsl/wsl_survey/outputs/disfa/grouped/'
groups = {}
for i in d.items():
    for ind in np.argwhere(i[1]):
        try:
            label = ind.item()
            src_path = os.path.join(orj_path, str(i[0]) + '.png')
            dest_path = os.path.join(grouped_path, str(label), str(i[0]).replace('/', '_') + '.png')
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copyfile(src_path, dest_path)
        except:
            pass
