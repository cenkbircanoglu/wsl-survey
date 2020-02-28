from PIL import Image
import numpy as np
aa = Image.open('/Users/cenk.bircanoglu/wsl/wsl_survey/outputs/voc12/results/resnet152/irn_label/2007_000032.png')
bb = np.array(aa)
print(bb)
aa.imshow()
