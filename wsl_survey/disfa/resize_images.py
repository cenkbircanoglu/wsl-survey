import os
from multiprocessing.pool import Pool

from PIL import Image


def resize_images(path):
    try:
        if '.jpg' in path:
            output_path = path.replace('disfa', 'resized_disfa')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            basewidth = 450
            img = Image.open(path)
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)
            img.save(output_path)
    except Exception as e:
        print(e)


p = Pool(8)

for (root, dirs, files) in os.walk('./data/disfa', topdown=True):
    p.map(resize_images, [os.path.join(root, file_path) for file_path in files])
p.close()
p.join()
