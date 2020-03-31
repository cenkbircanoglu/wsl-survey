import os
from multiprocessing.pool import Pool

import cv2


def capture(i):
    num = '%02d' % i
    video_path = './datasets/DISFA/DISFA/Videos_RightCamera/RightVideoSN0%s_Comp.avi' % num
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 1
    output_folder = './data/disfa/SN0%s/' % num
    os.makedirs(output_folder, exist_ok=True)
    while success:
        cv2.imwrite(os.path.join(output_folder, "%d.jpg" % count), image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success, count, num)
        count += 1


p = Pool(32)
p.map(capture, range(1, 33))

p.close()
p.join()
