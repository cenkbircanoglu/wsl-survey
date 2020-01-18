import os


def read_txt(path):
    with open(path, mode='r') as f:
        lines = [i.strip() for i in f.readlines()]
    print(lines)
    return lines


def main():
    train_aug = read_txt(
        './data/test2/VOC2012/ImageSets/Segmentation/train_aug.txt')
    train = read_txt(
        './data/test2/VOC2012/ImageSets/Segmentation/train.txt')
    val = read_txt('./data/test2/VOC2012/ImageSets/Segmentation/val.txt')
    test = read_txt('./data/test2/VOC2012/ImageSets/Segmentation/test.txt')
    print(len(train_aug))
    train_aug.extend(train)
    train_aug.extend(val)
    train_aug.extend(test)
    print(len(train_aug))
    folder = './data/test2/VOC2012/JPEGImages'
    for file in os.listdir(folder):
        if not file.split('.')[0] in train_aug:
            os.remove(os.path.join(folder, file))
    folder = './data/test2/VOC2012/SegmentationClass'
    for file in os.listdir(folder):
        if not file.split('.')[0] in train_aug:
            os.remove(os.path.join(folder, file))
    folder = './data/test2/VOC2012/SegmentationObject'
    for file in os.listdir(folder):
        if not file.split('.')[0] in train_aug:
            os.remove(os.path.join(folder, file))


if __name__ == '__main__':
    main()
