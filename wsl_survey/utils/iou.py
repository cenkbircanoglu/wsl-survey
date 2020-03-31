def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = [int(float(i)) for i in boxA]
    boxA[2] = boxA[2] + boxA[0]
    boxA[3] = boxA[3] + boxA[1]

    boxB = [int(float(i)) for i in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


if __name__ == '__main__':
    boxA = [0., 0., 10., 10.]
    boxB = [1., 1., 9., 9.]
    print(bb_intersection_over_union(boxA, boxB))

    boxA = [a / 100. for a in boxA]
    boxB = [b / 100. for b in boxB]
    print(bb_intersection_over_union(boxA, boxB))

    boxA = [0., 0., 10., 10.]
    boxB = [12., 12., 22., 22.]
    print(bb_intersection_over_union(boxA, boxB))

    boxA = [0., 0., 2., 2.]
    boxB = [1., 1., 3., 3.]
    print(bb_intersection_over_union(boxA, boxB))

    print(bb_intersection_over_union((65, 221, 858, 474), (86, 225, 839, 469)))
