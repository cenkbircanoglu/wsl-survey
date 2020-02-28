lines = []

with open('./networks/orj/CAMResNet152.txt', mode='r') as f:
    for i, line in enumerate(f.readlines()):
        if i % 3 == 0:
            lines.append(line)

with open('./networks/orj/CAMResNet152_small.txt', mode='w') as f:
    f.writelines(lines)

lines = []
with open('./networks/orj/CAMResNet152CAM.txt', mode='r') as f:
    for i, line in enumerate(f.readlines()):
        if i % 3 == 0:
            lines.append(line)

with open('./networks/orj/CAMResNet152CAM_small.txt', mode='w') as f:
    f.writelines(lines)

lines = []
with open('./networks/orj/IRNResNet152.txt', mode='r') as f:
    for i, line in enumerate(f.readlines()):
        if i % 3 == 0:
            lines.append(line)

with open('./networks/orj/IRNResNet152_small.txt', mode='w') as f:
    f.writelines(lines)

lines = []
with open('./networks/orj/IRNResNet152EdgeDisplacement.txt', mode='r') as f:
    for i, line in enumerate(f.readlines()):
        if i % 3 == 0:
            lines.append(line)

with open('./networks/orj/IRNResNet152EdgeDisplacement_small.txt', mode='w') as f:
    f.writelines(lines)



lines = []

with open('./networks/distilled/CAMResNet152.txt', mode='r') as f:
    for i, line in enumerate(f.readlines()):
        if i % 3 == 0:
            lines.append(line)

with open('./networks/distilled/CAMResNet152_small.txt', mode='w') as f:
    f.writelines(lines)

lines = []
with open('./networks/distilled/CAMResNet152CAM.txt', mode='r') as f:
    for i, line in enumerate(f.readlines()):
        if i % 3 == 0:
            lines.append(line)

with open('./networks/distilled/CAMResNet152CAM_small.txt', mode='w') as f:
    f.writelines(lines)

lines = []
with open('./networks/distilled/IRNResNet152.txt', mode='r') as f:
    for i, line in enumerate(f.readlines()):
        if i % 3 == 0:
            lines.append(line)

with open('./networks/distilled/IRNResNet152_small.txt', mode='w') as f:
    f.writelines(lines)

lines = []
with open('./networks/distilled/IRNResNet152EdgeDisplacement.txt', mode='r') as f:
    for i, line in enumerate(f.readlines()):
        if i % 3 == 0:
            lines.append(line)

with open('./networks/distilled/IRNResNet152EdgeDisplacement_small.txt', mode='w') as f:
    f.writelines(lines)

