import os


def parse_path(x):
    x = x.strip()
    splitted = x.split('/')
    make_id = int(splitted[0])
    model_id = int(splitted[1])
    try:
        year = int(splitted[2])
    except ValueError:
        year = -1
    image_name = str(splitted[3])
    image_path = os.path.join('data/image', x)
    return make_id, model_id, year, image_name, image_path
