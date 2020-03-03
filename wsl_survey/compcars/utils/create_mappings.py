import argparse

from wsl_survey.compcars.utils.parse_path import parse_path


def create_mapping(path):
    make_id_set = set()
    model_id_set = set()
    year_set = set()

    with open(path + '_train.txt', mode='r') as f:
        for line in f.readlines():
            make_id, model_id, year, image_name, image_path = parse_path(line)
            make_id_set.add(make_id)
            model_id_set.add(model_id)
            year_set.add(year)
    with open(path + '_test.txt', mode='r') as f:
        for line in f.readlines():
            make_id, model_id, year, image_name, image_path = parse_path(line)
            make_id_set.add(make_id)
            model_id_set.add(model_id)
            year_set.add(year)

    with open(path + '_make_mapping.txt', mode='w') as f:
        for i, uid in enumerate(make_id_set):
            f.write("%s, %s\n" % (str(i), str(uid)))

    with open(path + '_model_mapping.txt', mode='w') as f:
        for i, uid in enumerate(model_id_set):
            f.write("%s, %s\n" % (str(i), str(uid)))

    with open(path + '_year_mapping.txt', mode='w') as f:
        for i, uid in enumerate(year_set):
            f.write("%s, %s\n" % (str(i), str(uid)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    create_mapping(args.path)
