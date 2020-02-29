def check_network_input_size(image_size, version):
    if version == 'alexnet':
        assert image_size == 224

    if version.startswith('densenet'):
        assert image_size == 224

    if version.startswith('mnasnet'):
        assert image_size == 224

    if version.startswith('mobilenet'):
        assert image_size == 224

    if version.startswith('resne'):
        assert image_size == 224

    if version.startswith('shufflenet'):
        assert image_size == 224

    if version.startswith('vgg'):
        assert image_size == 224

    if version.startswith('wide_resnet'):
        assert image_size == 224
