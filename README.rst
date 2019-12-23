==========
WSL Survey
==========




.. image:: https://pyup.io/repos/github/cenkbircanoglu/wsl-survey/shield.svg
     :target: https://pyup.io/repos/github/cenkbircanoglu/wsl-survey/
     :alt: Updates

.. image:: https://pyup.io/repos/github/cenkbircanoglu/wsl-survey/python-3-shield.svg
     :target: https://pyup.io/repos/github/cenkbircanoglu/wsl-survey/
     :alt: Python 3

.. image:: https://travis-ci.com/cenkbircanoglu/wsl-survey.svg?branch=master
    :target: https://travis-ci.com/cenkbircanoglu/wsl-survey

.. image:: https://coveralls.io/repos/github/cenkbircanoglu/wsl-survey/badge.svg?branch=master
    :target: https://coveralls.io/github/cenkbircanoglu/wsl-survey?branch=master

.. image:: https://img.shields.io/github/forks/cenkbircanoglu/wsl-survey.svg?style=social&label=Fork&maxAge=2592000)
    :target: https://github.com/cenkbircanoglu/wsl-survey/network



Weakly Supervised Learning Experiments



Features
--------

* Wildcat

Datasets
--------

* Pascal VOC2007
* Pascal VOC2012
* COCO 2014
* COCO 2017
* Kitti
* Wider


1. Pascal VOC2007 dataset
    * Download

    .. code-block:: console

        docker-compose -f scripts/datasets/voc2007.yml run download_dataset

    * Extract Label and Annotation CSV Files

    .. code-block:: console

        docker-compose -f scripts/datasets/voc2007.yml run label_dataset
        docker-compose -f scripts/datasets/voc2007.yml run annotate_dataset

    * Train Wildcat on dataset

    .. code-block:: console

        docker-compose -f scripts/wildcat/voc2007.yml run trainer




TODO
----
1. Add Datasets
    nus-wide
    wider
    wish
