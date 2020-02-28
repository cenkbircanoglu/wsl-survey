import os


def create_html(img_filename_list_path, img_folders, output_file):
    image_names = ['2007_000333', '2007_000392', '2007_000515',
                   '2007_000528', '2007_000549', '2007_000648',
                   '2007_000720', '2007_000768', '2007_000793',
                   '2007_000836', '2007_000876', '2007_000904',
                   '2007_001185', '2007_001225', '2007_001340',
                   '2007_001397', '2007_001420', '2007_001595',
                   '2007_001724', '2007_001825', '2007_001857',
                   '2007_001960', '2007_002024', '2007_002055',
                   '2007_002216', '2007_002273', '2007_002281',
                   '2007_002370']
    image_filenames = []
    with open(img_filename_list_path, mode='r') as f:
        for i in f.readlines():
            if i.strip() in image_names:
                image_filenames.append(i)

    img_folders = img_folders.split(',')
    base_html = '''<head><style>
        .row {
          display: flex;
          flex-wrap: wrap;
          padding: 0 4px;
        }
        .column {
          flex: 50%;
          padding: 0 4px;

        }
        .inline-block {
           display: inline-block;
        }
        .column img {
          margin-top: 4px;
          vertical-align: middle;
          margin-right: 1px;
        }

        @media print {
          /* top-level divs with ids */
          body > div[10] {
            page-break-before: always;
          }
        }
        </style> </head><div class="row">'''
    html = base_html
    row_number_per_image = 50
    for i, filename in enumerate(image_filenames):
        html += '<div id=%s class="column"><p align="center">%s</p>' % (
            i, filename.strip())
        for img_folder in img_folders:
            width = str(100. / (len(img_folders) + 0.5)) + "%"
            if os.path.exists(
                os.path.join(img_folder.strip(), filename.strip()) + '.png'):
                html += '<img src="%s.png" alt="%s" title="%s" width="%s;"/>' % (
                    os.path.join(img_folder.strip(), filename.strip()),
                    filename.strip(), filename.strip(), width)
            else:
                html += '<img src="https://dummyimage.com/100/00ff48/ff0000.png&text=Image+Not+Found" alt="%s" title="%s" width="%s;"/>' % (
                    filename.strip(), filename.strip(), width)
        html += '</div>'
        if (i + 1) % row_number_per_image == 0:
            page_num = i // row_number_per_image
            html += '<a href="%s_%s.html" target="_self">Link to page %s</a>' % (
                output_file, page_num, page_num)
            html += '</div>'
            with open("%s_%s.html" % (output_file, page_num), mode='w') as f:
                f.write(html)
            html = base_html
    html += '</div>'
    with open("%s_%s.html" % (output_file, i // row_number_per_image),
              mode='w') as f:
        f.write(html)


if __name__ == '__main__':
    create_html(
        '/Users/cenk.bircanoglu/wsl/wsl_survey/data/voc12/train.txt',
        """/Users/cenk.bircanoglu/wsl/wsl_survey/results/original/SegmentationObject/
                ,/Users/cenk.bircanoglu/wsl/wsl_survey/results/original/SegmentationClass/
                ,/Users/cenk.bircanoglu/wsl/wsl_survey/results/subset1_resnet152/subset1_resnet152/irn_label
                ,/Users/cenk.bircanoglu/wsl/wsl_survey/results/subset2_resnet152/subset2_resnet152/irn_label
                ,/Users/cenk.bircanoglu/wsl/wsl_survey/results/subset3_resnet152/subset3_resnet152/irn_label
                ,/Users/cenk.bircanoglu/wsl/wsl_survey/results/subset4_resnet152/subset4_resnet152/irn_label
                ,/Users/cenk.bircanoglu/wsl/wsl_survey/results/subset5_resnet152/subset5_resnet152/irn_label
                ,/Users/cenk.bircanoglu/wsl/wsl_survey/results/subset6_resnet152/subset6_resnet152/irn_label
                ,/Users/cenk.bircanoglu/wsl/wsl_survey/results/subset7_resnet152/subset7_resnet152/irn_label
                ,/Users/cenk.bircanoglu/wsl/wsl_survey/results/subset8_resnet152/subset8_resnet152/irn_label
                ,/Users/cenk.bircanoglu/wsl/wsl_survey/results/subset9_resnet152/subset9_resnet152/irn_label
                ,/Users/cenk.bircanoglu/wsl/wsl_survey/results/subset10_resnet152/subset10_resnet152/irn_label
                """,
        '/Users/cenk.bircanoglu/wsl/wsl_survey/wsl_survey/utils/htmls/irn_label'
    )
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--img_filename_list_path",
    #                     type=str,
    #                     help='Path of the file that contains image filename list')
    # parser.add_argument("--img_folders",
    #                     type=str,
    #                     help='Comma seperated folder paths to load the data from')
    # parser.add_argument("--output_file",
    #                     type=str)
    # args = parser.parse_args()
    #
    # create_html(args.img_filename_list_path, args.img_folders, args.output_file)
