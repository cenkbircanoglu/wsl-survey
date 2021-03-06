import os

from wsl_survey.segmentation.irn.misc import pyutils
from wsl_survey.segmentation.irn_compcars.config import make_parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.cam_weights_name), exist_ok=True)
    os.makedirs(os.path.dirname(args.irn_weights_name), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_name), exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.train_cam_pass:
        import step.train_cam

        print('train_cam')
        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)

    if args.make_cam_pass:
        import step.make_cam

        print('make_cam')
        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)

    if args.cam_to_ir_label_pass:
        import step.cam_to_ir_label

        print('cam_to_ir_label')
        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn_pass:
        import step.train_irn

        print('train_irn')
        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)

    if args.make_ins_seg_pass:
        import step.make_ins_seg_labels

        print('make_ins_seg_pass')
        timer = pyutils.Timer('step.make_ins_seg_labels:')
        step.make_ins_seg_labels.run(args)

    if args.make_sem_seg_pass:
        import step.make_sem_seg_labels

        print('make_sem_seg_labels')
        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)
