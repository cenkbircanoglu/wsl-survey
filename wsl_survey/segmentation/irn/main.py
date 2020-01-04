import argparse
import os

from wsl_survey.segmentation.irn.misc import pyutils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--voc12_root", required=True, type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--class_label_dict_path", required=True, type=str)
    parser.add_argument("--train_list", required=True, type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", required=True, type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", required=True, type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", required=True, type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", required=True, type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", required=True, type=str)
    parser.add_argument("--irn_weights_name", required=True, type=str)
    parser.add_argument("--cam_out_dir", required=True, type=str)
    parser.add_argument("--ir_label_out_dir", required=True, type=str)
    parser.add_argument("--sem_seg_out_dir", required=True, type=str)
    parser.add_argument("--ins_seg_out_dir", required=True, type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=False, type=bool)
    parser.add_argument("--make_cam_pass", default=False, type=bool)
    parser.add_argument("--eval_cam_pass", default=False, type=bool)
    parser.add_argument("--cam_to_ir_label_pass", default=False, type=bool)
    parser.add_argument("--train_irn_pass", default=False, type=bool)
    parser.add_argument("--make_ins_seg_pass", default=False, type=bool)
    parser.add_argument("--eval_ins_seg_pass", default=False, type=bool)
    parser.add_argument("--make_sem_seg_pass", default=False, type=bool)
    parser.add_argument("--eval_sem_seg_pass", default=False, type=bool)

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

    if args.eval_cam_pass:
        import step.eval_cam

        print('eval_cam')
        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)

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

        print('eval_ins_seg_pass')
        timer = pyutils.Timer('step.make_ins_seg_labels:')
        step.make_ins_seg_labels.run(args)

    if args.eval_ins_seg_pass:
        import step.eval_ins_seg

        print('eval_ins_seg')
        timer = pyutils.Timer('step.eval_ins_seg:')
        step.eval_ins_seg.run(args)

    if args.make_sem_seg_pass:
        import step.make_sem_seg_labels

        print('make_sem_seg_labels')
        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass:
        import step.eval_sem_seg

        print('eval_sem_seg')
        timer = pyutils.Timer('step.eval_sem_seg:')
        step.eval_sem_seg.run(args)
