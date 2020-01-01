import argparse
import os

from wsl_survey.irn import pyutils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--dataset_dir',
                        metavar='DIR',
                        help='path to dataset (e.g. ../data/')
    parser.add_argument('--image_dir',
                        metavar='DIR',
                        help='path to dataset (e.g. ../data/')
    parser.add_argument('--voc_root',
                        metavar='DIR',
                        help='path to dataset (e.g. ../data/')
    parser.add_argument("--checkpoints", type=str)
    parser.add_argument("--num_workers", default=os.cpu_count() // 2, type=int)
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--batch_size", default=16, type=int)

    # Class Activation Map
    parser.add_argument("--cam_network",
                        default="wsl_survey.net.resnet50_cam",
                        type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=100, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales",
                        default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network",
                        default="wsl_survey.net.resnet50_irn",
                        type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=16, type=int)
    parser.add_argument("--irn_num_epoches", default=100, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument(
        "--exp_times",
        default=8,
        help=
        "Hyper-parameter that controls the number of random walk iterations,"
        "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=True)
    parser.add_argument("--make_cam_pass", default=True)
    parser.add_argument("--eval_cam_pass", default=True)
    parser.add_argument("--cam_to_ir_label_pass", default=True)
    parser.add_argument("--train_irn_pass", default=True)
    parser.add_argument("--make_ins_seg_pass", default=True)
    parser.add_argument("--eval_ins_seg_pass", default=True)
    parser.add_argument("--make_sem_seg_pass", default=True)
    parser.add_argument("--eval_sem_seg_pass", default=True)

    args = parser.parse_args()
    cam_weights_name = os.path.join(args.checkpoints, 'sess', 'res50_cam.pth')
    irn_weights_name = os.path.join(args.checkpoints, 'sess', 'res50_irn.pth')
    cam_out_dir = os.path.join(args.checkpoints, 'cam')
    ir_label_out_dir = os.path.join(args.checkpoints, 'ir_label')
    sem_seg_out_dir = os.path.join(args.checkpoints, 'sem_seg')
    ins_seg_out_dir = os.path.join(args.checkpoints, 'ins_seg')
    os.makedirs(os.path.dirname(cam_weights_name), exist_ok=True)
    os.makedirs(cam_out_dir, exist_ok=True)
    os.makedirs(ir_label_out_dir, exist_ok=True)
    os.makedirs(sem_seg_out_dir, exist_ok=True)
    os.makedirs(ins_seg_out_dir, exist_ok=True)

    pyutils.Logger(os.path.join(args.checkpoints, 'sample_train_eval.log'))
    print(vars(args))

    if args.train_cam_pass is True:
        import train_cam

        timer = pyutils.Timer('train_cam:')
        print('train_cam')
        train_cam.run(args, cam_weights_name=cam_weights_name)

    if args.make_cam_pass is True:
        import make_cam

        timer = pyutils.Timer('make_cam:')
        print('make_cam')
        make_cam.run(args,
                     cam_weights_name=cam_weights_name,
                     cam_out_dir=cam_out_dir)

    if args.eval_cam_pass is True:
        import eval_cam

        timer = pyutils.Timer('eval_cam:')
        print('eval_cam')
        eval_cam.run(args, cam_out_dir=cam_out_dir)

    if args.cam_to_ir_label_pass is True:
        import cam_to_ir_label

        timer = pyutils.Timer('cam_to_ir_label:')
        print('cam_to_ir_label')
        cam_to_ir_label.run(args,
                            ir_label_out_dir=ir_label_out_dir,
                            cam_out_dir=cam_out_dir)

    if args.train_irn_pass is True:
        import train_irn

        timer = pyutils.Timer('train_irn:')
        print('train_irn')
        train_irn.run(args,
                      irn_weights_name=irn_weights_name,
                      ir_label_out_dir=ir_label_out_dir)

    if args.make_ins_seg_pass is True:
        import make_ins_seg_labels

        timer = pyutils.Timer('make_ins_seg_labels:')
        print('make_ins_seg_labels')
        make_ins_seg_labels.run(args,
                                irn_weights_name=irn_weights_name,
                                ins_seg_out_dir=ins_seg_out_dir,
                                cam_out_dir=cam_out_dir)

    if args.eval_ins_seg_pass is True:
        import eval_ins_seg

        timer = pyutils.Timer('eval_ins_seg:')
        print('eval_ins_seg')
        eval_ins_seg.run(args, ins_seg_out_dir=ins_seg_out_dir)

    if args.make_sem_seg_pass is True:
        import make_sem_seg_labels

        timer = pyutils.Timer('make_sem_seg_labels:')
        print('make_sem_seg_labels')
        make_sem_seg_labels.run(args,
                                irn_weights_name=irn_weights_name,
                                sem_seg_out_dir=sem_seg_out_dir,
                                cam_out_dir=cam_out_dir)

    if args.eval_sem_seg_pass is True:
        import eval_sem_seg

        timer = pyutils.Timer('eval_sem_seg:')
        print('eval_sem_seg')
        eval_sem_seg.run(args, sem_seg_out_dir=sem_seg_out_dir)
