import argparse


def get_args():
    parser = argparse.ArgumentParser(description='all argument')
    parser.add_argument('--num_classes', type=int, default=340, help='image num classes')
    parser.add_argument('--loadsize', type=int, default=64, help='image size')
    parser.add_argument('--epochs', type=int, default=100, help='all epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='init lr')
    parser.add_argument('--use_lr_scheduler', type=bool, default=True, help='use lr scheduler')
    parser.add_argument('--dataset_train', type=str, default='./datasets/train', help='train path')
    parser.add_argument('--dataset_val', type=str, default="./datasets/val", help='val path')
    parser.add_argument('--dataset_test', type=str, default="./datasets/test", help='test path')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='ckpt path')
    parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard_dir', help='log path')
    parser.add_argument('--resume', type=bool, default=False, help='continue training')
    parser.add_argument('--resume_ckpt', type=str, default='./checkpoints/model_best.pth', help='choose breakpoint ckpt')
    parser.add_argument('--local-rank', type=int, default=-1, help='local rank')
    parser.add_argument('--use_mix_precision', type=bool, default=False, help='use mix pretrain')
    parser.add_argument('--test_img_path', type=str, default='datasets/test/zigzag/zigzag-4508464694951936.png', help='choose test image')
    parser.add_argument('--test_dir_path', type=str, default='./datasets/test', help='choose test path')
    return parser.parse_args()
