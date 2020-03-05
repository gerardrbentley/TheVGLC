import argparse


def parse_args(get_parser=False):
    parser = argparse.ArgumentParser(
        description='Game Image AutoEncoder Training')

    # Data and Model
    parser.add_argument('-d', '--dataset', default='mm', help='dataset')
    parser.add_argument('-m', '--model', default='unet1024', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--gpus', default='', type=str,
                        help='comma seperated indexes of gpus to use in training')

    # Augmentations
    parser.add_argument(
        "--no-aug",
        dest="no_augmentation",
        help="Don't augment training images",
        action="store_true",
    )
    parser.add_argument(
        "--center-inpaint",
        dest="center_inpaint",
        help="Cut out central region of input image",
        action="store_true",
    )
    parser.add_argument(
        "--noise",
        dest="noise",
        help="Add Gaussian noise to input image (takes precedent over center-inpaint)",
        action="store_true",
    )

    # Hyper Parameters
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('-a', '--accumulations', default=1, type=int,
                        help='number of batches to accumulate per backwards pass. Effectively increase batch-size')
    parser.add_argument('-e', '--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-lr', '--lr', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('-mo', '--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    # Logging and Saving
    parser.add_argument('-p', '--print-freq', default=5,
                        type=int, help='print frequency')
    parser.add_argument('-o', '--output-dir',
                        default='./outputs/', help='path where to save')
    parser.add_argument('-exp', '--experiment', default='affordancesegmentation',
                        help='experiment name to save mlflow output')
    parser.add_argument('-c', '--comment', default='',
                        help='extra comment for mlflow artiffacts')
    parser.add_argument('-cp', '--checkpoint', default='',
                        help='resume from checkpoint')

    # Boolean Flags
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--dataset-mean",
        dest="dataset_mean",
        help="calculate dataset mean and std for normalization transform",
        action="store_true",
    )
    parser.add_argument(
        "--fast",
        dest="fast_run",
        help="Only run dev test for train/val/test through model",
        action="store_true",
    )
    parser.add_argument("-log",
                        "--log-mlflow",
                        dest="do_log",
                        help="Record training to mlflow",
                        action="store_true",
                        )
    parser.add_argument(
        "--visualize",
        dest="do_visualize",
        help="Visualize the model outputs after train / test",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--backend', default='ddp',
                        help='data parallel (dp), per-gpu model parallel (ddp) [default], per-node data parallel (ddp2) [requires SLURM_LOCALID]')
    if get_parser:
        return parser
    args = parser.parse_args()
    return args
