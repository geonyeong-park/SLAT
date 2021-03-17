import argparse
import yaml
import os
import os.path as osp
import torch
from shutil import copyfile
from solver import Solver
from data.data_loader import DataWrapper
import torch.backends.cudnn as cudnn
from eval import eval

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generalization by Noise")
    parser.add_argument("--gpu", type=int, nargs='+', default=None, required=True,
                        help="choose gpu device.")
    parser.add_argument("--yaml", type=str, default='config.yaml',
                        help="yaml pathway")
    parser.add_argument("--exp_name", type=str, default='', required=True,
                        help="")
    parser.add_argument("--dataset_name", type=str, default=None, required=False,
                        help="")
    parser.add_argument("--model_structure", type=str, default='advGNI', required=True,
                        help="'base', 'advGNI', 'advGNI_GA', 'Free', 'CURE', 'PGD', 'FGSM', 'FGSM_RS', 'FGSM_GA', 'yopo', 'FGSM_ckpt'")
    parser.add_argument("--resume", type=str, default=None,
                        required=False, help="")
    parser.add_argument("--no_auto", default=False, action='store_true',
                        required=False, help="")
    parser.add_argument("--pretrain", default=False, action='store_true',
                        required=False, help="")
    parser.add_argument("--eta", type=float, default=None,
                        required=False, help="Variance")
    parser.add_argument("--PGD_iters", default=None, type=int, required=False, help="")
    parser.add_argument("--batch_size", default=None, type=int, required=False, help="")
    parser.add_argument("--advGNI_iters", default=None, type=int, required=False, help="")
    parser.add_argument("--alpha", default=None, type=float, required=False, help="")
    parser.add_argument("--GA_coeff", default=None, type=float, required=False, help="")
    parser.add_argument("--num_epochs", type=int, default=None,
                        required=False, help="")
    parser.add_argument("--lr_milestone", type=float, nargs='+', default=None, required=False, help=".")
    parser.add_argument("--schedule", type=str, default=None, required=False, help=".")
    parser.add_argument("--lr", type=float, default=None, required=False, help=".")

    return parser.parse_args()


def main(config, args):
    """Create the model and start the training."""

    # -------------------------------
    # Setting logging files

    snapshot_dir = config['exp_setting']['snapshot_dir']
    log_dir = config['exp_setting']['log_dir']
    exp_name = args.exp_name

    snapshot_dir, log_dir = os.path.join(snapshot_dir, exp_name), os.path.join(log_dir, exp_name)
    path_list = [snapshot_dir, log_dir]

    for item in path_list:
        if not os.path.exists(item):
            os.makedirs(item)

    config['exp_setting']['snapshot_dir'] = snapshot_dir
    config['exp_setting']['log_dir'] = log_dir

    # -------------------------------
    # Setting GPU

    gpus_tobe_used = ','.join([str(gpuNum) for gpuNum in args.gpu])
    print('gpus_tobe_used: {}'.format(gpus_tobe_used))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_tobe_used)

    cudnn.enabled = True
    cudnn.benchmark = True
    # -------------------------------
    # Setting Test arguments
    if args.dataset_name is not None:
        print('dataset: ', args.dataset_name)
        config['dataset']['name'] = args.dataset_name
    if args.model_structure is not None:
        structure = args.model_structure
        assert structure in config['model']['baseline']
        print('model: ', structure)
        config['model']['baseline'] = structure
    if args.schedule is not None:
        print('LR schedule: ', args.schedule)
        assert args.schedule == 'cyclic' or args.schedule == 'multistep'
        config['optimizer']['schedule'] = args.schedule
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cuda:0')
        print('load {}'.format(args.resume))
    if args.eta is not None:
        print('Eta: ', args.eta)
        config['model']['ResNet']['eta'] = args.eta
    if args.PGD_iters is not None:
        print('PGD iters: {}'.format(args.PGD_iters))
        config['model']['PGD']['iters'] = args.PGD_iters
    if args.batch_size is not None:
        dataset_name = config['dataset']['name']
        config['dataset'][dataset_name]['batch_size'] = args.batch_size
    if args.advGNI_iters is not None and args.model_structure == 'advGNI':
        print('{} iters: {}'.format(args.model_structure, args.advGNI_iters))
        config['model'][args.model_structure]['iters'] = args.advGNI_iters
    if args.alpha is not None:
        print('Alpha for hidden layers: {}'.format(args.alpha))
        config['model'][structure]['alpha'] = args.alpha
    if args.GA_coeff is not None:
        print('Coeff for Gradient Alignment: {}'.format(args.GA_coeff))
        config['model']['FGSM_GA']['coeff'] = args.GA_coeff
    if args.num_epochs is not None:
        print('epochs: ', args.num_epochs)
        s = config['optimizer']['schedule']
        config['train']['num_epochs'][s] = args.num_epochs
    if args.lr_milestone is not None:
        print('LR milestones: ', args.lr_milestone)
        config['optimizer']['lr_milestone'] = args.lr_milestone
    if args.lr is not None:
        print('LR: ', args.lr)
        s = config['optimizer']['schedule']
        config['optimizer']['lr'][s] = args.lr

    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    # -------------------------------

    dataset = DataWrapper(config)
    solver = Solver(dataset, config)

    if args.pretrain:
        solver.pretrain()
        print('Pretraining Finished')
        return

    if args.resume is None:
        solver.train()
    else:
        auto=False if args.no_auto else True
        BB_ckpt = torch.load('snapshots/BlackBox_eps8_PGD7/pretrain.pth')
        eval(solver, checkpoint, BB_ckpt, config['model']['ResNet']['eta'], auto, structure)



if __name__ == '__main__':
    args = get_arguments()
    config = yaml.load(open(args.yaml, 'r'))

    main(config, args)
