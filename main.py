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
                        help="'base', 'advGNI', 'advGNI_GA', 'Free', 'CURE', 'PGD', 'FGSM', 'FGSM_RS', 'FGSM_GA', 'GradPenal'")
    parser.add_argument("--resume", type=str, default=None,
                        required=False, help="")
    parser.add_argument("--pretrain", default=False, action='store_true',
                        required=False, help="")
    parser.add_argument("--eta", type=float, default=None,
                        required=False, help="Variance")
    parser.add_argument("--PGD_iters", default=None, type=int, required=False, help="")
    parser.add_argument("--coeff_lower", default=None, type=float, required=False, help="")
    parser.add_argument("--coeff_higher", default=None, type=float, required=False, help="")
    parser.add_argument("--GA_coeff", default=None, type=float, required=False, help="")
    parser.add_argument("--num_epochs", type=int, default=None,
                        required=False, help="")
    parser.add_argument("--lr_milestone", type=float, nargs='+', default=None, required=False, help=".")
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
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        print('load {}'.format(args.resume))
    if args.eta is not None:
        print('Eta: ', args.eta)
        config['model']['ResNet']['eta'] = args.eta
    if args.PGD_iters is not None:
        print('PGD iters: {}'.format(args.PGD_iters))
        config['model']['PGD']['iters'] = args.PGD_iters
    if args.coeff_lower is not None:
        print('Alpha for low layers: {}'.format(args.coeff_lower))
        config['model']['advGNI']['coeff_lower'] = args.coeff_lower
    if args.coeff_higher is not None:
        print('Alpha for high layers: {}'.format(args.coeff_higher))
        config['model']['advGNI']['coeff_higher'] = args.coeff_higher
    if args.GA_coeff is not None:
        print('Coeff for Gradient Alignment: {}'.format(args.GA_coeff))
        config['model']['FGSM_GA']['coeff'] = args.GA_coeff
    if args.num_epochs is not None:
        print('epochs: ', args.num_epochs)
        config['train']['num_epochs'] = args.num_epochs
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
        eval(solver, checkpoint, config['model']['ResNet']['eta'])



if __name__ == '__main__':
    args = get_arguments()
    config = yaml.load(open(args.yaml, 'r'))

    main(config, args)
