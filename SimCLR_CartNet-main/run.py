import argparse
import torch
import torch.backends.cudnn as cudnn
#from torchvision import models
#from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
#from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from models.cartnet import CartNet
from torch_geometric.graphgym.config import cfg, set_cfg
from utils import augment_batch
from loader.loader import create_loader
from tqdm import tqdm




#model_names = sorted(name for name in models.__dict__
#                     if name.islower() and not name.startswith("__")
#                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--dataset_path', metavar='DIR', default='./dataset/mp_139K/',
                    help='path to dataset')
parser.add_argument('-dataset_name', default='MP',
                    help='dataset name', choices=['stl10', 'cifar10', 'MP'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
                    #choices=model_names,
                    #help='model architecture: ' +
                    #     ' | '.join(model_names) +
                    #     ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=3, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int, ## crec que aquest pot ser el mateix que dim_in
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

## arguments de cartnet que afegeixo
parser.add_argument('--model', type=str, default="CartNet", help="Model Name")
parser.add_argument("--max_neighbours", type=int, default=25, help="Max neighbours (only for iComformer/eComformer)")
parser.add_argument("--radius", type=float, default=5.0, help="Radius for the Radius Graph Neighbourhood")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
parser.add_argument("--dim_in", type=int, default=32, help="Input dimension")
parser.add_argument("--dim_rbf", type=int, default=64, help="Number of RBF")
parser.add_argument("--augment", type=bool, default=True, help="")

parser.add_argument("--invariant", action="store_true", help="Rotation Invariant model")
parser.add_argument("--disable_temp", action="store_false", help="Disable Temperature")
parser.add_argument("--no_standarize_temp", action="store_false", help="Standarize temperature")
parser.add_argument("--disable_envelope", action="store_false", help="Disable envelope")
parser.add_argument('--disable_H', action='store_false', help='Hydrogens')
parser.add_argument('--disable_atom_types', action='store_false', help='Atom types')

parser.add_argument('--name', type=str, default="contrastive_loss_intents", help="name of the Wandb experiment" )
parser.add_argument("--wandb_project", type=str, default="CartNet SimCLR", help="Wandb project name")
parser.add_argument("--wandb_entity", type=str, default="laura-sola-garcia-universitat-polit-cnica-de-catalunya", help="Name of the wandb entity")



def main():
    set_cfg(cfg)
    args = parser.parse_args()

    cfg.data = args.dataset_name
    cfg.dataset_path = args.dataset_path
    cfg.model = args.model
    cfg.max_neighbours = -1 if cfg.model== "CartNet" else args.max_neighbours
    cfg.radius = args.radius
    cfg.num_layers = args.num_layers
    cfg.dim_in = args.dim_in
    cfg.dim_rbf = args.dim_rbf
    cfg.batch = args.batch_size
    cfg.workers = args.workers
    cfg.augment = args.augment
    cfg.invariant = args.invariant
    cfg.use_temp = False if cfg.dataset.name != "ADP" else args.disable_temp
    cfg.standarize_temp = args.no_standarize_temp
    cfg.envelope = args.disable_envelope
    cfg.use_H = args.disable_H
    cfg.use_atom_types = args.disable_atom_types
    cfg.seed = args.seed
    cfg.name = args.name
    cfg.run_dir = "results/"+cfg.name+"/"+str(cfg.seed)

    print('holi')
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # dataset = ContrastiveLearningDataset(args.data)
    # train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)

    train_loader = create_loader()


    # for iter, batch in tqdm(enumerate(train_loader), total=len(train_loader), ncols=50):
    #     print("batch", batch)
    #     print("batch.x", batch.x)
    #     print("id batch", batch.batch)
    #     print("id batch shape", batch.batch.shape)

    #     batch_aug_1 = augment_batch(batch.clone())
    #     batch_aug_2 = augment_batch(batch.clone())

    #     batch_aug = Batch.from_data_list([batch_aug_1, batch_aug_2]).to("cuda:0")

    #     print("batch", batch_aug)
    #     print("id batch", batch_aug.batch)
    #     print("id batch shape", batch_aug.batch.shape)
    #     print("batch_aug.x", batch_aug.x)
    #     return

    #model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    
    model = CartNet(dim_in=cfg.dim_in, dim_rbf=cfg.dim_rbf, num_layers=cfg.num_layers, 
                    invariant=cfg.invariant, temperature=cfg.use_temp, use_envelope=cfg.envelope,
                    atom_types=cfg.use_atom_types).to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        print('holi')
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
