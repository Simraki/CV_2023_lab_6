import argparse
import datetime
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import transforms as flow_transforms
import models
from datasets.flyingchairs import flying_chairs
from metrics import multiscale_epe, real_epe
from utils import flow2rgb, AverageMeter, save_checkpoint, load_from_checkpoint

model_names = ['flownets']
solver_names = ['adam', 'sgd']

parser = argparse.ArgumentParser(
    description='PyTorch FlowNet Training on several datasets',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Dataset args
parser.add_argument(
    'data', metavar='DIR', help='path to dataset'
)
parser.add_argument(
    '--split-value', default=0.8, type=float,
    help='test-val split proportion between 0 (only test) and 1 (only train), '
         'will be overwritten if a split file is set'
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Seed the train-val split to enforce reproducibility (consistent restart too)",
)
parser.add_argument(
    '--div-flow', type=int, default=20,
    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results'
)

# Architecture args
parser.add_argument(
    '--arch', '-a', metavar='ARCH', default='flownets',
    choices=model_names,
    help='model architecture: ' +
         ' | '.join(model_names)
)
parser.add_argument(
    '-bn', '--batch-norm', action='store_true',
    help='active active normalization'
)
parser.add_argument(
    '--solver', default='adam', choices=solver_names,
    help='solver algorithms: ' + ' | '.join(solver_names)
)
parser.add_argument(
    '--pretrained', dest='pretrained', action='store_true',
    help='use pretrained from checkpoints'
)
parser.add_argument(
    '--load-path', dest='load_path', default=None,
    help='path for checkpoints loading'
)
parser.add_argument(
    '--load-best', action='store_true',
    help='load best by epe checkpoint'
)

# Train args
parser.add_argument(
    '--epochs', default=300, type=int, metavar='N',
    help='number of total epochs to run'
)
parser.add_argument(
    '--epoch-size', default=1000, type=int, metavar='N',
    help='manual epoch size (will match dataset size if set to 0)'
)
parser.add_argument(
    '-b', '--batch-size', default=8, type=int,
    metavar='N', help='mini-batch size'
)
parser.add_argument(
    '--lr', '--learning-rate', default=1e-4, type=float,
    metavar='LR', help='initial learning rate'
)
parser.add_argument(
    '--milestones', default=[100, 150, 200], metavar='N', nargs='*',
    help='epochs at which learning rate is divided by 2'
)

# Test args
parser.add_argument(
    '-e', '--evaluate', dest='evaluate', action='store_true',
    help='evaluate model on test set'
)

# Other args
parser.add_argument(
    '-j', '--workers', default=8, type=int, metavar='N',
    help='number of data loading workers'
)
parser.add_argument(
    '--print-freq', '-p', default=None, type=int,
    metavar='N', help='print frequency'
)
parser.add_argument(
    '--no-date', action='store_true',
    help='don\'t append date timestamp to folder'
)

best_epe = -1
n_iter = int(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = {}


def main():
    global args, best_epe
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    save_path = '{},{},{}epochs{},b{},lr{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize' + str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr
    )

    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m%d%H%M")
        save_path = os.path.join(timestamp, save_path)

    save_path = os.path.join('processed', save_path)
    print(f'=> will save everything to {save_path}')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

    # Data loading code
    input_transform = transforms.Compose(
        [
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.45, 0.432, 0.411], std=[1, 1, 1])
        ]
    )
    target_transform = transforms.Compose(
        [
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0], std=[args.div_flow, args.div_flow])
        ]
    )

    print(f"=> fetching img pairs in '{args.data}'")

    train_set, test_set = flying_chairs(
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        split=args.split_value,
        seed=args.seed
    )

    print(
        f'{len(test_set) + len(train_set)} samples found, {len(train_set)} train samples and {len(test_set)} test samples'
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False
    )

    # create model
    model = models.make_flow_net_s()

    if args.pretrained and args.load_path:
        model.load_state_dict(load_from_checkpoint(args.load_path, load_best=args.load_best)['state_dict'])
        print(f"=> load{' best' if args.load_best else ''} from checkpoints: {args.load_path}")
        print(f"=> using pre-trained model: {args.arch}")
    else:
        print(f"=> creating model: {args.arch}")

    if device.type == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    assert (args.solver in ['adam', 'sgd'])
    print(f'=> setting solver: {args.solver}')

    if args.solver == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)

    if args.evaluate:
        best_epe = validate(test_loader, model, 0, output_writers)
        return

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    for epoch in range(0, args.epochs):

        # train for one epoch
        train_loss, train_epe = train(train_loader, model, optimizer, epoch, train_writer, scheduler)
        train_writer.add_scalar('mean epe', train_epe, epoch)

        # evaluate on test set
        with torch.no_grad():
            epe = validate(test_loader, model, epoch, output_writers)
        test_writer.add_scalar('mean epe', epe, epoch)

        if best_epe < 0:
            best_epe = epe

        is_best = epe < best_epe
        best_epe = min(epe, best_epe)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_epe': best_epe,
                'div_flow': args.div_flow
            }, is_best, save_path
        )


def train(train_loader, model, optimizer, epoch, train_writer, scheduler):
    global n_iter, args

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_epes = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(device)
        input = torch.cat(input, 1).to(device)

        # compute output
        output = model(input)

        loss = multiscale_epe(output, target)
        flow2_epe = args.div_flow * real_epe(output[0], target)
        # record loss and epe
        losses.update(loss.item(), target.size(0))
        train_writer.add_scalar('train_loss', loss.item(), n_iter)
        flow2_epes.update(flow2_epe.item(), target.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == 0 or (i >= epoch_size or (i + 1) == len(train_loader)) or (
                args.print_freq is not None and (i + 1) % args.print_freq == 0):
            print(
                f'Epoch: [{epoch}][{i + 1}/{epoch_size}]\t Time {batch_time}\t Data {data_time}\t Loss {losses}\t epe {flow2_epes}'
            )
        n_iter += 1
        if i >= epoch_size:
            break

    return losses.avg, flow2_epes.avg


def validate(val_loader, model, epoch, output_writers):
    global args

    batch_time = AverageMeter()
    flow2_epes = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = torch.cat(input, 1).to(device)

        # compute output
        output = model(input)
        flow2_epe = args.div_flow * real_epe(output, target)
        # record epe
        flow2_epes.update(flow2_epe.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i < len(output_writers):  # log first output of first batches
            if epoch == 0:
                mean_values = torch.tensor([0.45, 0.432, 0.411], dtype=input.dtype).view(3, 1, 1)
                output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0], max_value=10), 0)
                output_writers[i].add_image('Inputs', (input[0, :3].cpu() + mean_values).clamp(0, 1), 0)
                output_writers[i].add_image('Inputs', (input[0, 3:].cpu() + mean_values).clamp(0, 1), 1)
            output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

        if i == 0 or (i + 1) == len(val_loader) or (args.print_freq is not None and (i + 1) % args.print_freq == 0):
            print(
                f'Test: [{i + 1}/{len(val_loader)}]\t Time {batch_time}\t epe {flow2_epes}'
            )

    print(f' * epe {flow2_epes.avg:.3f}')

    return flow2_epes.avg


if __name__ == '__main__':
    main()
