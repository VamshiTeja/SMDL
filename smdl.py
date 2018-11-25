import argparse, time, os, logging, pprint

import torch

from models import *
from lib.utils import *
from lib.config import cfg, cfg_from_file
from lib.samplers.submodular_batch_sampler import SubmodularBatchSampler


def submodular_training(gpus):
    train_start_time = time.time()
    num_classes = cfg.dataset.total_num_classes

    accuracies = []
    for round_count in range(cfg.repeat_rounds):

        # Initialize the model
        model = get_model()
        model.apply(weights_init)
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        optimizer = torch.optim.SGD(model.parameters(), cfg.learning_rate, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        log(model)

        # Loading the Dataset
        train_dataset, test_dataset = setup_dataset()
        if not cfg.use_custom_batch_selector:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                       num_workers=1)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size_test, shuffle=False,
                                                  num_workers=1)

        train_accs = []
        test_accs = []
        losses = []
        # Repeat for each epoch
        for epoch_count in range(cfg.epochs):
            adjust_lr(epoch_count, optimizer, cfg.learning_rate)

            start_time = time.time()

            # Should find the ordering of the samples using SubModularity at the beginning of each epoch.
            if cfg.use_custom_batch_selector:
                submodular_batch_sampler = SubmodularBatchSampler(model, train_dataset, cfg.batch_size)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=submodular_batch_sampler,
                                                           num_workers=1)

            train_acc, loss = train(train_loader, model, criterion, optimizer, epoch_count, cfg.epochs,
                              round_count, cfg.repeat_rounds)
            test_acc = test(test_loader, model, epoch_count, cfg.epochs,
                            round_count, cfg.repeat_rounds)

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            losses.append(loss)
            log('Time per epoch: {0:.4f}s \n'.format(time.time() - start_time))

        # Saving model and metrics
        plot_per_epoch_accuracies(train_accs, test_accs, round_count)
        output_dir = cfg.output_dir + '/models'
        filename = 'round_' + str(round_count + 1) + 'epoch_' + str(cfg.epochs) + '.pth'
        save_model(model, output_dir + '/' + filename)
        log('Model saved to ' + output_dir + '/' + filename)

        save_accuracies(test_accs, cfg.output_dir + '/accuracies/'+cfg.run_label+'_test_acc_round_' + str(round_count))
        save_accuracies(train_accs, cfg.output_dir + '/accuracies/'+cfg.run_label+'_train_acc_round_' + str(round_count))
        save_accuracies(losses, cfg.output_dir + '/accuracies/'+cfg.run_label+'_loss_round_' + str(round_count))

    log('Training complete. Total time: {0:.4f} mins.'.format((time.time() - train_start_time)/60))


def train(train_loader, model, criterion, optimizer, epoch_count, max_epoch,
          round_count, max_rounds, logging_freq=10, detailed_logging=True):
    losses = Metrics()
    top1 = Metrics()

    model.train()

    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        output, _ = model(input)
        loss = criterion(output,target)

        acc = compute_accuracy(output, target)[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(acc.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % logging_freq == 0 and detailed_logging:
            log('Round: {0:3d}/{1}\t  Epoch {2:3d}/{3} [{4:3d}/{5}] ' \
                  '\t Loss: {loss.val:.4f}({loss.avg:.4f}) ' \
                  '\t Training_Accuracy: {accuracy.val:.4f}({accuracy.avg:.4f})'.format(round_count+1, max_rounds,
                                                                             epoch_count+1, max_epoch, i, len(train_loader),
                                                                             loss=losses, accuracy=top1))

    log('Round: {0:3d}/{1}\t  Epoch {2:3d}/{3} ' \
          '\t Loss: {loss.val:.4f}({loss.avg:.4f}) ' \
          '\t Training_Accuracy: {accuracy.val:.4f}({accuracy.avg:.4f})'.format(round_count + 1, max_rounds,
                                                                                epoch_count + 1, max_epoch,
                                                                                loss=losses, accuracy=top1))
    return top1.avg, losses.avg


def test(test_loader, model, epoch_count, max_epoch, round_count, max_rounds, logging_freq=10, detailed_logging=False):
    top1 = Metrics()
    model.eval()

    for i, (input, target) in enumerate(test_loader):
        input, target = input.cuda(), target.cuda()
        output,_ = model(input)
        acc = compute_accuracy(output, target)[0]
        top1.update(acc.item(), input.size(0))

        if i % logging_freq == 0 and detailed_logging:
            log('Round: {0:3d}/{1}\t  Epoch {2:3d}/{3}[{4:3d}/{5}] ' \
                '\t Testing_Accuracy: {accuracy.val:.4f}({accuracy.avg:.4f})'.format(round_count + 1, max_rounds,
                                                                                      epoch_count + 1, max_epoch, i,
                                                                                      len(test_loader), accuracy=top1))

    log('Round: {0:3d}/{1}\t  Epoch {2:3d}/{3} ' \
        '\t Testing_Accuracy: {accuracy.val:.4f}({accuracy.avg:.4f})'.format(round_count + 1, max_rounds,
                                                                              epoch_count + 1, max_epoch,
                                                                              accuracy=top1))

    return top1.avg


def adjust_lr(epoch, optimizer, base_lr):
    if epoch < 20:
        lr = base_lr
    elif epoch < 40:
        lr = base_lr * 0.1
    elif epoch < 50:
        lr = base_lr * 0.01
    elif epoch < 60:
        lr = base_lr * 0.001
    elif epoch < 80:
        lr = base_lr * 0.0001
    else:
        lr = base_lr * 0.00001
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def save_model(model, location):
    torch.save(model.state_dict(), location)


def main():
    parser = argparse.ArgumentParser(description="SMDL: SubModular Dataloader")
    parser.add_argument("--cfg", dest='cfg_file', default='./config/smdl.yml', type=str, help="An optional config file"
                                                                                               " to be loaded")
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    if not os.path.exists('output'):
        os.makedirs('output')

    timestamp = time.strftime("%m%d_%H%M%S")
    cfg.timestamp = timestamp

    output_dir = './output/' + cfg.run_label + '_' + cfg.timestamp
    cfg.output_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/models')
        os.makedirs(output_dir + '/plots')
        os.makedirs(output_dir + '/logs')
        os.makedirs(output_dir + '/accuracies')

    logging.basicConfig(filename=output_dir + '/logs/smdl_' + timestamp + '.log', level=logging.DEBUG,
                        format='%(levelname)s:\t%(message)s')

    log(pprint.pformat(cfg))

    gpu_list = cfg.gpu_ids.split(',')
    gpus = [int(iter) for iter in gpu_list]
    torch.cuda.set_device(gpus[0])
    torch.backends.cudnn.benchmark = True

    if cfg.seed != 0:
        np.random.seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(cfg.seed)

    submodular_training(gpus)



if __name__ == "__main__":
    main()
