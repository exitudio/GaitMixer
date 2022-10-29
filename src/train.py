import sys
import time
from torchvision import transforms
from datasets.augmentation import *
from evaluate import _evaluate_casia_b, evaluate
from losses import SupConLoss
from utils import AverageMeter
from common import *
from models.SpatialTransformerTemporalConv import SpatialTransformerTemporalConv, SpatioTemporalTransformer
from logs import UnifyLog
from pytorch_metric_learning import miners, losses
from sampler import BalancedBatchSampler
from tqdm import tqdm
from datasets.gait import (
    CasiaBPose,
    CasiaQueryDataset
)

miner = miners.MultiSimilarityMiner()


def train(train_loader, model, loss_func, optimizer, scheduler, scaler, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (points, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if opt.loss_func == 'supcon':
            bsz = points[0].shape[0]
            points = torch.cat([points[0], points[1]], dim=0)
        else:
            bsz = points.shape[0]
        labels = target[0]

        if torch.cuda.is_available():
            points = points.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        features = model(points)
        if opt.loss_func == 'supcon':
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = loss_func(features, labels)
        else:
            hard_pairs = miner(features, labels)
            loss = loss_func(features, labels, hard_pairs)

        # update metric
        losses.update(loss.item(), bsz)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        ############################################################################

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 10 == 0:
            print(
                f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t"
                f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"loss {losses.val:.3f} ({losses.avg:.3f})"
            )
            sys.stdout.flush()

    return losses.avg


def main(opt):
    opt = setup_environment(opt)

    # Dataset
    transform = transforms.Compose(
        [
            MirrorPoses(opt.mirror_probability),
            FlipSequence(opt.flip_probability),
            RandomSelectSequence(opt.sequence_length),
            PointNoise(std=opt.point_noise_std),
            JointNoise(std=opt.joint_noise_std),
            joint_drop if opt.joint_drop == 'single' else lambda x:x,
            remove_conf(enable=opt.rm_conf),
            normalize_width,
            ToTensor()
        ],
    )
    if opt.loss_func == 'supcon':
        transform = TwoNoiseTransform(transform)
    val_transform = transforms.Compose(
        [
            SelectSequenceCenter(opt.sequence_length),
            remove_conf(enable=opt.rm_conf),
            normalize_width,
            ToTensor()
        ]
    )

    if opt.dataset == "casia-b":
        dataset = CasiaBPose(
            opt.train_data_path,
            sequence_length=opt.sequence_length,
            duplicate_bgcl=True,
            transform=transform,
        )
        dataset_valid = CasiaBPose(
            opt.valid_data_path,
            sequence_length=opt.sequence_length,
            transform=ThreeCenterSequenceTransform(
                transform=val_transform, sequence_length=opt.sequence_length),
        )
    elif opt.dataset == "casia-b-query":
        dataset = CasiaQueryDataset(
            opt.train_data_path,
            id_range=opt.train_id,
            duplicate_bgcl=True,
            transform=transform
        )
        dataset_valid = CasiaQueryDataset(
            opt.train_data_path,
            id_range=opt.test_id,
            transform=ThreeCenterSequenceTransform(
                transform=val_transform, sequence_length=opt.sequence_length),
        )

    if opt.balance_sampler:
        if opt.train_id is None:
            num_class = 74
        else:
            num_class = int(opt.train_id.split('-')[1])
            
        print('---- balance sampler ----- class=', num_class)
        labels = []
        for i in dataset.targets:
            labels.append(i[0])
        labels = torch.tensor(labels)
        _sampler = BalancedBatchSampler(
            labels=labels, n_classes=num_class, n_samples=opt.sampler_num_sample)

        train_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=opt.num_workers,
            pin_memory=True,
            batch_sampler=_sampler,
        )
    else:
        print('---- No balance sampler -----')
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    val_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=opt.batch_size_validation,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    if opt.model_type == "spatialtransformer_temporalconv":
        model = SpatialTransformerTemporalConv(
            num_frame=opt.sequence_length, in_chans=2 if opt.rm_conf else 3, spatial_embed_dim=opt.embedding_spatial_size, out_dim=opt.embedding_layer_size, num_joints=17, kernel_frame=opt.kernel_frame)
    elif opt.model_type == "spatiotemporal_transformer":
        model = SpatioTemporalTransformer(
            num_frame=opt.sequence_length, in_chans=2 if opt.rm_conf else 3, spatial_embed_dim=opt.embedding_spatial_size, out_dim=opt.embedding_layer_size, num_joints=17)
    elif opt.model_type == "gaitgraph":
        model = get_model_resgcn()
    else:
        raise ValueError("No model type support:", opt.model_type)
    unify_log = UnifyLog(opt, model)

    print("# parameters: ", count_parameters(model))

    # Load checkpoint or weights
    load_checkpoint(model, opt)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if opt.loss_func == 'supcon':
        loss_func = SupConLoss(temperature=0.004, base_temperature=0.004)
    else:
        loss_func = losses.TripletMarginLoss(margin=0.01)
    if opt.cuda:
        model.cuda()
        loss_func.cuda()

    # Trainer
    optimizer, scheduler, scaler = get_trainer(model, opt, len(train_loader))

    best_acc = 0
    loss = 0
    for epoch in tqdm(range(opt.start_epoch, opt.epochs + 1)):
        # train for one epoch
        time1 = time.time()
        loss = train(
            train_loader, model, loss_func, optimizer, scheduler, scaler, epoch, opt
        )

        time2 = time.time()
        print(f"epoch {epoch}, total time {time2 - time1:.2f}")

        unify_log.log(
            {'loss/train': loss, 'lr': optimizer.param_groups[0]["lr"]}, step=epoch)

        if epoch % opt.test_epoch_interval == 0 or epoch == opt.epochs:
            # evaluation
            accuracy_avg, sub_accuracies = evaluate(
                val_loader, model, opt.evaluation_fn)
            unify_log.log(
                {'accuracy/validation/avg': accuracy_avg}, step=epoch)
            for key, sub_accuracy in sub_accuracies.items():
                unify_log.log(
                    {'accuracy/validation/'+key: sub_accuracy}, step=epoch)

            print(f"epoch {epoch}, avg accuracy {accuracy_avg:.4f}")
            is_best = accuracy_avg > best_acc
            if is_best:
                best_acc = accuracy_avg

    unify_log.save_model(model, 'last.pth')
    print(f"best accuracy: {best_acc*100:.2f}")


if __name__ == "__main__":

    opt = parse_option()
    opt.evaluation_fn = _evaluate_casia_b
    main(opt)
