import argparse
import torch
import models.ResGCNv1
from datasets.graph import Graph
from einops.layers.torch import Rearrange
import torch.nn as nn


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser(
        description="Training model on gait sequence")
    parser.add_argument("dataset", choices=["casia-b-query", "casia-b", "OUMVLP"])
    parser.add_argument("train_data_path", help="Path to train data CSV")
    parser.add_argument("--valid_data_path",
                        help="Path to validation data CSV")
    parser.add_argument("--train_id",
                        help="train_id")
    parser.add_argument("--test_id",
                        help="test_id")
    parser.add_argument("--loss_func", choices=["supcon", "triplet"], default="triplet")
    parser.add_argument('--balance_sampler', type=str2bool,
                        nargs='?', default=True)
    parser.add_argument("--kernel_frame", type=int, default=31)
    parser.add_argument("--rm_conf", type=str2bool,
                        nargs='?', default=True)
    parser.add_argument("--joint_drop", choices=["single", "none"], default="none") # rm drop arms
    parser.add_argument("--sampler_num_sample", type=int, default=4)
    parser.add_argument("--weight_path", help="Path to weights for model")
    parser.add_argument("--model_type", choices=["spatialtransformer_temporalconv", "spatiotemporal_transformer", "gaitgraph"], default="spatialtransformer_temporalconv")

    # Optionals
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--gpus", default="0", help="-1 for CPU, use comma for multiple gpus"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_validation", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--test_epoch_interval", type=int, default=10)
    parser.add_argument('--save_model', type=str2bool,
                        nargs='?', default=False)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--project", default="project_name")
    parser.add_argument("--name", help="experiment_name", default="project_name")

    parser.add_argument("--sequence_length", type=int, default=60)
    parser.add_argument("--embedding_layer_size", type=int, default=128)
    parser.add_argument("--embedding_spatial_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=6e-3)
    parser.add_argument("--point_noise_std", type=float, default=0.05)
    parser.add_argument("--joint_noise_std", type=float, default=0.1)
    parser.add_argument("--flip_probability", type=float, default=0.5)
    parser.add_argument("--mirror_probability", type=float, default=0.5)
    parser.add_argument('--debug', type=str2bool, nargs='?', default=True)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    opt = parser.parse_args()

    # Sanitize opts
    # opt.gpus_str = opt.gpus
    # opt.gpus = [i for i in range(len(opt.gpus.split(",")))]

    return opt


def setup_environment(opt):
    # HACK: Fix tensorboard
    # import tensorflow as tf
    # import tensorboard as tb

    # tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.cuda = True #opt.gpus[0] >= 0
    torch.device("cuda" if opt.cuda else "cpu")

    return opt

def get_model_resgcn():
    graph = Graph("coco")
    model_args = {
        "A": torch.tensor(graph.A, dtype=torch.float32, requires_grad=False),
        "num_class": 128,
        "num_input": 1,
        "num_channel": 3,
        "parts": graph.parts,
    }
    return nn.Sequential(
        Rearrange('b f j e -> b 1 e f j'),
        models.ResGCNv1.create('resgcn-n39-r8', **model_args))


def get_trainer(model, opt, steps_per_epoch):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    _optimizer_base = optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        _optimizer_base, opt.learning_rate, epochs=opt.epochs, steps_per_epoch=steps_per_epoch
    )
    return optimizer, scheduler, scaler

def load_checkpoint(model, opt):
    if opt.weight_path is not None:
        checkpoint = torch.load(opt.weight_path)
        model.load_state_dict(checkpoint, strict=True)

def count_parameters(model):
    """
    Useful function to compute number of parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
