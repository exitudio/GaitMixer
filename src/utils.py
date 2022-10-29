from functools import partial
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_sttransformer_layer(name, num_layers):
    if name.endswith(("cls_token", "mask_token", "pos_embed")):
        return 0
    elif 'to_embedding' in name:
        return 0
    elif 'spatial_blocks' in name:
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    elif 'temporal_blocks' in name:
        layer_id = int(name.split('.')[1]) + num_layers
        return layer_id + 1
    else:
        return num_layers*2 + 1

# https://github.com/microsoft/SimMIM/blob/0e7174608f5105f3d573f67724008c89b23b7fa2/optimizer.py#L139
# https://github.com/facebookresearch/mae/blob/6a2ba402291005b003a70e99f7c87d1a2c376b0d/util/lr_decay.py#L15


def get_finetune_param_groups_wrapper(model, lr, weight_decay, num_layers=4):
    LAYER_DECAY = .75
    get_layer_func = partial(get_sttransformer_layer,
                             num_layers=num_layers)
    scales = list(LAYER_DECAY ** i for i in reversed(range(num_layers*2 + 2)))
    params = get_finetune_param_groups(model,
                                       lr=lr,
                                       weight_decay=weight_decay,
                                       get_layer_func=get_layer_func,
                                       scales=scales)
    return params


def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # no weight decay for bias
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = weight_decay # if no lowerbound lr_decay gradient will exploid
        else:
            group_name = "decay"
            this_weight_decay = weight_decay # weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            # print('layer_id:', layer_id, name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def get_bone_length(points, b=128, f=60):
    bones = [[0,0], [0,1], [0,2], [1,3], [2,4], 
            [0,5], [0,6], [5,7], [6,8], [7,9], [8,10], [5,11], [6,12], [11,13], [12,14], [13,15], [14,16]]
    num_bone =  len(bones)
    
    bones = torch.tensor(bones, device=points.device)
    bones = bones[None, None].repeat(b,f,1,1)
    bones = bones.reshape(b, f, -1)[:, :, :, None].repeat(1,1,1,2)
    
    joint_bones = torch.gather(points, 2, bones)
    joint_bones = joint_bones.reshape(b, f, num_bone, 2, 2) #b, f, num_bone, (from, to), (x, y)
    bone_length = joint_bones[:,:,:,1] - joint_bones[:,:,:,0]
    return bone_length

def add_velocity(points):
    norm_points = points - points[:,:,0:1]
    
    velocity_1 = points[:,1:] - points[:,:59]
    velocity_1 = torch.cat((torch.zeros((points.shape[0], 1, 17,2), device=points.device), velocity_1), dim=1)
    
    velocity_2 = velocity_1[:, 2:] - velocity_1[:, 1:59]
    velocity_2 = torch.cat((torch.zeros((points.shape[0], 2, 17,2), device=points.device), velocity_2), dim=1)
    
    bones = get_bone_length(points, b=points.shape[0], f=points.shape[1])
    
    points = torch.cat((points, norm_points, velocity_1, velocity_2, bones), dim=3) #
    return points