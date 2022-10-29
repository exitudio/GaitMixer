
import wandb
import os
import torch
import shutil
import datetime


class UnifyLog():
    def __init__(self, opt, model):
        self.opt = opt
        if not opt.debug:
            # init path
            date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            opt.model_name = f"{date}_{opt.name}"
            opt.save_folder = f"../save/unify/{opt.project}/{opt.model_name}/"
            os.makedirs(opt.save_folder, exist_ok=False)
            shutil.copytree('../src', opt.save_folder +
                            'src', ignore=shutil.ignore_patterns('__pycache__', 'wandb'))
            shutil.copytree('../experiments', opt.save_folder +
                            'experiments', ignore=shutil.ignore_patterns('__pycache__'))
            # checkpoint path
            self.checkpoint_path = self.opt.save_folder+'checkpoint/'
            os.makedirs(self.checkpoint_path, exist_ok=False)
            # init wandb
            wandb.init(project=opt.project,
                       name=opt.name, entity='USER', config=opt)
            wandb.watch(model)
            print('---- init log -----')
            print('path:', opt.save_folder)
            print('opt:', opt)

    def log(self, *args, **kwargs):
        if not self.opt.debug:
            wandb.log(*args, **kwargs)

    def save_model(self, model, name):
        if not self.opt.debug and self.opt.save_model:
            if hasattr(model, 'module'):
                model = model.module
            torch.save(model.state_dict(), self.checkpoint_path+name)
