import argparse
import collections
import numpy as np
import pyro
import torch
import lightning.pytorch
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import training
from utils import read_json

# fix random seeds for reproducibility
SEED = 123
lightning.pytorch.seed_everything(SEED, workers=True)

def setup(config):
    logger = config.init_obj('logger', lightning.pytorch.loggers,
                             save_dir=config.log_dir)

    # set up data modules
    data_module = config.init_obj("data_module", training)
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # build model architecture and its Lightning module
    model = config.init_obj("arch", module_arch, data_module.dims)
    lmodule = config.init_obj("lmodule", training, model, data_module)

    # build Lightning trainer
    checkpoint = config.init_obj("checkpoint", lightning.pytorch.callbacks,
                                 dirpath=config.save_dir, save_top_k=-1,
                                 save_last=True)
    trainer = config.init_obj("trainer", lightning.pytorch,
                              callbacks=[checkpoint], logger=logger,
                              log_every_n_steps=1)
    return data_module, lmodule, trainer

def from_file(config_file, checkpoint=None):
    config = ConfigParser(read_json(config_file), resume=checkpoint)
    return config, setup(config)

def main(config):
    logger = config.get_logger("train")
    logger.info(config.log_dir)
    dm, model, trainer = setup(config)
    trainer.fit(model, dm)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Population Predictive Coding (PPC) training script')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='lmodule;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_module;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
