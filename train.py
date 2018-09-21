"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_logging_folders, \
    write_html, write_loss, get_config, write_images
from optparse import OptionParser
from torch.autograd import Variable
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
from trainer import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import os
import sys
import shutil
import time
from logger import create_logger
import tensorboard_logger

parser = OptionParser()
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--output_base', type=str, default='.', help="outputs path")
parser.add_option("--resume", action="store_true")
parser.add_option("--model_path", type=str)


def main(argv):
    (opts, args) = parser.parse_args(argv)
    cudnn.benchmark = True

    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']

    # Setup logger and output folders
    output_subfolders = prepare_logging_folders(config['output_root'], config['experiment_name'])
    logger = create_logger(os.path.join(output_subfolders['logs'], 'train_log.log'))
    shutil.copy(opts.config, os.path.join(output_subfolders['logs'], 'config.yaml')) # copy config file to output folder

    tb_logger = tensorboard_logger.Logger(output_subfolders['logs'])

    logger.info('============ Initialized logger ============')
    logger.info('Config File: {}'.format(opts.config))

    # Setup model and data loader
    trainer = MUNIT_Trainer(config, opts)
    trainer.cuda()
    loaders = get_all_data_loaders(config)
    val_display_images = next(iter(loaders['val']))
    logger.info('Test images: {}'.format(val_display_images['A_paths']))

    # Start training
    iterations = trainer.resume(opts.model_path, hyperparameters=config) if opts.resume else 0

    while True:
        for it, images in enumerate(loaders['train']):
            trainer.update_learning_rate()
            images_a = images['A']
            images_b = images['B']

            images_a, images_b = Variable(images_a.cuda()), Variable(images_b.cuda())

            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                for tag, value in trainer.loss.items():
                    tb_logger.scalar_summary(tag, value, iterations)

                val_output_imgs = trainer.sample(
                    Variable(val_display_images['A'].cuda()),
                    Variable(val_display_images['B'].cuda()))

                tb_imgs = []
                for imgs in val_output_imgs.values():
                    tb_imgs.append(torch.cat(torch.unbind(imgs, 0), dim=2))

                tb_logger.image_summary(list(val_output_imgs.keys()), tb_imgs, iterations)

            if (iterations + 1) % config['print_iter'] == 0:
                logger.info("Iteration: {:08}/{:08} Discriminator Loss: {:.4f} Generator Loss: {:.4f}".format(
                    iterations + 1, max_iter, trainer.loss['D/total'], trainer.loss['G/total']))

            # Write images
            # if (iterations + 1) % config['image_save_iter'] == 0:
            #     val_output_imgs = trainer.sample(
            #         Variable(val_display_images['A'].cuda()),
            #         Variable(val_display_images['B'].cuda()))
            #
            #     for key, imgs in val_output_imgs.items():
            #         key = key.replace('/', '_')
            #         write_images(imgs, config['display_size'], '{}/{}_{:08}.jpg'.format(output_subfolders['images'], key, iterations+1))
            #
            #     logger.info('Saved images to: {}'.format(output_subfolders['images']))

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(output_subfolders['models'], iterations)

            iterations += 1
            if iterations >= max_iter:
                return


if __name__ == '__main__':
    main(sys.argv)
