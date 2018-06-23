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
import tensorboardX
import shutil
import time
from logger import create_logger

parser = OptionParser()
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--output_base', type=str, default='.', help="outputs path")
parser.add_option("--resume", action="store_true")


def main(argv):
    (opts, args) = parser.parse_args(argv)
    cudnn.benchmark = True
    model_name = os.path.splitext(os.path.basename(opts.config))[0]

    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']

    # Setup logger and output folders
    output_folder = os.path.join(config['output_root'], time.strftime("%Y%m%d-%H%M%S"))
    output_subfolders = prepare_logging_folders(output_folder)
    logger = create_logger(os.path.join(output_folder, 'train.log'))
    shutil.copy(opts.config, os.path.join(output_folder, 'config.yaml')) # copy config file to output folder
    train_writer = tensorboardX.SummaryWriter(output_subfolders['logs'])

    logger.info('============ Initialized logger ============')
    logger.info('Config File: {}'.format(opts.config))
    logger.info('Logging folder: {}'.format(output_folder))

    # Setup model and data loader
    trainer = MUNIT_Trainer(config, opts)
    trainer.cuda()
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
    test_display_images_a = Variable(torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda())
    test_display_images_b = Variable(torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda())
    train_display_images_a = Variable(torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda())
    train_display_images_b = Variable(torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda())


    # Start training
    iterations = trainer.resume(output_subfolders['models'], hyperparameters=config) if opts.resume else 0


    while True:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            trainer.update_learning_rate()
            images_a, images_b = Variable(images_a.cuda()), Variable(images_b.cuda())

            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                write_loss(iterations, trainer, train_writer)

            if (iterations + 1) % config['print_iter'] == 0:
                logger.info("Iteration: {:08}/{:08} Discriminator Loss: {:.4f} Generator Loss: {:.4f}".format(
                    iterations + 1, max_iter, getattr(trainer, 'loss_dis_total').item(),
                    getattr(trainer, 'loss_gen_total').item()))


            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:
                # Test set images
                image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                write_images(image_outputs[0:4], display_size, '%s/gen_a2b_test_%08d.jpg' % (output_subfolders['images'], iterations + 1))
                write_images(image_outputs[4:8], display_size, '%s/gen_b2a_test_%08d.jpg' % (output_subfolders['images'], iterations + 1))

                # Train set images
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_images(image_outputs[0:4], display_size, '%s/gen_a2b_train_%08d.jpg' % (output_subfolders['images'], iterations + 1))
                write_images(image_outputs[4:8], display_size, '%s/gen_b2a_train_%08d.jpg' % (output_subfolders['images'], iterations + 1))

                # HTML
                write_html(output_folder + "/index.html", iterations + 1, config['image_save_iter'], 'images')

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(output_subfolders['models'], iterations)

            iterations += 1
            if iterations >= max_iter:
                return


if __name__ == '__main__':
    main(sys.argv)
