import datetime
import math
import os
import gc
import time

import numpy as np
import torch
from torch.autograd import Variable

import utils
import tqdm

class Extractor(object):

    def __init__(self, cuda, model, val_loader, log_file, feature_dir, flatten_feature=True, print_freq=1):
        """
        :param cuda:
        :param model:
        :param val_loader:
        :param log_file: log file name. logs are appended to this file.
        :param feature_dir:
        :param flatten_feature:
        :param print_freq:
        """
        self.cuda = cuda

        self.model = model
        self.val_loader = val_loader
        self.log_file = log_file
        self.feature_dir = feature_dir
        self.flatten_feature = flatten_feature
        self.print_freq = print_freq

        self.timestamp_start = datetime.datetime.now()


    def print_log(self, log_str):
        with open(self.log_file, 'a') as f:
            f.write(log_str + '\n')


    def extract(self):
        batch_time = utils.AverageMeter()

        self.model.eval()
        end = time.time()

        for batch_idx, (imgs, target, img_files, class_ids) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Extract', ncols=80, leave=False):

            gc.collect()

            if self.cuda:
                imgs = imgs.cuda()
            imgs = Variable(imgs, volatile=True)
            output = self.model(imgs)  # N C H W torch.Size([1, 1, 401, 600])
            if self.flatten_feature:
                output = output.view(output.size(0), -1)
            output = output.data.cpu().numpy()

            assert output.shape[0] == len(img_files)
            for i, img_file in enumerate(img_files):
                base_name = os.path.splitext(img_file)[0]
                feature_file = os.path.join(self.feature_dir, base_name + ".npy")
                utils.create_dir(os.path.dirname(feature_file))
                np.save(feature_file, output[i])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % self.print_freq == 0:
                log_str = 'Extract: [{0}/{1}]\tTime: {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(self.val_loader), batch_time=batch_time)
                print(log_str)
                self.print_log(log_str)

