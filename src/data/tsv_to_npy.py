import csv
import os
import sys
import time
import numpy as np
import base64
from argparse import ArgumentParser


def tsv_to_npy(path, save_path):
    """Load object features from tsv file.
    :param path: The path to the tsv file.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """

    if not os.path.exists(os.path.join(save_path, 'features')):
        os.mkdir(os.path.join(save_path, 'features'))
        os.mkdir(os.path.join(save_path, 'box'))

    csv.field_size_limit(sys.maxsize)
    FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
                  "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % path)
    with open(path) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):
            boxes = int(item['num_boxes'])

            im_id = item['img_id']
            loc = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32)
            loc = loc.reshape(boxes, 4)
            loc.setflags(write=False)
            array = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32)
            array = array.reshape(boxes, -1)
            array.setflags(write=False)

            np.save('%s/features/%s.npy' % (save_path, im_id), array)
            np.save('%s/box/%s.npy' % (save_path, im_id), loc)
    elapsed_time = time.time() - start_time
    print("Converted Image Features in file %s in %d seconds." % (path, elapsed_time))


arg_parser = ArgumentParser(description='convert tsv file to numpy features')
arg_parser.add_argument('-p', default='', type=str, help='path to tsv file')
arg_parser.add_argument('-s', default='', type=str, help='save directory')

args = arg_parser.parse_args()

tsv_to_npy(args.p, args.s)
