"""
Mask R-CNN
Configurations and data loading code for MS miotcd.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained miotcd weights
    python3 miotcd.py train --dataset=/path/to/miotcd/ --model=miotcd

    # Train a new model starting from ImageNet weights
    python3 miotcd.py train --dataset=/path/to/miotcd/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 miotcd.py train --dataset=/path/to/miotcd/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 miotcd.py train --dataset=/path/to/miotcd/ --model=last

    # Run miotcd evaluatoin on the last model you trained
    python3 miotcd.py evaluate --dataset=/path/to/miotcd/ --model=last
"""
import csv
import json
import os
import time
from collections import OrderedDict

import cv2
import h5py
import numpy as np

import model as modellib
import utils
from config import Config
from PIL import Image

pjoin = os.path.join
# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
miotcd_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class miotcdConfig(Config):
    """Configuration for training on MS miotcd.
    Derives from the base Config class and overrides values specific
    to the miotcd dataset.
    """
    # Give the configuration a recognizable name
    NAME = "miotcd"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # miotcd has 10 classes


############################################################
#  Dataset
############################################################

class JsonHandler():
    classes = ["articulated_truck", "bicycle", "bus", "car", "motorcycle",
               "non-motorized_vehicle", "motorized_vehicle",
               "pedestrian", "pickup_truck", "single_unit_truck", "work_van"]

    def __init__(self, path):
        self.path = path
        self.json = 'internal_cvpr2016.json'
        file = pjoin(self.path, self.json)
        jsonfile = json.load(open(file, "r"), object_pairs_hook=OrderedDict)
        self.datas = OrderedDict([self.__handle_json_inner(v) for v in jsonfile.values()])
        self.gt_train = 'gt_train.csv'
        self.gt_test = 'gt_test.csv'
        with open(pjoin(self.path, self.gt_train)) as f:
            self.gt_train = list(set([pjoin(self.path, 'images', x[0] + '.jpg') for x in csv.reader(f)]))
        with open(pjoin(self.path, self.gt_test)) as f:
            self.gt_test = set([pjoin(self.path, 'images', x[0] + '.jpg') for x in csv.reader(f)])

        self.gt_train, self.gt_val = np.split(self.gt_train, [int(0.8*len(self.gt_train))])

    def __handle_json_inner(self, value):
        annotation = value['annotations']
        polygons = [
            (x['classification'], x['outline_xy'])
            for x in annotation]
        jpgfile = value['external_id'] + '.jpg'
        return pjoin(self.path, 'images', jpgfile), polygons


class miotcdDataset(utils.Dataset):
    def load_miotcd(self,jsonHandler,phase, return_miotcd=False):
        """Load a subset of the miotcd dataset.
        dataset_dir: The root directory of the miotcd dataset.
        subset: What to load (train, val, minival, val35k)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_miotcd: If True, returns the miotcd object.
        """
        # Load all classes or a subset?
        # NOTE NO SUBSET
        self.jsonHandler = jsonHandler
        assert phase in ['train', 'val', 'test']
        class_ids = jsonHandler.classes

        # All images or a subset?
        if phase == 'train':
            image_ids = jsonHandler.gt_train
        elif phase == 'val':
            image_ids = jsonHandler.gt_val
        else:
            image_ids = jsonHandler.gt_test

        # Add classes
        for i,cls  in enumerate(class_ids):
            self.add_class("miotcd", i, cls)

        if not os.path.exists('shapes.h5'):
            sh_file = h5py.File('shapes.h5')
            for i in image_ids:
                k = Image.open(i)
                sh_file.create_dataset(i.split('/')[-1],data=np.array([k.width,k.height]))
            sh_file.close()

        sh_file = h5py.File('shapes.h5')




        # Add images
        for i in image_ids:
            self.add_image(
                "miotcd", image_id=i,
                path=i,
                width=sh_file[i.split('/')[-1]][0],
                height=sh_file[i.split('/')[-1]][1],
                annotations=self.jsonHandler.datas[i])
        if return_miotcd:
            return miotcd

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a miotcd image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "miotcd":
            return super(self.__class__).load_mask(image_id)

        instance_masks = []
        class_ids = []
        w,h = image_info['width'],image_info['height']
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for cls,outline in annotations:
            mask = np.zeros([h,w,1])
            cv2.drawContours(mask,outline,-1,1,-1)
            cl = self.jsonHandler.classes.index(cls)
            instance_masks.append(mask)
            class_ids.append(cl)

        instance_masks = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)

        return instance_masks, class_ids



############################################################
#  miotcd Evaluation
############################################################

def build_miotcd_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match miotcd specs in http://miotcddataset.org/#format
    """
    # If no results, return an empty list
    raise NotImplemented
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "miotcd"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_miotcd(dataset, miotcd, eval_type="bbox", limit=0):
    """Runs official miotcd evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick miotcd images from the dataset
    raise NotImplemented
    image_ids = dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding miotcd image IDs.
    miotcd_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to miotcd format
        image_results = build_miotcd_results(dataset, miotcd_image_ids[i:i + 1],
                                             r["rois"], r["class_ids"],
                                             r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    miotcd_results = miotcd.loadRes(results)

    # Evaluate
    miotcdEval = miotcdeval(miotcd, miotcd_results, eval_type)
    miotcdEval.params.imgIds = miotcd_image_ids
    miotcdEval.evaluate()
    miotcdEval.accumulate()
    miotcdEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS miotcd.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS miotcd")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/miotcd/",
                        help='Directory of the MS-miotcd dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'miotcd'")
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)

    # Configurations
    if args.command == "train":
        config = miotcdConfig()
    else:
        class InferenceConfig(miotcdConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.print()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=MODEL_DIR)

    # Select weights file to load
    if args.model.lower() == "miotcd":
        model_path = miotcd_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = miotcdDataset()
        dataset_train.load_miotcd(args.dataset, "train")
        dataset_train.load_miotcd(args.dataset, "val35k")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = miotcdDataset()
        dataset_val.load_miotcd(args.dataset, "minival")
        dataset_val.prepare()

        # This training schedule is an example. Update to fit your needs.

        # Training - Stage 1
        # Adjust epochs and layers as needed
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Training Resnet layer 4+")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=100,
                    layers='4+')

        # Training - Stage 3
        # Finetune layers from ResNet stage 3 and up
        print("Training Resnet layer 3+")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 100,
                    epochs=200,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = miotcdDataset()
        miotcd = dataset_val.load_miotcd(args.dataset, "minival", return_miotcd=True)
        dataset_val.prepare()

        # TODO: evaluating on 500 images. Set to 0 to evaluate on all images.
        evaluate_miotcd(dataset_val, miotcd, "bbox", limit=500)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
