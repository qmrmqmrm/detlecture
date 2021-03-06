import os.path as op
import parameter_pool as params
import numpy as np


class Config:
    class Paths:
        RESULT_ROOT = "/home/ri-1080/workspace/detlecture/result"
        TFRECORD = op.join(RESULT_ROOT, "tfrecord")
        CHECK_POINT = op.join(RESULT_ROOT, "ckpt")

    class Datasets:
        # specific dataset configs MUST have the same items
        class Kitti:
            NAME = "kitti"
            PATH = "/home/ri-1080/workspace/detlecture/data/kitti_detection"
            CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck", "Cyclist"]
            CATEGORY_REMAP = {"Pedestrian": "Person", "Cyclist": "Bicycle"}
            INPUT_RESOLUTION = (256, 832)   # (4,13) * 64
            CROP_TLBR = [0, 0, 0, 0]        # crop [top, left, bottom, right] or [y1 x1 y2 x2]

        DATASET_CONFIGS = {"kitti": Kitti}
        TARGET_DATASET = "kitti"

        @classmethod
        def get_dataset_config(cls, dataset):
            return cls.DATASET_CONFIGS[dataset]

    class Tfrdata:
        DATASETS_FOR_TFRECORD = {"kitti": ("train", "val")}
        MAX_BBOX_PER_IMAGE = 20
        CATEGORY_NAMES = ["Person", "Car", "Van", "Bicycle"]
        SHARD_SIZE = 2000
        ANCHORS_PIXEL = None  # assigned by set_anchors()

        @classmethod
        def set_anchors(cls):
            basic_anchor = params.Anchor.COCO_YOLOv3
            target_dataset = Config.Datasets.TARGET_DATASET
            dataset_cfg = Config.Datasets.get_dataset_config(target_dataset)
            input_resolution = np.array(dataset_cfg.INPUT_RESOLUTION, dtype=np.float32)
            anchor_resolution = np.array(params.Anchor.COCO_RESOLUTION, dtype=np.float32)
            scale = np.min(input_resolution / anchor_resolution)
            Config.Tfrdata.ANCHORS_PIXEL = np.around(basic_anchor * scale, 1)
            print("[set_anchors] anchors in pixel:\n", Config.Tfrdata.ANCHORS_PIXEL)

    class Model:
        class Output:
            FEATURE_SCALES = {"feature_s": 8, "feature_m": 16, "feature_l": 32}
            FEATURE_ORDER = ["feature_s", "feature_m", "feature_l"]
            NUM_ANCHORS_PER_SCALE = 3
            OUT_CHANNELS = 0                    # assigned by set_out_channel()
            OUT_COMPOSITION = ()                # assigned by set_out_channel()

            @classmethod
            def set_out_channel(cls):
                num_cats = len(Config.Tfrdata.CATEGORY_NAMES)
                Config.Model.Output.OUT_COMPOSITION = [('bbox', 4), ('object', 1), ('category', num_cats)]
                Config.Model.Output.OUT_CHANNELS = sum([val for key, val in Config.Model.Output.OUT_COMPOSITION])

        class Structure:
            BACKBONE = ["Darknet53", "Resnet50"][1]
            HEAD = "FPN"
            BACKBONE_CONV_ARGS = {"activation": "leaky_relu", "scope": "back"}
            HEAD_CONV_ARGS = {"activation": "leaky_relu", "scope": "head"}

    class Train:
        CKPT_NAME = "yolo2_resnet50"
        MODE = ["eager", "graph"][0]
        BATCH_SIZE = 4
        TRAINING_PLAN = params.TrainingPlan.KITTI_SIMPLE

    @classmethod
    def summary(cls):
        # return dict of important parameters
        pass

    @classmethod
    def get_img_shape(cls, code="HW", dataset="kitti", scale_div=1):
        dataset_cfg = cls.Datasets.get_dataset_config(dataset)
        imsize = dataset_cfg.INPUT_RESOLUTION
        code = code.upper()
        if code == "H":
            return imsize[0] // scale_div
        elif code == "W":
            return imsize[1] // scale_div
        elif code == "HW":
            return imsize[0] // scale_div, imsize[1] // scale_div
        elif code == "WH":
            return imsize[1] // scale_div, imsize[0] // scale_div
        elif code == "HWC":
            return imsize[0] // scale_div, imsize[1] // scale_div, 3
        elif code == "BHWC":
            return cls.Train.BATCH_SIZE, imsize[0] // scale_div, imsize[1] // scale_div, 3
        else:
            assert 0, f"Invalid code: {code}"

    @classmethod
    def get_valid_category_mask(cls, dataset="kitti"):
        """
        :param dataset: dataset name
        :return: binary mask e.g. when
            Tfrdata.CATEGORY_NAMES = ["Person", "Car", "Van", "Bicycle"] and
            Dataset.CATEGORIES_TO_USE = ["Pedestrian", "Car", "Van", "Truck"]
            Dataset.CATEGORY_REMAP = {"Pedestrian": "Person"}
            this function returns [1 1 1 0] because ["Person", "Car", "Van"] are included in dataset categories
            but "Bicycle" is not
        """
        dataset_cfg = cls.Datasets.get_dataset_config(dataset)
        renamed_categories = [dataset_cfg.CATEGORY_REMAP[categ] if categ in dataset_cfg.CATEGORY_REMAP else categ
                              for categ in dataset_cfg.CATEGORIES_TO_USE]

        mask = np.zeros((len(cls.Tfrdata.CATEGORY_NAMES), ), dtype=np.int32)
        for categ in renamed_categories:
            if categ in cls.Tfrdata.CATEGORY_NAMES:
                index = cls.Tfrdata.CATEGORY_NAMES.index(categ)
                mask[index] = 1
        return mask



Config.Tfrdata.set_anchors()
Config.Model.Output.set_out_channel()
