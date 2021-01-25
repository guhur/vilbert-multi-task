import base64
from dataclasses import dataclass
from typing import DefaultDict, Dict, Tuple, List, Union
from collections import defaultdict
from pathlib import Path
import pickle
import lmdb
import numpy as np


@dataclass
class Record:
    photo_id: int
    listing_id: int
    num_boxes: int
    image_width: int
    image_height: int
    cls_prob: np.ndarray
    features: np.ndarray
    boxes: np.ndarray


def _convert_item(key: str, item: Dict) -> Record:
    photo_id, listing_id = list(map(int, key.split("-")))
    image_w = int(item["image_width"])  # pixels
    image_h = int(item["image_height"])  # pixels
    num_boxes = int(item["num_boxes"])
    features = np.frombuffer(item["feature"], dtype=np.float32)
    features = features.reshape((-1, 2048))  # K x 2048 region features
    boxes = np.frombuffer(item["bbox"], dtype=np.float32)
    boxes = boxes.reshape((-1, 4))  # K x 4 region coordinates (x1, y1, x2, y2)
    cls_prob = np.frombuffer(item["cls_prob"], dtype=np.float32)
    cls_prob = cls_prob.reshape(
        (-1, 1601)
    )  # K x 1601 region object class probabilities
    return Record(
        photo_id,
        listing_id,
        num_boxes,
        image_w,
        image_h,
        cls_prob,
        features,
        boxes,
    )


def _get_boxes(record: Record) -> np.ndarray:
    image_width = record.image_width
    image_height = record.image_height

    boxes = record.boxes
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area /= image_width * image_height

    N = len(boxes)
    output = np.zeros(shape=(N, 5), dtype=np.float32)

    # region encoding
    output[:, 0] = boxes[:, 0] / image_width
    output[:, 1] = boxes[:, 1] / image_height
    output[:, 2] = boxes[:, 2] / image_width
    output[:, 3] = boxes[:, 3] / image_height
    output[:, 4] = area

    return output


def _get_locations(boxes: np.ndarray):
    """ Convert boxes and orientation information into locations. """
    N = len(boxes)
    locations = np.ones(shape=(N, 11), dtype=np.float32)

    # region encoding
    locations[:, 0] = boxes[:, 0]
    locations[:, 1] = boxes[:, 1]
    locations[:, 2] = boxes[:, 2]
    locations[:, 3] = boxes[:, 3]
    locations[:, 4] = boxes[:, 4]

    # other indices are used for Room-to-Room

    return locations


class BnBFeaturesReader:
    def __init__(self, path: Union[Path, str]):
        # open database
        self.env = lmdb.open(
            str(path),
            readonly=True,
            readahead=False,
            max_readers=20,
            lock=False,
            map_size=int(1e9),
        )

        # get keys
        with self.env.begin(write=False, buffers=True) as txn:
            self.keys = [k.decode() for k in pickle.loads(txn.get("keys".encode()))]  # type: ignore

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, query: Tuple):
        key: str = query[0]
        if key not in self.keys:
            raise TypeError(f"invalid key: {key}")

        index = self.keys.index(key)

        # load from disk
        with self.env.begin(write=False) as txn:
            item = pickle.loads(txn.get(key.encode()))  # type: ignore
            record = _convert_item(key, item)

            boxes = _get_boxes(record)
            probs = record.cls_prob
            features = record.features

        locations = _get_locations(boxes)

        if features is None:
            raise RuntimeError("Features could not be correctly read")

        # add a global feature vector
        g_feature = features.mean(axis=0, keepdims=True)
        g_location = np.array(
            [
                [
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                ]
            ]
        )
        g_prob = np.ones(shape=(1, 1601)) / 1601  # uniform probability

        features = np.concatenate([g_feature, features], axis=0)
        locations = np.concatenate([g_location, locations], axis=0)
        probs = np.concatenate([g_prob, probs], axis=0)

        return features, locations, probs
