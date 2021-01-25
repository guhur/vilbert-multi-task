# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
from dataclasses import dataclass
from typing import DefaultDict, Dict, Tuple, List, Union
import os
from collections import defaultdict
from pathlib import Path
import pickle
import logging
import random
import tensorpack.dataflow as td  # type: ignore
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.distributed as dist
from transformers import BertTokenizer
import msgpack
import msgpack_numpy
import lmdb
import numpy as np


# def _convert_item(key: str, item: Dict) -> Record:
#     photo_id, listing_id = map(int, key.split("-"))
#     image_w = int(item["image_width"])  # pixels
#     image_h = int(item["image_height"])  # pixels
#     num_boxes = int(item["num_boxes"])
#     features = np.frombuffer(item["feature"], dtype=np.float32)
#     features = features.reshape((-1, 2048))  # K x 2048 region features
#     boxes = np.frombuffer(item["bbox"], dtype=np.float32)
#     boxes = boxes.reshape((-1, 4))  # K x 4 region coordinates (x1, y1, x2, y2)
#     cls_prob = np.frombuffer(item["cls_prob"], dtype=np.float32)
#     cls_prob = cls_prob.reshape(
#         (-1, 1601)
#     )  # K x 1601 region object class probabilities
#     return Record(
#         photo_id, listing_id, num_boxes, image_w, image_h, cls_prob, features, boxes,
#     )


# def _get_boxes(record: Record) -> np.ndarray:
#     image_width = record.image_width
#     image_height = record.image_height
#
#     boxes = record.boxes
#     area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     area /= image_width * image_height
#
#     N = len(boxes)
#     output = np.zeros(shape=(N, 5), dtype=np.float32)
#
#     # region encoding
#     output[:, 0] = boxes[:, 0] / image_width
#     output[:, 1] = boxes[:, 1] / image_height
#     output[:, 2] = boxes[:, 2] / image_width
#     output[:, 3] = boxes[:, 3] / image_height
#     output[:, 4] = area
#
#     return output


# def _get_locations(boxes: np.ndarray):
#    """ Convert boxes and orientation information into locations. """
#    N = len(boxes)
#    locations = np.ones(shape=(N, 11), dtype=np.float32)
#
#    # region encoding
#    locations[:, 0] = boxes[:, 0]
#    locations[:, 1] = boxes[:, 1]
#    locations[:, 2] = boxes[:, 2]
#    locations[:, 3] = boxes[:, 3]
#    locations[:, 4] = boxes[:, 4]
#
#    # other indices are used for Room-to-Room
#
#    return locations


msgpack_numpy.patch()

MAX_MSGPACK_LEN = 1000000000

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).reshape(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).reshape(N, 1)

    boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
    query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)

    iw = (
        np.minimum(boxes[:, :, 2], query_boxes[:, :, 2])  # type: ignore
        - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])  # type: ignore
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        np.minimum(boxes[:, :, 3], query_boxes[:, :, 3])  # type: ignore
        - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1])  # type: ignore
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


@dataclass
class InputExample:
    """A single training/test example for the language model."""

    image_feat: np.ndarray
    image_target: np.ndarray
    caption: List[int]
    is_next: int
    image_loc: np.ndarray
    num_boxes: int
    overlaps: np.ndarray  # (N, K) ndarray of overlap between boxes and query_boxes


@dataclass
class InputFeatures:
    """A single set of features of data."""

    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    is_next: np.ndarray
    lm_label_ids: np.ndarray
    image_feat: np.ndarray
    image_loc: np.ndarray
    image_label: np.ndarray
    image_target: np.ndarray
    image_mask: np.ndarray
    masked_label: np.ndarray


class FetchInputFeatures(object):
    """
    Receive a data from the LMDB.
    Load a corresponding (or a random) caption.
    Generate an InputExample. Transform it into an InputFeatures
    """

    def __init__(
        self,
        caption_path: Union[Path, str],
        tokenizer: BertTokenizer,
        bert_model: nn.Module,
        seq_len: int,
        region_len: int,
        data_size: int,
        split: str = "Train",
        visual_target: int = 0,
        visualization: bool = False,
        objective: int = 0,
    ):

        self.split = split
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.visual_target = visual_target
        self.num_caps = data_size
        self.captions = list(json.load(open(caption_path, "r")))
        self.visualization = visualization
        self.objective = objective
        self.bert_model = bert_model
        self.captions_per_photo_id = {
            caption["photo_id"]: caption for caption in self.captions
        }

    def __call__(self, data: List) -> Tuple[np.ndarray, ...]:

        (
            image_feature_wp,
            image_target_wp,
            image_location_wp,
            num_boxes,
            image_h,
            image_w,
            image_id,
        ) = data

        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_location = np.zeros((self.region_len, 5), dtype=np.float32)

        # calculate the IOU here.
        overlaps = iou(image_location_wp, image_location_wp)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        image_target[:num_boxes] = image_target_wp
        image_location[:num_boxes, :4] = image_location_wp

        image_location[:, 4] = (
            (image_location[:, 3] - image_location[:, 1])
            * (image_location[:, 2] - image_location[:, 0])
            / (float(image_w) * float(image_h))
        )

        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        if self.visual_target == 0:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_target)
        else:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_feature)

        listing_id, photo_id = list(map(int, image_id.split("-")))
        caption, label = self.random_cap(photo_id)

        tokens_caption = self.tokenizer.encode(caption)

        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            is_next=label,
            image_loc=image_location,
            num_boxes=num_boxes,
            overlaps=overlaps,
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(
            cur_example, self.seq_len, self.tokenizer, self.region_len
        )

        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_target,
            cur_features.image_label,
            cur_features.image_mask,
            cur_features.masked_label,
            image_id,
        )
        return cur_tensors

    def random_cap(self, photo_id: int) -> Tuple[str, int]:
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, int), sentence, isNextSentence Label
        """
        if not self.visualization and self.objective != 2 and random.random() > 0.5:
            caption = self.get_random_caption()
            label = 1
        else:
            caption = self.captions_per_photo_id[photo_id]["instructions"][0]
            label = 0

        return caption, label

    def get_random_caption(self):
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # add the hard negative mining objective here.
        rand_doc_idx = random.randint(0, self.num_caps - 1)
        caption = self.captions[rand_doc_idx]

        return caption

    def convert_example_to_features(
        self, example, max_seq_length, tokenizer, max_region_length
    ):
        """"""
        image_feat = example.image_feat
        tokens = example.caption
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)
        is_next = example.is_next
        overlaps = example.overlaps

        self._truncate_seq_pair(tokens, max_seq_length - 2)

        tokens, tokens_label = self.random_word(tokens, tokenizer, is_next)
        image_feat, image_loc, image_label, masked_label = self.random_region(
            image_feat, image_loc, num_boxes, is_next, overlaps
        )

        # concatenate lm labels and account for CLS, SEP, SEP
        lm_label_ids = [-1] + tokens_label + [-1]
        tokens = tokenizer.add_special_tokens_single_sentence(tokens)
        segment_ids = [0] * len(tokens)

        input_ids = tokens  # tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            is_next=np.array(example.is_next),
            image_feat=image_feat,
            image_target=image_target,
            image_loc=image_loc,
            image_label=np.array(image_label),
            image_mask=np.array(image_mask),
            masked_label=masked_label,
        )
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

    def random_word(self, tokens, tokenizer, is_next):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability

            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # not sample mask
            if prob < 0.15 and (not self.visualization):
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(tokenizer))  # type: ignore
                    # torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes, is_next, overlaps):
        """"""
        output_label = []
        masked_label = np.zeros((image_feat.shape[0]))

        for i in range(num_boxes):
            prob = random.random()
            # mask token with 15% probability

            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # if the target is inaligned mask, then not sample mask
            if prob < 0.15 and not self.visualization:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0
                # mask the overlap regions into zeros
                masked_label = np.logical_or(masked_label, overlaps[i] > 0.4)

                # 10% randomly change token to random token
                # elif prob < 0.9:
                # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, image_loc, output_label, masked_label


class AirbnbLoaderTrain(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the
            GPU, which is faster).
    """

    def __init__(
        self,
        corpus_path,
        tokenizer,
        bert_model,
        seq_len,
        encoding="utf-8",
        visual_target=0,
        hard_negative=False,
        batch_size=512,
        shuffle=False,
        num_workers=25,
        cache=10000,
        drop_last=False,
        cuda=False,
        local_rank=-1,
        objective=0,
        visualization=False,
    ):
        if dist.is_available() and local_rank != -1:
            rank = dist.get_rank()
            lmdb_file = Path(corpus_path) / f"airbnb_train_feat_part_{str(rank)}.lmdb"
        else:
            lmdb_file = Path(corpus_path) / "airbnb_train_feat_all.lmdb"
            print(("Loading from %s" % lmdb_file))

        ds = td.LMDBSerializer.load(str(lmdb_file), shuffle=False)
        self.num_dataset = len(ds)
        ds = td.LocallyShuffleData(ds, cache)
        caption_path = os.path.join(corpus_path, "captions.json")

        preprocess_function = FetchInputFeatures(
            caption_path,
            tokenizer,
            bert_model,
            seq_len,
            36,
            self.num_dataset,
            visual_target=visual_target,
            objective=objective,
        )

        ds = td.PrefetchData(ds, 5000, 1)
        ds = td.MapData(ds, preprocess_function)
        # self.ds = td.PrefetchData(ds, 1)
        ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        # self.ds = ds
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):

        for batch in self.ds.get_data():
            (
                input_ids,
                input_mask,
                segment_ids,
                lm_label_ids,
                is_next,
                image_feat,
                image_loc,
                image_target,
                image_label,
                image_mask,
                masked_label,
                image_id,
            ) = batch

            batch_size = input_ids.shape[0]

            sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
            sum_count[sum_count == 0] = 1  # type: ignore
            g_image_feat = np.sum(image_feat, axis=1) / sum_count
            image_feat = np.concatenate(
                [np.expand_dims(g_image_feat, axis=1), image_feat], axis=1
            )
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(
                np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0
            )
            image_loc = np.concatenate(
                [np.expand_dims(g_image_loc, axis=1), image_loc], axis=1
            )

            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (
                input_ids,
                input_mask,
                segment_ids,
                lm_label_ids,
                is_next,
                image_feat,
                image_loc,
                image_target,
                image_label,
                image_mask,
            )

            yield tuple([torch.tensor(data) for data in batch] + [image_id])

    def __len__(self):
        return self.ds.size()


class AirbnbLoaderVal(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the
            GPU, which is faster).
    """

    def __init__(
        self,
        corpus_path,
        tokenizer,
        bert_model,
        seq_len,
        encoding="utf-8",
        visual_target=0,
        batch_size=512,
        shuffle=False,
        num_workers=25,
        cache=5000,
        drop_last=False,
        cuda=False,
        objective=0,
        visualization=False,
    ):

        lmdb_file = Path(corpus_path) / "airbnb_test_feat_all.lmdb"
        caption_path = Path(corpus_path) / "captions.json"
        print(("Loading from %s" % lmdb_file))

        ds = td.LMDBSerializer.load(str(lmdb_file), shuffle=False)
        self.num_dataset = len(ds)
        preprocess_function = FetchInputFeatures(
            caption_path,
            tokenizer,
            bert_model,
            seq_len,
            36,
            self.num_dataset,
            visual_target=visual_target,
            visualization=visualization,
            objective=objective,
        )

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):
        for batch in self.ds.get_data():
            (
                input_ids,
                input_mask,
                segment_ids,
                lm_label_ids,
                is_next,
                image_feat,
                image_loc,
                image_target,
                image_label,
                image_mask,
                masked_label,
                image_id,
            ) = batch

            batch_size = input_ids.shape[0]
            sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
            sum_count[sum_count == 0] = 1  # type: ignore
            g_image_feat = np.sum(image_feat, axis=1) / sum_count
            image_feat = np.concatenate(
                [np.expand_dims(g_image_feat, axis=1), image_feat], axis=1
            )
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(
                np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0
            )
            image_loc = np.concatenate(
                [np.expand_dims(g_image_loc, axis=1), image_loc], axis=1
            )

            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (
                input_ids,
                input_mask,
                segment_ids,
                lm_label_ids,
                is_next,
                image_feat,
                image_loc,
                image_target,
                image_label,
                image_mask,
            )

            yield tuple([torch.tensor(data) for data in batch] + [image_id])

    def __len__(self):
        return self.ds.size()
