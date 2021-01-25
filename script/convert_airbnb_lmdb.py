"""
Convert Airbnb lmdb into a format for Vilbert
"""
from typing import List, Dict, Union, Tuple, Any, DefaultDict
from collections import defaultdict
import pickle
import json
from pathlib import Path
from tqdm.auto import tqdm
import lmdb
import argtyped


class Arguments(argtyped.Arguments):
    airbnb: Path
    vilbert: Path
    num_splits: int = 1
    split_id: int = 0
    is_training: bool = False
    train_ratio: float = 0.9
    listing_train: Path = Path("airbnb_listing_train.txt")
    listing_test: Path = Path("airbnb_listing_test.txt")
    captions: Path = Path("airbnb/captions.json")
    map_size: int = int(3e11)
    buffer_size: int = 1000


def load_json(filename):
    with open(filename, "r") as fid:
        return json.load(fid)


class LMDBReader:
    def __init__(self, path: Union[Path, str]):
        self.env = lmdb.open(str(path), readonly=True, readahead=False, lock=False)

        # get keys
        with self.env.begin(write=False, buffers=True) as txn:
            self.keys = [k.decode() for k in pickle.loads(txn.get("keys".encode()))]  # type: ignore

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, key: str):
        if key not in self.keys:
            raise TypeError(f"invalid key: {key}")

        # load from disk
        with self.env.begin(write=False) as txn:
            return pickle.loads(txn.get(key.encode()))  # type: ignore


class LMDBWriter:
    def __init__(self, path: Union[Path, str], map_size: int, buffer_size: int):
        self._env = lmdb.open(str(path), map_size=map_size)
        self._buffer: List[Tuple[bytes, bytes]] = []
        self._buffer_size = buffer_size
        with self._env.begin(write=False) as txn:
            value = txn.get("keys".encode())
            self._keys: List[bytes] = [] if value is None else pickle.loads(value)

    def write(self, key: str, value: bytes):
        if key in self._keys:
            return
        bkey = key.encode()
        self._keys.append(bkey)
        self._buffer.append((bkey, value))
        if len(self._buffer) == self._buffer_size:
            self.flush()

    def flush(self):
        with self._env.begin(write=True) as txn:
            txn.put("keys".encode(), pickle.dumps(self._keys))
            for bkey, value in self._buffer:
                txn.put(bkey, value)

        self._buffer = []

    def __close__(self):
        self.flush()


def _convert_item(key: str, item: Dict[str, Any]) -> List:
    return [
        item["feature"],
        item["bbox"],
        item["num_boxes"],
        item["image_height"],
        item["image_width"],
        key,
    ]


def extract_lmdb(keys: List[str], reader: LMDBReader, output: LMDBWriter):
    for key in tqdm(keys):
        item = reader[key]
        value = _convert_item(key, item)
        output.write(key, pickle.dumps(value))
    output.flush()


if __name__ == "__main__":

    args = Arguments()
    print((args.to_string()))

    args.vilbert.mkdir(exist_ok=True, parents=True)

    reader = LMDBReader(args.airbnb)

    captions_per_key = {
        f"{c['listing_id']}-{c['photo_id']}": c for c in load_json(args.captions)
    }

    keys_per_listing: DefaultDict[str, List[str]] = defaultdict(list)
    for key in reader.keys:
        listing_id, photo_id = list(map(int, key.split("-")))
        keys_per_listing[str(listing_id)].append(key)
    listings = list(keys_per_listing.keys())

    # extract keys for train / val
    train_size = int(len(listings) * args.train_ratio)

    if args.is_training:
        listings = listings[:train_size]
        listing_file = args.listing_train
    else:
        listings = listings[train_size:]
        listing_file = args.listing_test

    with open(listing_file, "w") as fid:
        fid.write("\n".join(listings))

    # gather keys for captions that are not empty
    keys: List[str] = []
    for listing in listings:
        for key in keys_per_listing[listing]:
            if key not in captions_per_key:
                continue
            caption = captions_per_key[key]
            if caption["instructions"][0] == "":
                continue
            keys.append(key)

    # extract only keys for the given split
    per_split = len(keys) // args.num_splits
    if args.split_id == args.num_splits - 1:
        keys = keys[per_split * args.split_id :]
    else:
        keys = keys[per_split * args.split_id : per_split * (1 + args.split_id)]

    output = (
        args.vilbert
        / f"airbnb_{'train' if args.is_training else 'test'}_feat_{'all' if args.num_splits == 1 else f'part_{args.split_id}'}.lmdb"
    )
    print(output)
    writer = LMDBWriter(output, args.map_size, args.buffer_size)
    extract_lmdb(keys, reader, writer)
