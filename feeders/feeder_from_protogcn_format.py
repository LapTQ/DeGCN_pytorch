import numpy as np
from torch.utils.data import Dataset
from feeders import tools
import numpy as np
from torch.utils.data import Dataset
import pickle


def normalize_by_minmax(seq_kpts):
    kmin = np.min(seq_kpts, axis=2, keepdims=True)
    kmax = np.max(seq_kpts, axis=2, keepdims=True)
    seq_kpts = (seq_kpts - kmin) / (kmax - kmin + 1e-8)

    return seq_kpts


def select_keypoints(seq_kpts, layout):
    if layout == "coco":
        return seq_kpts
    elif layout == "coco_onlyhand":
        return seq_kpts[:, :, 5:11, :]
    elif layout == "coco_headless":
        return seq_kpts[:, :, 5:17, :]


def rescale_to_neg1_pos1(seq_kpts):
    return seq_kpts * 2 - 1


class JointToBone:

    def __init__(self, dataset, target="keypoint"):
        self.dataset = dataset
        self.target = target
        if self.dataset not in [
            "nturgb+d",
            "openpose",
            "openpose_new",
            "coco",
            "coco_new",
            "coco_headless",
            "coco_onlyhand",
        ]:
            raise ValueError(f"The dataset type {self.dataset} is not supported")
        if self.dataset == "nturgb+d":
            self.pairs = (
                (0, 1),
                (1, 20),
                (2, 20),
                (3, 2),
                (4, 20),
                (5, 4),
                (6, 5),
                (7, 6),
                (8, 20),
                (9, 8),
                (10, 9),
                (11, 10),
                (12, 0),
                (13, 12),
                (14, 13),
                (15, 14),
                (16, 0),
                (17, 16),
                (18, 17),
                (19, 18),
                (21, 22),
                (20, 20),
                (22, 7),
                (23, 24),
                (24, 11),
            )
        elif self.dataset == "openpose":
            self.pairs = (
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 2),
                (4, 3),
                (5, 1),
                (6, 5),
                (7, 6),
                (8, 2),
                (9, 8),
                (10, 9),
                (11, 5),
                (12, 11),
                (13, 12),
                (14, 0),
                (15, 0),
                (16, 14),
                (17, 15),
            )
        elif self.dataset == "openpose_new":
            self.pairs = (
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 2),
                (4, 3),
                (5, 1),
                (6, 5),
                (7, 6),
                (8, 18),
                (9, 8),
                (10, 9),
                (11, 18),
                (12, 11),
                (13, 12),
                (14, 0),
                (15, 0),
                (16, 14),
                (17, 15),
                (18, 19),
                (19, 1),
            )
        elif self.dataset == "coco":
            self.pairs = (
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 1),
                (4, 2),
                (5, 0),
                (6, 0),
                (7, 5),
                (8, 6),
                (9, 7),
                (10, 8),
                (11, 0),
                (12, 0),
                (13, 11),
                (14, 12),
                (15, 13),
                (16, 14),
            )
        elif self.dataset == "coco_new":
            self.pairs = (
                (0, 19),
                (1, 0),
                (2, 0),
                (3, 1),
                (4, 2),
                (5, 19),
                (6, 19),
                (7, 5),
                (8, 6),
                (9, 7),
                (10, 8),
                (11, 17),
                (12, 17),
                (13, 11),
                (14, 12),
                (15, 13),
                (16, 14),
                (17, 18),
                (18, 19),
                (19, 19),
            )
        elif self.dataset == "coco_headless":
            self.pairs = (
                (0, 1),
                (1, 0),
                (2, 0),
                (3, 2),
                (4, 2),
                (5, 3),
                (6, 0),
                (7, 1),
                (8, 6),
                (9, 7),
                (10, 8),
                (11, 9),
            )
        elif self.dataset == "coco_onlyhand":
            self.pairs = (
                (0, 1),
                (1, 0),
                (2, 0),
                (3, 2),
                (4, 2),
                (5, 3),
            )

    def __call__(self, results):

        keypoint = results["keypoint"]
        M, T, V, C = keypoint.shape
        bone = np.zeros((M, T, V, C), dtype=np.float32)

        assert C in [2, 3]
        for v1, v2 in self.pairs:
            bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
            if C == 3 and self.dataset in [
                "openpose",
                "openpose_new",
                "coco",
                "coco_new",
                "handmp",
            ]:
                score = (keypoint[..., v1, 2] + keypoint[..., v2, 2]) / 2
                bone[..., v1, 2] = score

        results[self.target] = bone
        return results


class JointToKB:

    def __init__(self, dataset="nturgb+d", target="keypoint"):
        self.dataset = dataset
        self.target = target
        if self.dataset not in [
            "nturgb+d",
            "openpose",
            "openpose_new",
            "coco",
            "coco_new",
            "coco_headless",
            "coco_onlyhand",
        ]:
            raise ValueError(f"The dataset type {self.dataset} is not supported")
        if self.dataset == "nturgb+d":
            self.pairs = (
                (0, 20),
                (1, 1),
                (2, 2),
                (3, 20),
                (4, 4),
                (5, 20),
                (6, 4),
                (7, 5),
                (8, 8),
                (9, 20),
                (10, 8),
                (11, 9),
                (12, 1),
                (13, 0),
                (14, 12),
                (15, 13),
                (16, 1),
                (17, 0),
                (18, 16),
                (19, 17),
                (21, 7),
                (20, 20),
                (22, 6),
                (23, 11),
                (24, 10),
            )
        elif self.dataset == "openpose":
            self.pairs = (
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 1),
                (4, 2),
                (5, 5),
                (6, 1),
                (7, 5),
                (8, 1),
                (9, 2),
                (10, 8),
                (11, 1),
                (12, 5),
                (13, 11),
                (14, 1),
                (15, 1),
                (16, 0),
                (17, 0),
            )
        elif self.dataset == "openpose_new":
            self.pairs = (
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 1),
                (4, 2),
                (5, 5),
                (6, 1),
                (7, 5),
                (8, 19),
                (9, 18),
                (10, 8),
                (11, 19),
                (12, 18),
                (13, 11),
                (14, 1),
                (15, 1),
                (16, 0),
                (17, 0),
                (18, 1),
                (19, 19),
            )
        elif self.dataset == "coco":
            self.pairs = (
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 0),
                (4, 0),
                (5, 5),
                (6, 6),
                (7, 0),
                (8, 0),
                (9, 5),
                (10, 6),
                (11, 11),
                (12, 12),
                (13, 0),
                (14, 0),
                (15, 11),
                (16, 12),
            )
        elif self.dataset == "coco_new":
            self.pairs = (
                (0, 0),
                (1, 19),
                (2, 19),
                (3, 0),
                (4, 0),
                (5, 5),
                (6, 6),
                (7, 19),
                (8, 19),
                (9, 5),
                (10, 6),
                (11, 18),
                (12, 18),
                (13, 17),
                (14, 17),
                (15, 11),
                (16, 12),
                (17, 19),
                (18, 18),
                (19, 19),
            )
        elif self.dataset == "coco_headless":
            self.pairs = (
                (0, 0),
                (1, 1),
                (2, 1),
                (3, 0),
                (4, 0),
                (5, 1),
                (6, 6),
                (7, 7),
                (8, 0),
                (9, 1),
                (10, 6),
                (11, 7),
            )
        elif self.dataset == "coco_onlyhand":
            self.pairs = (
                (0, 0),
                (1, 1),
                (2, 1),
                (3, 0),
                (4, 0),
                (5, 1),
            )

    def __call__(self, results):

        keypoint = results["keypoint"]
        M, T, V, C = keypoint.shape
        bone = np.zeros((M, T, V, C), dtype=np.float32)

        assert C in [2, 3]
        for v1, v2 in self.pairs:
            bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
            if C == 3 and self.dataset in ["openpose", "coco", "coco_headless", "coco_onlyhand"]:
                score = (keypoint[..., v1, 2] + keypoint[..., v2, 2]) / 2
                bone[..., v1, 2] = score

        results[self.target] = bone
        return results

class ToMotion:

    def __init__(self, dataset="nturgb+d", source="keypoint", target="motion"):
        self.dataset = dataset
        self.source = source
        self.target = target

    def __call__(self, results):
        data = results[self.source]
        M, T, V, C = data.shape
        motion = np.zeros_like(data)

        assert C in [2, 3]
        motion[:, : T - 1] = np.diff(data, axis=1)
        if C == 3 and self.dataset in ["openpose", "coco"]:
            score = (data[:, : T - 1, :, 2] + data[:, 1:, :, 2]) / 2
            motion[:, : T - 1, :, 2] = score

        results[self.target] = motion

        return results


class MergeSkeFeat:
    def __init__(self, feat_list=["keypoint"], target="keypoint", axis=-1):
        """Merge different feats (ndarray) by concatenate them in the last axis."""

        self.feat_list = feat_list
        self.target = target
        self.axis = axis

    def __call__(self, results):
        feats = []
        for name in self.feat_list:
            feats.append(results.pop(name))
        feats = np.concatenate(feats, axis=self.axis)
        results[self.target] = feats
        return results

class Rename:
    """Rename the key in results.

    Args:
        mapping (dict): The keys in results that need to be renamed. The key of
            the dict is the original name, while the value is the new name. If
            the original name not found in results, do nothing.
            Default: dict().
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, results):
        for key, value in self.mapping.items():
            if key in results:
                assert isinstance(key, str) and isinstance(value, str)
                assert value not in results, ('the new name already exists in '
                                              'results')
                results[value] = results[key]
                results.pop(key)
        return results

class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        # assert isinstance(transforms, Sequence)
        self.transforms = transforms
        
    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class GenSkeFeat:
    def __init__(self, dataset="nturgb+d", feats=["j"], axis=-1):
        self.dataset = dataset
        self.feats = feats
        self.axis = axis
        ops = []
        if "b" in feats or "bm" in feats:
            ops.append(JointToBone(dataset=dataset, target="b"))
        if "k" in feats or "km" in feats:
            ops.append(JointToKB(dataset=dataset, target="k"))
        ops.append(Rename({"keypoint": "j"}))
        if "jm" in feats:
            ops.append(ToMotion(dataset=dataset, source="j", target="jm"))
        if "bm" in feats:
            ops.append(ToMotion(dataset=dataset, source="b", target="bm"))
        if "km" in feats:
            ops.append(ToMotion(dataset=dataset, source="k", target="km"))
        ops.append(MergeSkeFeat(feat_list=feats, axis=axis))
        self.ops = Compose(ops)

    def __call__(self, results):
        if "keypoint_score" in results and "keypoint" in results:
            assert self.dataset != "nturgb+d"
            assert (
                results["keypoint"].shape[-1] == 2
            ), "Only 2D keypoints have keypoint_score. "
            keypoint = results.pop("keypoint")
            keypoint_score = results.pop("keypoint_score")
            results["keypoint"] = np.concatenate(
                [keypoint, keypoint_score[..., None]], -1
            )
        return self.ops(results)

class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        split,
        window_size=64,
        random_rot=False,
        layout="coco",
        debug=False,
        class_map=None,
        ske_feat="j",
    ):
        """
        :param data_path:
        :param label_path:
        :param random_rot: rotate skeleton around xyz axis
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.data_path = data_path
        self.split = split
        self.window_size = window_size
        self.random_rot = random_rot
        self.layout = layout
        self.class_map = np.array(class_map) if class_map is not None else None
        self.load_data()
        self.ske_feat = ske_feat

        self.gen_ske_feat = {
            "j": GenSkeFeat(dataset=self.layout, feats=["j"]),
            "b": GenSkeFeat(dataset=self.layout, feats=["b"]),
            "k": GenSkeFeat(dataset=self.layout, feats=["k"]),
            "jm": GenSkeFeat(dataset=self.layout, feats=["jm"]),
            "bm": GenSkeFeat(dataset=self.layout, feats=["bm"]),
            "km": GenSkeFeat(dataset=self.layout, feats=["km"]),
        }

    def load_data(self):

        with open(self.data_path, "rb") as f:
            data = pickle.load(f)["annotations"]
        self.data = np.concatenate(
            [d["keypoint"] for d in data], axis=0
        )  # (num_seq, seq_len, num_kpt, kpt_dim)
        self.label = np.array([d["label"] for d in data])  # (num_seq,)
        if self.split == "train":
            self.sample_name = ["train_" + str(i) for i in range(len(self.data))]
        else:
            self.sample_name = ["val" + str(i) for i in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]  # (seq_len, num_kpt, kpt_dim)
        label = self.label[index]

        # preprocess
        data_numpy = data_numpy[np.newaxis, ...]
        data_numpy = normalize_by_minmax(data_numpy)
        data_numpy = select_keypoints(data_numpy, self.layout)
        data_numpy = rescale_to_neg1_pos1(data_numpy)
        data_numpy = self.gen_ske_feat[self.ske_feat]({"keypoint": data_numpy})["keypoint"]
        data_numpy = data_numpy[0]

        data_numpy = data_numpy.transpose(2, 0, 1)  # (kpt_dim, seq_len, num_kpt)
        data_numpy = data_numpy[
            ..., np.newaxis
        ]  # (kpt_dim, seq_len, num_kpt, 1 person)

        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        if self.class_map is None:
            self.class_map = np.arange(score.shape[1])
        else:
            assert len(self.class_map) == score.shape[1]
        rank = self.class_map[score.argsort()]
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
