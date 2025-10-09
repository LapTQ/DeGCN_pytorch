from __future__ import print_function


import os
import random
import sys
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings

import torch
from tqdm import tqdm

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #     torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Processor:
    """
    Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.load_model()

        self.model = self.model.cuda(self.output_device)

        self.load_data()

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed,
        )

    def load_model(self):
        output_device = (
            self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        )
        self.output_device = output_device
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args)

        if self.arg.weights:
            weights = torch.load(self.arg.weights)
            weights = OrderedDict(
                [
                    [k.split("module.")[-1], v.cuda(output_device)]
                    for k, v in weights.items()
                ]
            )
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print("Can not find these weights:")
                for d in diff:
                    print("  " + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def eval(
        self,
        epoch,
        save_score=False,
        loader_name=["test"],
        wrong_file=None,
        result_file=None,
    ):
        self.model.eval()
        for ln in loader_name:
            score_frag = []
            label_list = []
            pred_list = []
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    output = self.model(data)
                    output = sum(output)

                    score_frag.append(output.data.cpu().numpy())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())

            score = np.concatenate(score_frag)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            print(accuracy)
            print(each_acc)
            print(confusion)

    def start(self):
        self.eval(
            epoch=0,
            save_score=False,
            loader_name=["test"],
            wrong_file=None,
            result_file=None,
        )


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition(".")
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(
            "Class %s cannot be found (%s)"
            % (class_str, traceback.format_exception(*sys.exc_info()))
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.chdir(os.getcwd())

    class Arg:
        feeder = "feeders.feeder_from_protogcn_format.Feeder"
        test_feeder_args = dict(
            data_path="/home/laptq/laptq-fs26-shoplifting-detection/outputs/convert_pkl_STGCN_to_ProtoGCN/fs26/shoplift25min_satudoraR9--r1-0x1-1x3--1s-15frames/val.pkl",
            layout="coco_onlyhand",
            split="val",
        )
        test_batch_size = 256
        num_worker = 0
        device = 0
        model = "model.degcn.Model"
        model_args = dict(
            num_class=2,
            num_point=6,
            num_person=1,
            base_frame=15,
            graph="graph.coco_onlyhand.Graph",
            in_channels=2,
            k=8,
            eta=4,
            num_stream=2,
            graph_args=dict(labeling_mode="spatial"),
        )
        weights = "/home/laptq/DeGCN_pytorch/outputs/train/fs26/v102--mnit_poselift_roboflow_satudora_awlrecord--r1.9-0xauto-1x1--only-normal-satudora--1s-15frames/model_8/ 20251009 214021/epoch_2_1428.pt"

    arg = Arg()

    processor = Processor(arg)
    processor.start()
