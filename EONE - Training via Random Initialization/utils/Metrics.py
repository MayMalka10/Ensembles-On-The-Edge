
import numpy as np
import torch
import torchmetrics

from torchmetrics import Metric

class PrecisionAtRecall(Metric):
    def __init__(self, dist_sync_on_step=False, recall_point=0.95):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.recall_point = recall_point
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("wrong", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, distances: torch.Tensor, labels: torch.Tensor):
        labels = labels[torch.argsort( distances )]
        # Sliding threshold: get first index where recall >= recall_point.
        # This is the index where the number of elements with label==1 below the threshold reaches a fraction of
        # 'recall_point' of the total number of elements with label==1.
        # (np.argmax returns the first occurrence of a '1' in a bool array).
        threshold_index = torch.where( torch.cumsum( labels, dim=0 ) >= self.recall_point * torch.sum( labels ) )[0][0]
        self.correct += torch.sum(labels[threshold_index:] == 0)
        self.wrong += torch.sum(labels[:threshold_index] == 0)

    def compute(self):
        return self.correct.float() / (self.correct + self.wrong)


class RecallAtPrecision(Metric):
    def __init__(self, dist_sync_on_step=False, precision_point=0.95):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.precision_point = precision_point
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("wrong", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, distances: torch.Tensor, labels: torch.Tensor):

        labels = labels[torch.argsort( distances )]
        distances = distances.sort()[0]

        best_correct = 0
        positives = labels.sum()
        for idx, tsh in enumerate(distances):
            retrieved = idx
            false = (1-labels)[0:idx].sum()
            fpr = false / retrieved
            if fpr < 1-self.precision_point:
                best_correct = max(best_correct, labels[0:idx].sum())
        self.correct += best_correct.type(torch.int64)
        self.wrong += (positives-best_correct).type(torch.int64)


    def compute(self):
        return self.correct.float() / (self.correct + self.wrong)



class PrecisionAtRecallRealistic(Metric):
    def __init__(self, dist_sync_on_step=False, recall_point=0.95):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.recall_point = recall_point
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("wrong", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, distances: torch.Tensor, labels: torch.Tensor):
        labels = labels[torch.argsort( distances )]
        # Sliding threshold: get first index where recall >= recall_point.
        # This is the index where the number of elements with label==1 below the threshold reaches a fraction of
        # 'recall_point' of the total number of elements with label==1.
        # (np.argmax returns the first occurrence of a '1' in a bool array).
        threshold_index = torch.where( torch.cumsum( labels, dim=0 ) >= self.recall_point * torch.sum( labels ) )[0][0]
        self.correct += torch.sum(labels[threshold_index:] == 0)
        self.wrong += torch.sum(labels[:threshold_index] == 0)

    def compute(self):
        return self.correct.float() / (self.correct + self.wrong)