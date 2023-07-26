import torch
import numpy as np
import warnings

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve,\
    average_precision_score, f1_score, precision_score, recall_score,\
        mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,\
            r2_score, d2_pinball_score, top_k_accuracy_score

from .basemetric import BaseMetric

warnings.filterwarnings('ignore')



class Acc1(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return accuracy_score(answers, labels)


class Acc5(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).softmax(-1).numpy()
        answers = torch.cat(self.answers).numpy()
        num_classes = scores.shape[-1]
        return top_k_accuracy_score(answers, scores, k=5, labels=np.arange(num_classes))

class Auroc(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).softmax(-1).numpy()
        answers = torch.cat(self.answers).numpy()
        num_classes = scores.shape[-1]
        return roc_auc_score(answers, scores, average='weighted', multi_class='ovr', labels=np.arange(num_classes))

class Auprc(BaseMetric): # only for binary classification
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).sigmoid().numpy()
        answers = torch.cat(self.answers).numpy()
        return average_precision_score(answers, scores, average='weighted')

class Youdenj(BaseMetric):  # only for binary classification
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).sigmoid().numpy()
        answers = torch.cat(self.answers).numpy()
        fpr, tpr, thresholds = roc_curve(answers, scores)
        return thresholds[np.argmax(tpr - fpr)]

class F1(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return f1_score(answers, labels, average='weighted', zero_division=0)

class Precision(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return precision_score(answers, labels, average='weighted', zero_division=0)

class Recall(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return recall_score(answers, labels, average='weighted', zero_division=0)

class Seqacc(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        num_classes = pred.size(-1)
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p.view(-1, num_classes))
        self.answers.append(t.view(-1))

    def summarize(self):
        labels = torch.cat(self.scores).argmax(-1).numpy()
        answers = torch.cat(self.answers).numpy()

        # ignore special tokens
        labels = labels[answers != -1]
        answers = answers[answers != -1]
        return accuracy_score(answers, labels)

class Mse(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return mean_squared_error(answers, scores)

class Rmse(Mse):
    def __init__(self):
        super(Rmse, self).__init__()

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return mean_squared_error(answers, scores, squared=False)

class Mae(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return mean_absolute_error(answers, scores)

class Mape(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return mean_absolute_percentage_error(answers, scores)

class R2(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self, *args):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return r2_score(answers, scores)

class D2(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self, *args):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return d2_pinball_score(answers, scores)
