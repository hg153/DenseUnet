import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from torch import nn
import torch

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds, thres = 0.3):

        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten(), thres )

    def update_tv(self, label_trues, label_preds, tv):    # calculate for threshold vector

        for lt, lp in zip(label_trues, label_preds):
            confusion_matrix_tv = self._fast_hist_tv( lt.flatten(), lp.flatten(), tv )
        
        return confusion_matrix_tv
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k not in ["Class IoU", "Precision", "Recall", "F1"]:
                string += "%s: %f\n"%(k, v)
            else:
                string += '%s: %f\n'%(k,v[1])
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    @staticmethod
    def to_excel(results, save_path):
        r_dict = {}
        for k, v in results.items():
            if k not in ["Class IoU", "Precision", "Recall", "F1"]:
                r_dict[k] = v
            else:
                r_dict[k] = v[1]

        out = pd.DataFrame.from_dict(r_dict, orient= 'index')
        out.to_csv(save_path)

    def _fast_hist(self, label_true, label_pred, thres):
        label_pred[label_pred>= thres] = 1
        label_pred[label_pred< thres] = 0
        hist = np.bincount(
            2 * label_true.astype(int) + label_pred.astype(int),
            minlength= 4,
        ).reshape(2,2)
        return hist

    def _fast_hist_tv(self, label_true, label_pred, tv):
        n = tv.shape[0]
        
        pred_tv = np.tile(label_pred, (n,1))
        tv = tv.reshape((n,1))
        pred_tv = ((pred_tv-tv)>=0).astype(int)
        hist_tv = np.zeros((n,4))

        for i in range(n):
            hist = np.bincount(
                2 * label_true.astype(int) + pred_tv[i,:],
                minlength= 4)
            hist_tv[i,:] = hist

        return hist_tv

    # obtain recall with relaxation 
    def update_tv_r(self, label_trues, label_preds, tv):    

        for lt, lp in zip(label_trues, label_preds):
            confusion_matrix_tv = self._fast_hist_tv_r( lt, lp, tv )
        
        return confusion_matrix_tv


    def _fast_hist_tv_r(self, label_true, label_pred, tv):
        
        n = tv.shape[0]
        tv = tv.reshape((n,1,1))
        
        x = torch.tensor(label_pred)
        x = x.unsqueeze(0)
        x = x.repeat(n,1,1)
        x = ((x - tv)>0).type(torch.float32)
        
        buff = nn.MaxPool2d(5, stride = 1, padding = 2)
        
        x = buff(x).numpy().astype(int)
        
        label_true = label_true.flatten()
        
        hist_tv = np.zeros((n,4))

        for i in range(n):
            hist = np.bincount(
                2 * label_true.astype(int) + x[i,...].flatten(),
                minlength= 4)
            hist_tv[i,:] = hist

        return hist_tv

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        # calculate precision, recall, and f1
        eps = 1e-6
        precision = np.diag(hist) / (hist.sum(axis=0) + eps)
        recall = np.diag(hist) / (hist.sum(axis=1) + eps)
        f1 = 2*precision*recall / (precision + recall + eps)

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
