#!/usr/bin/env python3

from ...abstract import TrainingFiltering

from typing import TYPE_CHECKING
import argparse
import os

import numpy as np
import torch
import time
import math
from trojanzoo.utils.data import TensorListDataset, sample_batch
from sklearn.decomposition import PCA
import pickle

if TYPE_CHECKING:
    import torch.utils.data


EPS = 1e-5


class SCAn(TrainingFiltering):
    name: str = 'scan'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--sc_threshold', type=float,
                           help='treat the output of this layer as the representation of the input'
                                '(default: "epx(2)")')
        group.add_argument('--repr_layer',
                           help='treat the output of this layer as the representation of the input'
                                '(default: "flatten")')
        group.add_argument('--repr_dim', type=int,
                           help='maximum representation dimensions'
                                '(default: 512)')
        group.add_argument('--defense_input_ratio', type=float,
                           help='ratio of clean training data used to launch SCAn'
                                '(default: 0.1)')
        return group

    def __init__(self,
                 sc_threshold: float = math.exp(2),
                 repr_layer: str = 'flatten',
                 repr_dim: int = 512,
                 defense_input_ratio: float = 0.1,
                 **kwargs):
        defense_input_ratio = max(min(defense_input_ratio, 1.0), 0.0)
        defense_input_num = round(defense_input_ratio * len(kwargs['dataset'].loader['train'].dataset))
        self.defense_input_ratio = defense_input_ratio
        super().__init__(defense_input_num=defense_input_num, **kwargs)
        self.param_list['scan'] = ['sc_threshold', 'repr_layer', 'repr_dim', 'defense_input_ratio']

        self.sc_threshold = sc_threshold
        self.repr_layer = repr_layer
        self.repr_dim = repr_dim
        self.scan = SCAn_Agent()
        self.global_model = None

    @torch.no_grad()
    def get_repr(self, dataset):
        loader = self.dataset.get_dataloader('train', dataset=dataset)
        repr_list = list()
        label_list = list()
        model = self.attack.model
        model.eval()
        for data in loader:
            _input, _label = model.get_data(data)
            repr = model.get_layer(_input, layer_output=self.repr_layer)
            if len(repr.shape) > 2:
                repr = torch.flatten(repr, start_dim=1)
            repr_list.append(repr.detach().cpu().numpy())
            label_list.append(_label.detach().cpu().numpy())
        repr_list = np.concatenate(repr_list)
        label_list = np.concatenate(label_list)
        return repr_list, label_list

    def detect(self, **kwargs):
        def_rst = dict()
        num_classes = self.dataset.num_classes
        clean_repr, clean_label = self.get_repr(self.clean_set)
        print('clean representations', clean_repr.shape)

        reduce_dim = None
        if self.repr_dim < clean_repr.shape[1]:
            reduce_dim = self.repr_dim

        st_time = time.time()
        self.global_model = self.scan.build_global_model(clean_repr, clean_label, num_classes, reduce_dim=reduce_dim)
        ed_time = time.time()
        print('global model has been built in {:.3f} seconds'.format(ed_time - st_time))

        def_rst['global_model'] = self.global_model

        '''
        st_time = time.time()
        cls_rst = dict()
        for cls in range(num_classes):
            _repr = clean_repr[clean_label == cls]
            rst = self.scan.build_local_model(_repr, self.global_model)
            cls_rst[cls] = rst
        ed_time = time.time()
        print('clean local model has been built in {:.3f} seconds'.format(ed_time - st_time))
        '''

        # -------------------local--------------------------

        train_set = self.dataset.loader['train'].dataset
        train_repr, train_label = self.get_repr(train_set)
        poison_repr, poison_label = self.get_repr(self.poison_set)
        print('poison representations', poison_repr.shape)

        def_rst['class_rst'] = list()

        a = [0] * num_classes
        for tgt in range(num_classes):
            _poison_repr = poison_repr[poison_label == tgt]
            if len(_poison_repr) == 0:
                _repr = train_repr[train_label == tgt]
            else:
                _train_repr = train_repr[train_label == tgt]
                npo, ntr = len(_poison_repr), len(_train_repr)
                _npo = round(ntr * (npo / (ntr + npo)))
                _ntr = ntr - _npo
                idx_p = np.random.choice(np.arange(npo), _npo)
                idx_t = np.random.choice(np.arange(ntr), _ntr)
                _repr = np.concatenate([_train_repr[idx_t], _poison_repr[idx_p]])

            # _repr = _poison_repr

            st_time = time.time()
            rst = self.scan.build_local_model(_repr, self.global_model)
            ed_time = time.time()
            print(tgt, 'local model has been built in {:.3f} seconds'.format(ed_time - st_time))

            def_rst['class_rst'].append(rst)
            a[tgt] = rst['sc']

        print(a)
        class_scores = self.scan.calc_anomaly_index(a / np.max(a))
        print(class_scores)

        def_rst['class_scores'] = class_scores

        suspicious_classes = list()
        for c, sc in enumerate(class_scores):
            if sc > self.sc_threshold:
                suspicious_classes.append(c)

        if len(suspicious_classes) == 0:
            suspicious_classes = None
        print('suspicious classes: ', suspicious_classes)

        # --------------------------- save defense results ----------------------------
        file_path = os.path.normpath(os.path.join(
            self.folder_path, self.get_filename() + '.pkl'))
        with open(file_path, 'wb') as fh:
            pickle.dump(def_rst, fh)

        return suspicious_classes

    def get_pred_labels(self) -> torch.Tensor:
        raise NotImplementedError

    def get_datasets(self) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        if self.attack.poison_set is None:
            self.attack.poison_set = self.attack.get_poison_dataset()
        if not self.defense_input_num:
            return self.dataset.loader['train'].dataset, self.attack.poison_set

        poison_set = self.attack.poison_set
        clean_input, clean_label = sample_batch(self.dataset.loader['train'].dataset, batch_size=self.defense_input_num)
        clean_set = TensorListDataset(clean_input, clean_label.tolist())
        return clean_set, poison_set


class SCAn_Agent:
    def __init__(self):
        self.gb_model = None
        self.lc_model = None

    def calc_final_score(self, lc_model=None):
        if lc_model is None:
            lc_model = self.lc_model
        sts = lc_model['sts']
        y = sts[:, 1]
        ai = self.calc_anomaly_index(y / np.max(y))
        return ai

    def build_global_model(self, reprs, labels, n_classes, reduce_dim=None):
        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        if reduce_dim:
            projector = PCA(n_components=reduce_dim, svd_solver='full')
            X = projector.fit_transform(X)
            print('clean representations deduce to', X.shape)

        N = X.shape[0]  # num_samples
        M = X.shape[1]  # len_features
        L = n_classes

        cnt_L = np.zeros(L)
        mean_f = np.zeros([L, M])
        for k in range(L):
            idx = (labels == k)
            cnt_L[k] = np.sum(idx)
            mean_f[k] = np.mean(X[idx], axis=0)

        u = np.zeros([N, M])
        e = np.zeros([N, M])
        for i in range(N):
            k = labels[i]
            u[i] = mean_f[k]  # class-mean
            e[i] = X[i] - u[i]  # sample-variantion
        Su = np.cov(np.transpose(u))
        Se = np.cov(np.transpose(e))

        # EM
        dist_Su = 1e5
        dist_Se = 1e5
        n_iters = 0
        while (dist_Su + dist_Se > EPS) and (n_iters < 100):
            n_iters += 1
            last_Su = Su
            last_Se = Se

            F = np.linalg.pinv(Se)
            SuF = np.matmul(Su, F)

            G_set = list()
            for k in range(L):
                G = -np.linalg.pinv(cnt_L[k] * Su + Se)
                G = np.matmul(G, SuF)
                G_set.append(G)

            u_m = np.zeros([L, M])
            e = np.zeros([N, M])
            u = np.zeros([N, M])

            for i in range(N):
                vec = X[i]
                k = labels[i]
                G = G_set[k]
                dd = np.matmul(np.matmul(Se, G), np.transpose(vec))
                u_m[k] = u_m[k] - np.transpose(dd)

            for i in range(N):
                vec = X[i]
                k = labels[i]
                e[i] = vec - u_m[k]
                u[i] = u_m[k]

            # max-step
            Su = np.cov(np.transpose(u))
            Se = np.cov(np.transpose(e))

            dif_Su = Su - last_Su
            dif_Se = Se - last_Se

            dist_Su = np.linalg.norm(dif_Su)
            dist_Se = np.linalg.norm(dif_Se)
            # print(dist_Su,dist_Se)

        gb_model = dict()
        gb_model['Su'] = Su
        gb_model['Se'] = Se
        gb_model['mean'] = mean_f
        gb_model['mean_a'] = mean_a
        if reduce_dim:
            gb_model['projector'] = projector
        self.gb_model = gb_model
        return gb_model

    def build_local_model(self, reprs, gb_model):
        Su = gb_model['Su']
        Se = gb_model['Se']

        if 'F' not in gb_model:
            gb_model['F'] = np.linalg.pinv(Se)
        F = gb_model['F']

        # mean_a = np.mean(reprs, axis=0)
        mean_a = gb_model['mean_a']
        X = reprs - mean_a
        if 'projector' in gb_model:
            X = gb_model['projector'].transform(X)

        subg, i_u1, i_u2 = self.find_split(X, F)
        i_sc = self.calc_test(X, Su, Se, F, subg, i_u1, i_u2)
        ret = {
            'sc': i_sc.item(),
            'mu_1': i_u1,
            'mu_2': i_u2,
            'subg': subg,
        }
        return ret

    def calc_anomaly_index(self, a):
        ma = np.median(a)
        b = abs(a - ma)
        mm = np.median(b) * 1.4826
        index = b / mm
        return index

    def find_split(self, X, F):
        N = X.shape[0]
        M = X.shape[1]
        subg = np.random.rand(N)

        if (N == 1):
            subg[0] = 0
            return (subg, X.copy(), X.copy())

        if np.sum(subg >= 0.5) == 0:
            subg[0] = 1
        if np.sum(subg < 0.5) == 0:
            subg[0] = 0
        last_z1 = -np.ones(N)

        # EM
        steps = 0
        while (np.linalg.norm(subg - last_z1) > EPS) and (np.linalg.norm((1 - subg) - last_z1) > EPS) and (steps < 100):
            steps += 1
            last_z1 = subg.copy()

            # max-step
            # calc u1 and u2
            idx1 = (subg >= 0.5)
            idx2 = (subg < 0.5)
            if (np.sum(idx1) == 0) or (np.sum(idx2) == 0):
                break
            if np.sum(idx1) == 1:
                u1 = X[idx1]
            else:
                u1 = np.mean(X[idx1], axis=0)
            if np.sum(idx2) == 1:
                u2 = X[idx2]
            else:
                u2 = np.mean(X[idx2], axis=0)

            bias = np.matmul(np.matmul(u1, F), np.transpose(u1)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
            e2 = u1 - u2  # (64,1)
            for i in range(N):
                e1 = X[i]
                delta = np.matmul(np.matmul(e1, F), np.transpose(e2))
                if bias - 2 * delta < 0:
                    subg[i] = 1
                else:
                    subg[i] = 0

        return (subg, u1, u2)

    def calc_test(self, X, Su, Se, F, subg, u1, u2):
        N = X.shape[0]
        M = X.shape[1]

        G = -np.linalg.pinv(N * Su + Se)
        mu = np.zeros([1, M])
        for i in range(N):
            vec = X[i]
            dd = np.matmul(np.matmul(Se, G), np.transpose(vec))
            mu = mu - dd

        b1 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u1, F), np.transpose(u1))
        b2 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
        n1 = np.sum(subg >= 0.5)
        n2 = N - n1
        sc = n1 * b1 + n2 * b2

        for i in range(N):
            e1 = X[i]
            if subg[i] >= 0.5:
                e2 = mu - u1
            else:
                e2 = mu - u2
            sc -= 2 * np.matmul(np.matmul(e1, F), np.transpose(e2))

        return sc / N
