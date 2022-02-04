import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


import seaborn as sns

from tqdm.auto import tqdm

from copy import deepcopy


def plot_cv_indices(cv, X, y, group, ax, lw=10):
    from matplotlib.colors import ListedColormap

    n_splits = cv.n_splits

    cmap_cv = plt.cm.coolwarm
    jet = plt.cm.get_cmap("jet", 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)  # inplace
    cmap_data = ListedColormap(jet(seq))
    for ii, (tr, tt) in enumerate(list(cv.split(X=X, y=y, groups=group))):
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=plt.cm.Set3
    )
    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )
    yticklabels = list(range(n_splits)) + ["target", "day"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, len(y)],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax


class BaseModelWrapper(object):

    """Docstring for BaseModel."""

    def __init__(self, build_params={}, fit_params={}):
        # self._x_train = x_train
        # self._y_train = y_train
        # self._x_val = x_val
        # self._y_val = y_val
        self.build_params = build_params
        self.fit_params = fit_params
        # self.__dict__[k] = v
        # self.model = self.build(*args, **kwargs)

    def build(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError

    def fit(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError

    def pred(self, x):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def __str__(self):
        return type(self).__name__


class DatasetWrapperBase(object):
    def __init__(self, data=None, process_func=lambda x: x):
        if data is not None:
            self.data = self.data
        else:
            self.data = self.read_data()
        self.process_func = process_func
        self.X, self.y = self.get_X_y()

    def read_data(self):
        pass

    def get_X_y(self):
        train = self.data
        train_y: pd.Series = train.pop("target")
        return self.process_func(train), train_y


class CrossValidationBase(object):
    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        data_wrapper: DatasetWrapperBase,
        cv,
        *args,
        # asset_id=0,
        **kwargs,
    ):
        self.model_wrapper = model_wrapper
        self.data_wrapper = data_wrapper
        self.models = []

        self.cv = cv

        # self.asset_id = asset_id

        self.data_params = kwargs

    def cv_split(self):
        return self.cv.split(self.data_wrapper.X, self.data_wrapper.y)

    def stuff_after_fold(slef, fold, model, x_train, y_train, x_val, y_val):
        pass

    def stuff_after_cv(self):
        pass

    def train(self):
        # self.asset_id = asset_id

        # self.oof_preds = np.zeros(len(X))
        # self.importances, self.scores, self.models = [], [], []

        for fold, (train_idx, val_idx) in tqdm(
            enumerate(self.cv_split()), desc="Cross validation training..."
        ):
            # GET TRAINING, VALIDATION SET
            X, y = self.data_wrapper.X, self.data_wrapper.y
            x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # DISPLAY FOLD INFO
            print("#" * 25)
            print(f"#### FOLD {fold+1}")
            model = self.model_wrapper
            model.build(x_train, y_train, x_val, y_val)
            model.fit(x_train, y_train, x_val, y_val)
            self.models.append(deepcopy(model))

            self.stuff_after_fold(fold, model, x_train, y_train, x_val, y_val)

        self.stuff_after_cv()

        return self.models

    def plot_cv_indices(self):

        X, y, groups = (
            self.data_wrapper.X,
            self.data_wrapper.y,
            self.data_wrapper.groups,
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_cv_indices(
            self.cv,
            X,
            y,
            groups,
            ax,
        )


def plot_importance(importances, features_names, PLOT_TOP_N=20, figsize=(12, 20)):
    try:
        plt.close()
    except Exception:
        pass
    importance_df = pd.DataFrame(data=importances, columns=features_names)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale("log")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Importance")
    plt.title("Feature Importances")
    sns.boxplot(data=sorted_importance_df[plot_cols], orient="h", ax=ax)
    plt.show()

    # return None
