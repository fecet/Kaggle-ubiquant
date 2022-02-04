# %%
import os
import numpy as np
import pandas as pd

from pathlib import Path
import joblib

from sklearn.model_selection import GroupKFold

# from sklearn.linear_model import Ridge
# from scipy.stats import pearsonr as p
import lightgbm as lgb
from utils import find_gpus

from loss import LOSS

os.environ["CUDA_VISIBLE_DEVICES"] = find_gpus(
    num_of_cards_needed=4
)  # must before `import torch`

# if isnotebook():
#     from tqdm.notebook import tqdm
# else:
#     from tqdm import tqdm
from tqdm.auto import tqdm
import tensorflow as tf
import keras_tuner as kt

import warnings

warnings.filterwarnings("ignore")

# %%

DATAPATH = Path("data/ubiquant-parquet")
TRAIN_DATAPATH = DATAPATH / "train.parquet"


# %%


# %%

from base import (
    DatasetWrapperBase,
    CrossValidationBase,
    BaseModelWrapper,
)


class UbiquantDatasetWrapper(DatasetWrapperBase):
    def __init__(self, process_func):
        super().__init__(process_func)
        self.groups = self.data["time_id"]

    def read_data(self):
        print("Loading training data")
        train: pd.DataFrame = pd.read_parquet(TRAIN_DATAPATH)
        return train

    def get_X_y(self):
        train = self.data
        train_y: pd.Series = train.pop("target")
        return self.process_func(train), train_y

    def refresh_X(self, process_func):
        self.X = process_func(self.data)


class UbiquantCrossValidation(CrossValidationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fold_scores = []

    def cv_split(self):
        cv = self.cv
        data = self.data_wrapper
        return cv.split(data.X, data.y, groups=data.groups)

    def stuff_after_fold(self, fold, model, x_train, y_train, x_val, y_val):

        # train_score = LOSS["corr"](model.pred(x_train), y_train)
        val_score = LOSS["corr"](model.pred(x_val), y_val)
        print(
            # f"Fold {fold + 1}: training score: {train_score}, validation score: {val_score}"
            f"Fold {fold + 1}: validation score: {val_score}"
        )
        self.fold_scores.append(val_score)

    def stuff_after_cv(self):
        #         preds = []
        #         for model in tqdm(self.models, desc="Evaluating model..."):
        #             preds.append(model.pred(self.data_wrapper.X))
        #
        #         ensemble_func = np.median if len(self.models) > 10 else np.mean
        #
        #         ensemble_pred = ensemble_func(np.stack(preds), axis=0)
        # score = LOSS["corr"](ensemble_pred, self.data_wrapper.y)

        print(f"Overall score: {np.mean(self.fold_scores)}")
        # print(f"Ensemble cross-validation score: {score}")


class LGBModelWrapper(BaseModelWrapper):

    """Docstring for LGBMModel."""

    def __init__(self, *args, **kwargs):
        """TODO: to be defined."""
        super().__init__(*args, **kwargs)
        self.count = 0

    def fit(self, x_train, y_train, x_val, y_val):
        self.model = self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            **self.fit_params,
        )

    def pred(self, x):
        return self.model.predict(x)

    def build(self, x_train, y_train, x_val, y_val):
        self.count += 1
        self.model: lgb.LGBMRegressor = lgb.LGBMRegressor(**self.build_params)

    def get_importance(self):
        return self.model.feature_importances_

    def save(self):
        # pd.to_datetime("now")
        path = Path(f"{self}")
        path.mkdir(exist_ok=True, parents=True)
        joblib.dump(self.model, path / f"{self.count}.pkl")

    def __copy__(self):
        model = object.__new__(type(self))
        model.__dict__ = self.__dict__
        return model

    @classmethod
    def load(cls, path):
        model = cls()
        model.model = joblib.load(path)
        return model


# %%


# from lets_plot import *
#
# LetsPlot.setup_html()
#
# (
#     ggplot(train[:10000], aes(x="target"))
#     + geom_density(color="dark_green", alpha=0.7)
# )

# %%


def agg_across_col(df, across):
    df_agg = df.groupby([across]).agg([np.mean])
    df_agg.columns = ["_".join(col) for col in df_agg.columns]
    feats = [f for f in df_agg.columns if f.startswith("f")]
    return df_agg[feats]


def process_data(train: pd.DataFrame) -> pd.DataFrame:
    feat_across_time = agg_across_col(train, "time_id")
    # feat_across_ivm = agg_across_col(train, "investment_id")
    train = train.merge(feat_across_time, on="time_id")

    feats = [f for f in train.columns if f not in ["time_id", "row_id", "target"]]
    train_x = train[feats]
    train_x.drop("investment_id", axis=1)
    return train_x


# %%

train: pd.DataFrame = pd.read_parquet(TRAIN_DATAPATH)

data = UbiquantDatasetWrapper(data=train, process_func=process_data)
model = LGBModelWrapper(
    build_params=dict(
        objective="regression",
        metric="mse",
        n_estimators=1000,
        num_leaves=63,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        feature_fraction=0.6,
        early_stopping_rounds=100,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=62,
        verbose=-1,
        device="gpu",
        n_jobs=8,
    ),
    fit_params=dict(
        eval_metric="rmse",
    ),
)

cv = GroupKFold(n_splits=5)

# cv = PurgedGroupTimeSeriesSplit(
#     n_splits=5,
#     group_gap=10,
#     max_train_group_size=400,
#     max_test_group_size=100,
# )

cvv = UbiquantCrossValidation(model_wrapper=model, data_wrapper=data, cv=cv)


# %%


# cvv.plot_cv_indices()
models = cvv.train()


# %%

for model in tqdm(models, desc="Saving models"):
    # print(model, model.count)
    # model.save()
    score = np.mean(model.pred(data.X))
    print(score)

# %%

model_type = LGBModelWrapper
models = [model_type.load(f) for f in Path(model_type.__name__).glob("*.pkl")]

# %%

models

# %%

# %%

#
# pearson_corr = LOSS["corr"]
# n_splits = 100
# fold_scores = []
# models = []
#
# kfold = GroupKFold(n_splits=n_splits)
# cvs = kfold.split(train_x, train_y, groups=train["time_id"])
# for fold, (trn_idx, val_idx) in tqdm(
#     enumerate(cvs),
#     desc="Training data...",
# ):
#     # fold += 5
#     # X_train, y_train = train.iloc[trn_idx][feats], train.iloc[trn_idx][target]
#     # X_val, y_val = train.iloc[val_idx][feats], train.iloc[val_idx][target]
#     X_train, y_train = train_x.iloc[trn_idx], train_y.iloc[trn_idx]
#     X_val, y_val = train_x.iloc[val_idx], train_y.iloc[val_idx]
#
#     model = _create_lgb_model(X_train, y_train, X_val, y_val)
#
#     y_pred = model.predict(X_val)
#
#     score = pearson_corr(y_pred, y_val)
#     print(f"Fold {fold + 1}: {score}")
#     fold_scores.append(score)
#     models.append(model)
#     joblib.dump(model, f"large_lgb/lgb_{fold}.pkl")
#     # break
#
# print(f"Overall score: {np.mean(fold_scores)}")
#
#
# %%


# %%
#
# from tabnet.models.classify import TabNetClassifier
# from tabnet.schedules import DecayWithWarmupSchedule
#
#
# SEARCH_DIR = ".search"
# SEED = 42
# DEFAULTS = {"num_features": 601, "n_classes": 1, "min_learning_rate": 1e-6}  # 28x28
#
#
# # because doing a training on MNIST is something I MUST do, no?
# # this time let's add a twist & do hyperparameter optimization with kerastuner
#
#
# def build_model(hp):
#     model = TabNetClassifier(
#         num_features=DEFAULTS["num_features"],
#         feature_dim=hp.Choice("feature_dim", values=[16, 32, 64], default=32),
#         output_dim=hp.Choice("output_dim", values=[16, 32, 64], default=32),
#         n_classes=DEFAULTS["n_classes"],
#         n_step=hp.Choice("n_step", values=[2, 4, 5, 6], default=4),
#         relaxation_factor=hp.Choice(
#             "relaxation_factor", values=[1.0, 1.25, 1.5, 2.0, 3.0], default=1.5
#         ),
#         sparsity_coefficient=hp.Choice(
#             "sparsity_coefficient",
#             values=[0.0001, 0.001, 0.01, 0.02, 0.05],
#             default=0.0001,
#         ),
#         bn_momentum=hp.Choice("bn_momentum", values=[0.6, 0.7, 0.9], default=0.7),
#         bn_virtual_divider=1,  # let's not use Ghost Batch Normalization. batch sizes are too small
#         dp=hp.Choice("dp", values=[0.0, 0.1, 0.2, 0.3, 0.4], default=0.0),
#     )
#     lr = DecayWithWarmupSchedule(
#         hp.Choice(
#             "learning_rate", values=[0.001, 0.005, 0.01, 0.02, 0.05], default=0.02
#         ),
#         DEFAULTS["min_learning_rate"],
#         hp.Choice("warmup", values=[1, 5, 10, 20], default=5),
#         hp.Choice("decay_rate", values=[0.8, 0.90, 0.95, 0.99], default=0.95),
#         hp.Choice("decay_steps", values=[10, 100, 500, 1000], default=500),
#     )
#
#     optimizer = tf.keras.optimizers.Adam(
#         learning_rate=lr,
#         clipnorm=hp.Choice("clipnorm", values=[1, 2, 5, 10], default=2),
#     )
#
#     # lossf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#     lossf = tf.keras.metrics.RootMeanSquaredError()
#
#     model.compile(
#         optimizer,
#         loss=lossf,
#         # metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
#     )
#
#     return model
#
#
# # %%
#
# hp = kt.HyperParameters()
# model = build_model(hp)
#
# # %%
#
# import tensorflow_io as tfio
#
# ds = tfio.experimental.IODataset.from_numpy((train_x.to_numpy(), train_y.to_numpy()))
#
# # %%
#
# es_callback = tf.keras.callbacks.EarlyStopping(
#     monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
# )
# model.fit(x=ds.shuffle(int(4e6)).batch(1024 * 4), epochs=100, callbacks=[es_callback], verbose=2,)
#
#
# # %%
# project_name = "tabnet"
# tuner = kt.RandomSearch(
#     build_model,
#     # hyperparameters=hp,
#     objective=kt.Objective("val_loss", direction="min"),
#     max_trials=40,
#     overwrite=False,
#     directory="kt",
#     project_name=project_name,
#     seed=64,
# )
# tuner.search
