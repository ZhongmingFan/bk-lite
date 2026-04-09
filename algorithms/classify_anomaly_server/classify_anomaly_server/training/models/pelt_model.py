"""PELT changepoint 异常检测模型。"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from loguru import logger

from .base import BaseAnomalyModel, ModelRegistry
from .pelt_utils import breakpoints_to_changepoints, event_window_scores


@ModelRegistry.register("PELT")
class PELTModel(BaseAnomalyModel):
    """基于 ruptures PELT 的点级异常检测模型。"""

    def __init__(
        self,
        cost_model: str = "l2",
        pen: float = 10.0,
        min_size: int = 3,
        jump: int = 1,
        event_window: int = 1,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            cost_model=cost_model,
            pen=pen,
            min_size=min_size,
            jump=jump,
            event_window=event_window,
            threshold=threshold,
            **kwargs,
        )
        self.cost_model = str(cost_model)
        self.pen = float(pen)
        self.min_size = int(min_size)
        self.jump = int(jump)
        self.event_window = int(event_window)
        self.threshold = float(threshold)
        self.feature_names_: list[str] | None = None
        self.n_samples_train_: int | None = None

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame | pd.Series] = None,
        **kwargs: Any,
    ) -> "PELTModel":
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("train_data 必须是 pandas.DataFrame")
        if list(train_data.columns) != ["value"]:
            raise ValueError("PELTModel 训练数据必须只包含 value 列")

        self.feature_names_ = train_data.columns.tolist()
        self.n_samples_train_ = len(train_data)
        self.threshold_ = self.threshold
        self.is_fitted = True

        logger.info(
            f"PELT 模型训练完成: samples={self.n_samples_train_}, threshold={self.threshold_}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.int_]:
        scores = self.predict_proba(X)
        return (scores > self.threshold_).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        self._check_fitted()
        self._validate_features(X)

        signal = X["value"].to_numpy(dtype=float)
        if len(signal) == 0 or len(signal) < max(2, self.min_size * 2):
            return np.zeros(len(signal), dtype=float)

        rpt = import_module("ruptures")
        algo = rpt.Pelt(model=self.cost_model, min_size=self.min_size, jump=self.jump)
        breakpoints = algo.fit_predict(signal, pen=self.pen)
        changepoints = breakpoints_to_changepoints(breakpoints, len(signal))
        return event_window_scores(len(signal), changepoints, self.event_window)

    def optimize_hyperparams(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        train_labels: NDArray[np.int_] | pd.Series,
        val_labels: NDArray[np.int_] | pd.Series,
        config: Any,
    ) -> dict[str, Any]:
        search_config = config.get_search_config()
        search_space = search_config["search_space"]
        metric = search_config["metric"]

        best_score = float("-inf")
        best_params = {
            "pen": self.pen,
            "min_size": self.min_size,
            "jump": self.jump,
        }

        jump_values = search_space.get("jump", [self.jump])
        eval_count = 0
        total_evals = (
            len(search_space["pen"]) * len(search_space["min_size"]) * len(jump_values)
        )
        for pen in search_space["pen"]:
            for min_size in search_space["min_size"]:
                for jump in jump_values:
                    eval_count += 1
                    candidate = PELTModel(
                        pen=pen,
                        min_size=min_size,
                        jump=jump,
                        cost_model=config.cost_model or self.cost_model,
                        event_window=config.event_window or self.event_window,
                        threshold=self.threshold,
                    )
                    candidate.fit(train_data)
                    if isinstance(val_labels, pd.Series):
                        labels_array: NDArray[np.int_] = val_labels.to_numpy(dtype=int)
                    else:
                        labels_array = val_labels.astype(int, copy=False)
                    metrics = candidate.evaluate(val_data, labels_array)
                    score = float(metrics.get(metric, metrics["f1"]))

                    logger.debug(
                        f"PELT trial [{eval_count}/{total_evals}] "
                        f"pen={pen}, min_size={min_size}, jump={jump}: "
                        f"{metric}={score:.4f}"
                    )

                    # 每轮试验记录到 MLflow（镜像 ECOD 的 hyperopt/* 命名）
                    if mlflow.active_run():
                        mlflow.log_metric(
                            f"hyperopt/val_{metric}", score, step=eval_count
                        )
                        mlflow.log_metric(
                            "hyperopt/val_f1",
                            float(metrics.get("f1", score)),
                            step=eval_count,
                        )
                        mlflow.log_metric(
                            "hyperopt/val_precision",
                            float(metrics.get("precision", 0.0)),
                            step=eval_count,
                        )
                        mlflow.log_metric(
                            "hyperopt/val_recall",
                            float(metrics.get("recall", 0.0)),
                            step=eval_count,
                        )

                    if score > best_score:
                        best_score = score
                        best_params = {
                            "pen": float(pen),
                            "min_size": int(min_size),
                            "jump": int(jump),
                        }
                        if mlflow.active_run():
                            mlflow.log_metric(
                                "hyperopt/best_so_far", score, step=eval_count
                            )

        # 汇总指标（镜像 ECOD 的 hyperopt_summary/* 命名）
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "hyperopt_summary/total_evals": float(total_evals),
                    "hyperopt_summary/actual_evals": float(eval_count),
                    "hyperopt_summary/best_score": best_score,
                }
            )
            logger.info(
                f"PELT 超参数搜索完成: {eval_count} 轮, 最优 {metric}={best_score:.4f}, "
                f"参数={best_params}"
            )

        self.pen = float(best_params["pen"])
        self.min_size = int(best_params["min_size"])
        self.jump = int(best_params["jump"])
        self.config.update(best_params)
        return best_params

    def save_mlflow(self, artifact_path: str = "model") -> None:
        self._check_fitted()

        if mlflow.active_run():
            mlflow.log_dict(
                {
                    "model_type": "PELT",
                    "cost_model": self.cost_model,
                    "pen": self.pen,
                    "min_size": self.min_size,
                    "jump": self.jump,
                    "event_window": self.event_window,
                    "threshold": float(self.threshold_),
                    "feature_names": self.feature_names_,
                    "n_samples_train": self.n_samples_train_,
                },
                "model_metadata.json",
            )

        pelt_wrapper_module = import_module(
            "classify_anomaly_server.training.models.pelt_wrapper"
        )
        PELTWrapper = pelt_wrapper_module.PELTWrapper

        wrapped_model = PELTWrapper(
            cost_model=self.cost_model,
            pen=self.pen,
            min_size=self.min_size,
            jump=self.jump,
            event_window=self.event_window,
            threshold=float(self.threshold_),
        )

        import cloudpickle

        cloudpickle.dumps(wrapped_model)
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=wrapped_model,
        )

    def _validate_features(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X 必须是 pandas.DataFrame")
        if self.feature_names_ is None or list(X.columns) != self.feature_names_:
            raise ValueError(
                f"特征列不匹配。期望: {self.feature_names_}, 实际: {list(X.columns)}"
            )
