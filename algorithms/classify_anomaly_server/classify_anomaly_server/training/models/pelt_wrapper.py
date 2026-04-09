"""PELT 模型的 MLflow 推理包装器。"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
from .pelt_utils import breakpoints_to_changepoints, event_window_scores


class PELTWrapper(mlflow.pyfunc.PythonModel):
    """将 PELT changepoint 结果映射为点级输出。"""

    def __init__(
        self,
        cost_model: str,
        pen: float,
        min_size: int,
        jump: int,
        event_window: int,
        threshold: float,
    ) -> None:
        self.cost_model = cost_model
        self.pen = pen
        self.min_size = min_size
        self.jump = jump
        self.event_window = event_window
        self.threshold = threshold

    def predict(self, context, model_input, params=None):
        data, threshold = self._parse_input(model_input)
        series = self._to_series(data)
        signal = series.to_numpy(dtype=float)

        if len(signal) == 0:
            scores = np.zeros(0, dtype=float)
        elif len(signal) < max(2, self.min_size * 2):
            scores = np.zeros(len(signal), dtype=float)
        else:
            rpt = import_module("ruptures")
            algo = rpt.Pelt(
                model=self.cost_model, min_size=self.min_size, jump=self.jump
            )
            breakpoints = algo.fit_predict(signal, pen=self.pen)
            changepoints = breakpoints_to_changepoints(breakpoints, len(signal))
            scores = event_window_scores(
                length=len(signal),
                changepoints=changepoints,
                event_window=self.event_window,
            )

        anomaly_severity = np.minimum(scores / (threshold * 2), 1.0)
        labels = (scores > threshold).astype(int)

        result: Any = {
            "labels": labels.tolist(),
            "scores": scores.tolist(),
            "anomaly_severity": anomaly_severity.tolist(),
        }
        return result

    def _parse_input(
        self, model_input: dict[str, Any]
    ) -> tuple[pd.Series | pd.DataFrame, float]:
        if not isinstance(model_input, dict):
            raise ValueError("输入格式错误，需要 dict 类型")

        data = model_input.get("data")
        if data is None:
            raise ValueError("输入必须包含 'data' 字段")

        threshold = float(model_input.get("threshold", self.threshold))
        if threshold <= 0:
            raise ValueError("threshold 必须 > 0")

        return data, threshold

    def _to_series(self, data: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(data, pd.Series):
            return data
        if isinstance(data, pd.DataFrame):
            if "value" not in data.columns:
                raise ValueError("DataFrame 输入必须包含 value 列")
            return cast(pd.Series, data["value"])
        raise ValueError(
            f"data 必须是 pd.Series 或 pd.DataFrame，实际类型: {type(data)}"
        )
