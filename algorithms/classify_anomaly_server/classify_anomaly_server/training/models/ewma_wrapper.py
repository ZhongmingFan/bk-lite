"""EWMA 模型的 MLflow 推理包装器。"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from mlflow.pyfunc.model import PythonModel


class EWMAWrapper(PythonModel):
    """将 EWMA 计算结果映射为统一 serving 输出。

    返回 {labels, scores, anomaly_severity}，
    可选地通过 runtime threshold 覆盖训练时阈值。
    """

    def __init__(
        self,
        alpha: float,
        scale_window: int,
        threshold: float,
        n_consecutive: int,
        scale_method: str,
        severity_cap: float,
    ) -> None:
        self.alpha = alpha
        self.scale_window = scale_window
        self.threshold = threshold
        self.n_consecutive = n_consecutive
        self.scale_method = scale_method
        self.severity_cap = severity_cap

    def predict(self, context, model_input, params=None):
        """推理接口。

        Args:
            context: MLflow context
            model_input: dict {'data': pd.Series | pd.DataFrame, 'threshold': float (可选)}

        Returns:
            dict {labels, scores, anomaly_severity}
        """
        data, threshold = self._parse_input(model_input)
        series = self._to_series(data)
        values = series.to_numpy(dtype=float)

        from classify_anomaly_server.training.models.ewma_model import (
            _apply_consecutive_filter,
            _ewma_scores,
        )

        scores = _ewma_scores(values, self.alpha, self.scale_window)
        raw_labels = (scores > threshold).astype(int)
        labels = _apply_consecutive_filter(raw_labels, self.n_consecutive)
        anomaly_severity = np.clip(scores / self.severity_cap, 0.0, 1.0)

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

        threshold = model_input.get("threshold", self.threshold)
        if threshold is not None:
            threshold = float(threshold)
            if threshold <= 0:
                raise ValueError("threshold 必须 > 0")
        else:
            threshold = self.threshold

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
