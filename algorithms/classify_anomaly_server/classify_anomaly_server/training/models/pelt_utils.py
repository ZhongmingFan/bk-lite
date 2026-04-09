"""PELT changepoint 到点级异常窗口的映射工具。"""

from typing import Iterable

import numpy as np


def breakpoints_to_changepoints(breakpoints: Iterable[int], length: int) -> list[int]:
    """将 ruptures 返回的断点转换为真实 changepoint 索引。"""
    changepoints: list[int] = []
    for breakpoint in breakpoints:
        if 0 <= breakpoint < length:
            changepoints.append(int(breakpoint))
    return changepoints


def event_window_scores(
    length: int,
    changepoints: Iterable[int],
    event_window: int,
    score: float = 1.0,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """将 changepoint 展开成边界裁剪后的对称点级分数。"""
    scores = np.zeros(length, dtype=float)
    if length <= 0:
        return scores

    for changepoint in changepoints:
        start = max(0, int(changepoint) - event_window)
        end = min(length - 1, int(changepoint) + event_window)
        scores[start : end + 1] = np.maximum(scores[start : end + 1], score)

    return scores
