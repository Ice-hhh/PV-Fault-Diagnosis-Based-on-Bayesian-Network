from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


POWER_COLUMNS = [f"p{i}" for i in range(1, 97)]
TRAIN_PREFIX = "A榜-训练集_分布式光伏发电预测"
TEST_PREFIX = "A榜-测试集_分布式光伏发电预测"


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    test_template: pd.DataFrame
    feature_names: list[str]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="gbk")


def _load_split(data_dir: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    info = _read_csv(data_dir / f"{prefix}_基本信息.csv")
    weather = _read_csv(data_dir / f"{prefix}_气象变量数据.csv")
    power = _read_csv(data_dir / f"{prefix}_实际功率数据.csv")
    return info, weather, power


def _expand_power_rows(power: pd.DataFrame, include_targets: bool) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row_id, row in power.reset_index(drop=True).iterrows():
        base_time = pd.to_datetime(row["时间"])
        for step, col in enumerate(POWER_COLUMNS):
            record = {
                "row_id": row_id,
                "step": step,
                "光伏用户编号": row["光伏用户编号"],
                "综合倍率": row["综合倍率"],
                "时间": base_time + pd.Timedelta(minutes=15 * step),
            }
            if include_targets:
                record["target"] = row[col]
            records.append(record)
    return pd.DataFrame.from_records(records)


def _make_features(
    expanded_power: pd.DataFrame,
    weather: pd.DataFrame,
    info: pd.DataFrame,
    include_targets: bool,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    weather = weather.copy()
    weather["时间"] = pd.to_datetime(weather["时间"])

    frame = expanded_power.merge(weather, on=["光伏用户编号", "时间"], how="left")
    frame = frame.merge(info, on="光伏用户编号", how="left")
    frame["月份"] = frame["时间"].dt.month
    frame["日期"] = frame["时间"].dt.day
    frame["小时"] = frame["时间"].dt.hour
    frame["分钟"] = frame["时间"].dt.minute

    y = None
    if include_targets:
        y = frame.sort_values(["row_id", "step"])["target"].to_numpy().reshape(-1, 96)

    drop_cols = {"row_id", "step", "时间", "光伏用户名称", "target"}
    features = frame.drop(columns=[c for c in drop_cols if c in frame.columns])
    features = pd.get_dummies(features, columns=["光伏用户编号"], dtype=float)
    features = features.sort_index()
    return features, y


def prepare_data(data_dir: Path | str = "data") -> PreparedData:
    data_dir = Path(data_dir)
    train_info, train_weather, train_power = _load_split(data_dir, TRAIN_PREFIX)
    test_info, test_weather, test_power = _load_split(data_dir, TEST_PREFIX)

    train_features, y_train = _make_features(
        _expand_power_rows(train_power, include_targets=True),
        train_weather,
        train_info,
        include_targets=True,
    )
    test_features, _ = _make_features(
        _expand_power_rows(test_power, include_targets=False),
        test_weather,
        test_info,
        include_targets=False,
    )

    train_features, test_features = train_features.align(test_features, join="outer", axis=1, fill_value=0)
    train_features = train_features.apply(pd.to_numeric, errors="coerce")
    test_features = test_features.apply(pd.to_numeric, errors="coerce")

    fill_values = train_features.median(numeric_only=True).fillna(0)
    train_features = train_features.fillna(fill_values).fillna(0)
    test_features = test_features.fillna(fill_values).fillna(0)

    scaler = MinMaxScaler()
    X_train_flat = scaler.fit_transform(train_features)
    X_test_flat = scaler.transform(test_features)

    n_features = X_train_flat.shape[1]
    X_train = X_train_flat.reshape(-1, 96, n_features)
    X_test = X_test_flat.reshape(-1, 96, n_features)

    assert y_train is not None
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        test_template=test_power.copy(),
        feature_names=list(train_features.columns),
    )


def flatten_daily_features(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def build_submission(test_template: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    output = test_template.copy()
    predictions = np.asarray(predictions).reshape(len(output), 96)
    output[POWER_COLUMNS] = predictions
    return output.round(4)
