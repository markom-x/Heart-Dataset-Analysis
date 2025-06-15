import pandas as pd
import numpy as np
import pytest


from Bio_analyzer import HeartDatasetLoader, show_map, gaussian, create_table

dfs = [
    pd.DataFrame({
        "age": [50+i, 60+i, 70+i],
        "trestbps": [120+i, 130+i, np.nan]
    })
    for i in range(1, 5)
]

df1, df2, df3, df4 = dfs


def test_create_table():
    assert create_table() ==...





"""
def test_gaussian_basic():
    df = make_test_df()
    result = gaussian(df, df, df, df, loader)
    assert isinstance(result, dict)
    assert "San Francisco" in result
    mean, std = result["San Francisco"]
    assert np.isclose(mean, 125.0)
    assert std > 0


def test_loader_get(tmp_path):
    csv_content = "50,1,2,120,220,0,0,150,0,2.3,1,0,3,0\n"
    file_path = tmp_path / "processed.cleveland.data"
    file_path.write_text(csv_content)
    # Write empty files for the others to satisfy init
    for name in ["va", "hungarian", "switzerland"]:
        (tmp_path / f"processed.{name}.data").write_text(csv_content)
    loader = HeartDatasetLoader(str(tmp_path))
    df = loader.get("cleveland")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", 
        "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    assert df.iloc[0]["age"] == 50
"""