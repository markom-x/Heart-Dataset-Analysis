import pandas as pd
import numpy as np
from project import create_table, HeartDatasetLoader  

def test_create_table():
    cleveland = pd.DataFrame({'age': [25, 35, 45], 'chol': [200, 220, 240]})
    california = pd.DataFrame({'age': [32, 42, 52], 'chol': [180, 210, 230]})
    hungarian = pd.DataFrame({'age': [27, 37, 47], 'chol': [190, 215, 250]})
    switzerland = pd.DataFrame({'age': [30, 40, 50], 'chol': [0, 0, 0]}) 
    
    table = create_table(cleveland, california, hungarian, switzerland)
    assert set(table.index) == {'Cleveland', 'San Francisco', 'Budapest'}
    assert '30-39' in table.columns
    assert np.isclose(table.loc['Cleveland', '20-29'], 200)
    assert np.isnan(table.loc['San Francisco', '20-29'])

def test_HeartDatasetLoader(tmp_path):
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    for name in ["processed.cleveland.data", "processed.va.data", "processed.hungarian.data", "processed.switzerland.data"]:
        pd.DataFrame([[25,1,1,120,200,0,0,150,0,1.0,2,0,3,0]], columns=columns).to_csv(tmp_path / name, index=False, header=False)
    loader = HeartDatasetLoader(str(tmp_path))
    df = loader.get("cleveland")
    assert isinstance(df, pd.DataFrame)
    assert "age" in df.columns
    assert df.iloc[0]["chol"] == 200

def test_get_city_color():
    assert HeartDatasetLoader.get_city_color("Cleveland") == "#eb4034"
    assert HeartDatasetLoader.get_city_color("Unknown") == "#000000"
