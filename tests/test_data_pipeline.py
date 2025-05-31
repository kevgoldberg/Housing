import pandas as pd
from data.pipeline import load_data, clean_data, save_data, create_visualizations


def test_clean_data_removes_na(tmp_path):
    data = pd.DataFrame({'A': [1, 2, None], 'SalePrice': [100, 200, 300]})
    cleaned = clean_data(data)
    assert cleaned.isna().sum().sum() == 0

    out_csv = tmp_path / "clean.csv"
    save_data(cleaned, out_csv)
    assert out_csv.exists()

    viz_dir = tmp_path / "viz"
    create_visualizations(cleaned, viz_dir)
    assert (viz_dir / "SalePrice_distribution.png").exists()
    assert (viz_dir / "correlation_matrix.png").exists()


def test_load_data(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({'A': [1, 2], 'B': [3, 4]}).to_csv(csv_path, index=False)
    df = load_data(csv_path)
    assert df.shape == (2, 2)

