
import pytest
import pandas as pd
from pytrade.utils.utils import compute_stock_returns



@pytest.mark.parametrize(
    "input_df, expected_df",
    [
        (
            pd.DataFrame(
                {
                    "Close": [100, 110, 120, 130, 140, 150]
                }
            ),
            pd.DataFrame(
                {
                    "Close": [110, 120, 130, 140, 150],
                    "returns": [0.10000000000000009, 0.09090909090909083, 0.08333333333333326, 0.07692307692307687, 0.0714285714285714]
                },
                index = [1, 2, 3, 4, 5]
            )
        ),
    ]
)
def test_compute_stock_returns(input_df: pd.DataFrame, expected_df: pd.DataFrame):
    result = compute_stock_returns(input_df, 1)
    pd.testing.assert_frame_equal(result, expected_df)


