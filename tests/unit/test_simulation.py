

import pytest
import numpy as np
from datetime import datetime, timedelta
from pytrade.data_models.option import Option, OptionDirection, OptionType
from pytrade.simulation.simulation import compute_hedge_cost, compute_hedge_value


TEST_NUMERICAL_ACCURACY = 1e-8

@pytest.mark.parametrize(
    "option_list, output",
    [
        (
            [Option(
                strike = 100,
                premium = 1.5,
                iv = 0.01,
                r = 0.01,
                option_type=OptionType.PUT,
                option_direction=OptionDirection.LONG,
                expiration_date=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                contracts=1
            )],
            150
        ),
        (
            [Option(
                strike = 100,
                premium = 1.5,
                iv = 0.01,
                r = 0.01,
                option_type=OptionType.PUT,
                option_direction=OptionDirection.LONG,
                expiration_date=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                contracts=2
            )],
            300
        ),
        (
            [
                Option(
                    strike = 100,
                    premium = 1.5,
                    iv = 0.01,
                    r = 0.01,
                    option_type=OptionType.PUT,
                    option_direction=OptionDirection.LONG,
                    expiration_date=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    contracts=2
                ),
                Option(
                    strike = 90,
                    premium = 0.5,
                    iv = 0.01,
                    r = 0.01,
                    option_type=OptionType.PUT,
                    option_direction=OptionDirection.SHORT,
                    expiration_date=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    contracts=2
                )
            ],
            200
        )
    ]
)

def test_compute_hedge_cost(option_list, output):
    result = compute_hedge_cost(option_list)
    assert result == output




@pytest.mark.parametrize(
    "option_list, output",
    [
        (
            [Option(
                strike = 100,
                premium = 1.5,
                iv = 0.01,
                r = 0.01,
                option_type=OptionType.PUT,
                option_direction=OptionDirection.LONG,
                expiration_date=(datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
                contracts=1
            )],
            63.5462792590566
        ),
        (
            [
                Option(
                    strike = 100,
                    premium = 1.5,
                    iv = 0.01,
                    r = 0.01,
                    option_type=OptionType.PUT,
                    option_direction=OptionDirection.LONG,
                    expiration_date=(datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
                    contracts=2
                ),
                Option(
                    strike = 90,
                    premium = 0.5,
                    iv = 0.01,
                    r = 0.01,
                    option_type=OptionType.PUT,
                    option_direction=OptionDirection.SHORT,
                    expiration_date=(datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
                    contracts=2
                )
            ],
            127.09255850968149
        )
    ]
)

def test_compute_hedge_value(option_list, output):
    UNDERLYING = 100
    IV_CHANGE = 0.05
    DTE_CHANGE = -30
    result = compute_hedge_value(option_list, UNDERLYING, IV_CHANGE, DTE_CHANGE)
    assert np.abs(result - output) < TEST_NUMERICAL_ACCURACY



