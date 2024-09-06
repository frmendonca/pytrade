
import numpy as np
from typing import Union

from pytrade.utils import read_json, df_factory
from pytrade.data_models.portfolio import Portfolio
from pytrade.data_models.option import Option, OptionUnderlying
from pytrade.data_models.sequences import Sequences

from pytrade.simulation.constants import (
    PORTFOLIO_CONFIG_PATH,
    OPTIONS_CONFIG_PATH,
    SIMULATION_CONFIG_PATH
)


def compute_simulation_statistics(return_paths: np.array, num_year: int):
    compounded_returns = np.cumprod(1 + return_paths, axis = 1)
    avg_compounded_returns = np.mean(compounded_returns, axis = 0)
    median_compounded_returns = np.median(compounded_returns, axis = 0)
    pct5_compounded_returns = np.quantile(compounded_returns, axis = 0, q = 0.05)
    
    return {
        "median": median_compounded_returns[-1]**(1/num_year) - 1,
        "average": avg_compounded_returns[-1]**(1/num_year) - 1,
        "percentile_5th": pct5_compounded_returns[-1]**(1/num_year) - 1,
        "prob_neg_cagr": np.mean(compounded_returns[:,-1]**(1/num_year) - 1 < 0)
    }


class Simulator:
    def __init__(self) -> None:
        self.sim_params = read_json(SIMULATION_CONFIG_PATH)
        self.portfolio_config = read_json(PORTFOLIO_CONFIG_PATH)
        self.option_config = read_json(OPTIONS_CONFIG_PATH)
        self.model_results = []


    def simulate(self):
        # Load the portfolio
        print("Loading portfolio")
        portfolio = Portfolio(self.portfolio_config)
        portfolio.fit(kwargs=self.sim_params)

        # Loop over different option configs
        print("Loading option chains")
        option_list = []
        for ticker, chain in self.option_config.items():
            option_underlying = OptionUnderlying(ticker, kwargs=self.sim_params)
            for config in chain:
                option = Option(**config, underlying=option_underlying, kwargs=self.sim_params)
                option_list.append(option)

        # Define sequences
        sequences = Sequences(portfolio, option_list)
        sequences.fit(sim_params=self.sim_params)

        # Simulate
        nb_periods_within_year = int(np.floor(365/self.sim_params["freq"]))
        nb_draws = nb_periods_within_year*self.sim_params["nb_year"]
        
        simulated_returns = np.array(
            [
                np.random.choice(sequences.strategy_returns, size = nb_draws, replace = True)
                for _ in range(self.sim_params["nb_sim"])
            ]
        )

        statistics = compute_simulation_statistics(simulated_returns, self.sim_params["nb_year"])
        model_result = (statistics['median'], statistics['average'], statistics['percentile_5th'])
        self.model_results.append(model_result)


        np.median(np.exp(np.mean(np.log(1 + simulated_returns), axis = 1)) - 1)