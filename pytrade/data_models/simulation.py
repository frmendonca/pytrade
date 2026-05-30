
import numpy as np
from typing import Any
from dataclasses import dataclass
from pytrade.simulation.utils import block_resample
from pytrade.simulation.utils import block_resample_joint

@dataclass
class BaseSimulationResults:
    simulation_output: dict[str, Any]
    


class BaseSimulationModel:

    def block_bootstrap_returns(
        self,
        original_sequence: np.array,
        block_length: int = 30,
        resample_sequence_length: int = 30,
        seed: int = 12345
    ) -> np.array:
        """Generate a single bootstrapped return sequence."""
        
        return block_resample(
            original_sequence,
            block_length=block_length,
            resample_sequence_length=resample_sequence_length,
            seed=seed
        )
    

    def block_resample_joint(
        self,
        sequences: list[np.ndarray],
        block_length: int,
        resample_sequence_length: int,
        seed: int | None = None
    ) -> list[np.ndarray]:
        
        return block_resample_joint(
            sequences,
            block_length,
            resample_sequence_length,
            seed
        )
    

    def generate_bootstrap_blocks(
        self,
        original_sequence: np.array,
        seq_len: int,
        num_resamples: int = 1000,
        block_length: int = 10,
        target_ann_vol: float | None = None,
        seed: int | None = None
    ) -> np.array:
        """
        Generate a (num_resamples, seq_len) matrix of bootstrapped return paths
        using the Stationary Block Bootstrap.

        Each row is an independent path of `seq_len` daily returns, constructed
        by stitching geometrically-distributed blocks drawn from `original_sequence`
        with circular wrap-around indexing (Politis & Romano, 1994).

        Parameters
        ----------
        original_sequence : 1-D array of historical daily returns.
        seq_len           : Path length in days — should match the option strategy's DTE.
        num_resamples     : Number of simulation paths to generate.
        block_length      : Target mean block length. Controls how much short-term
                            autocorrelation (momentum/mean-reversion) is preserved.
                            Rule of thumb: 5-15 days for daily equity return series.
        target_ann_vol    : If provided, each path is rescaled so its daily vol equals
                                    target_ann_vol / sqrt(252). Set to the ATM IV of the option
                                    strategy to ensure paths are consistent with current option
                                    pricing. None → preserve historical realized vol as-is
                                    (paths may be calmer or wilder than the IV implies).
        seed              : Master RNG seed for full reproducibility.
        """

        rng = np.random.default_rng(seed)
        hist_daily_vol = np.std(original_sequence)

        paths = np.empty((num_resamples, seq_len))
        for i in range(num_resamples):
            path = block_resample(
                original_sequence,
                block_length=block_length,
                resample_sequence_length=seq_len,
                seed=int(rng.integers(0, 2**31))
            )

            # --- Vol scaling ------------------------------------------------
            # Rescale so the path's daily vol matches target_ann_vol / sqrt(252).
            # Without this, a mismatch between historical realized vol and current
            # IV directly biases the POP estimate — calmer history → overstated
            # POP for credit strategies; wilder history → understated POP.
            if target_ann_vol is not None:
                target_daily_vol = target_ann_vol / np.sqrt(252)
                path = path * (target_daily_vol / hist_daily_vol)

            paths[i]  = path.flatten()

        return paths
    