import typer
from typing import Iterable
from pytrade.simulation.simulation import Simulator
from pytrade.utils import df_factory


cli_app = typer.Typer(name="pytrade")


@cli_app.command()
def dummy() -> None:
    # If not at least 2 commands, typer forces to drop command name from CLI invocation
    # So it's `path-to-app` --option rather than `path-to-app command-x --option`
    # Remove dummy once we have two commands
    return None


@cli_app.command()
def simulate_hedged_path() -> None:

    simulator = Simulator()
    simulator.simulate()

    model_results_df_factory = df_factory(
        columns=["geom_return", "avg_return", "percentile_5"]
    )
    df_results = model_results_df_factory(simulator.model_results)
    df_results.to_csv("model_result.csv", index=False)
    typer.echo("Done")


if __name__ == "__main__":
    cli_app()
