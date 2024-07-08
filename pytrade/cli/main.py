

import typer
from typing import Iterable
from pytrade.simulation.simulation import Simulator

cli_app = typer.Typer(name="pytrade")

@cli_app.command()
def dummy() -> None:
    # If not at least 2 commands, typer forces to drop command name from CLI invocation
    # So it's `path-to-app` --option rather than `path-to-app command-x --option`
    # Remove dummy once we have two commands
    return None


@cli_app.command()
def simulate_hedged_path(
    number_contracts: int = typer.Option(help="Nummber of put options", default=0)
) -> None:
    
    simulator = Simulator()
    simulator.simulate(number_contracts=number_contracts)

    typer.echo("Done")