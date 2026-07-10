from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def calculate_generation_cost_eur_per_hour(
    demand_mw: np.ndarray,
    quadratic_coefficient: float,
    linear_coefficient: float,
) -> np.ndarray:
    """
    Calculate hourly generation cost.

    Cost model:

        C(D) = a D² + b D

    where:
        D is demand in MW,
        C is generation cost in euros per hour.
    """
    if np.any(demand_mw < 0):
        raise ValueError("Demand cannot be negative.")

    if quadratic_coefficient < 0:
        raise ValueError(
            "The quadratic coefficient cannot be negative."
        )

    if linear_coefficient < 0:
        raise ValueError(
            "The linear coefficient cannot be negative."
        )

    return (
        quadratic_coefficient * demand_mw**2
        + linear_coefficient * demand_mw
    )


def calculate_marginal_price_eur_per_mwh(
    demand_mw: np.ndarray,
    quadratic_coefficient: float,
    linear_coefficient: float,
) -> np.ndarray:
    """
    Calculate the marginal generation cost.

    If:

        C(D) = a D² + b D

    then:

        dC/dD = 2aD + b

    The result is expressed in €/MWh.
    """
    return (
        2.0 * quadratic_coefficient * demand_mw
        + linear_coefficient
    )


def eur_per_mwh_to_eur_per_kwh(
    price_eur_per_mwh: np.ndarray,
) -> np.ndarray:
    """Convert €/MWh to €/kWh."""
    return price_eur_per_mwh / 1000.0


def main() -> None:
    hours = np.arange(24)

    number_of_households = 10_000

    fixed_demand_per_household_kw = np.array(
        [
            1.2, 1.1, 1.0, 0.9, 0.9, 1.0,
            1.3, 1.8, 2.2, 2.0, 1.7, 1.6,
            1.5, 1.6, 1.7, 1.9, 2.3, 2.8,
            3.0, 2.7, 2.3, 1.9, 1.6, 1.4,
        ],
        dtype=float,
    )

    # Aggregate household demand in kW.
    aggregate_fixed_demand_kw = (
        fixed_demand_per_household_kw
        * number_of_households
    )

    # Convert kW to MW.
    aggregate_fixed_demand_mw = (
        aggregate_fixed_demand_kw / 1000.0
    )

    # Generation cost parameters.
    #
    # These values were chosen so that the marginal price is
    # approximately:
    #
    #   140 €/MWh at 9 MW
    #   400 €/MWh at 30 MW
    #
    # They are illustrative values, not real market estimates.
    quadratic_coefficient = 6.190476
    linear_coefficient = 28.571432

    generation_cost_eur_per_hour = (
        calculate_generation_cost_eur_per_hour(
            demand_mw=aggregate_fixed_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
        )
    )

    dynamic_price_eur_per_mwh = (
        calculate_marginal_price_eur_per_mwh(
            demand_mw=aggregate_fixed_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
        )
    )

    dynamic_price_eur_per_kwh = (
        eur_per_mwh_to_eur_per_kwh(
            dynamic_price_eur_per_mwh
        )
    )

    print("=== Dynamic price generated from demand ===")
    print()
    print(
        "Hour | Demand | Generation cost | "
        "Marginal price | Consumer price"
    )
    print(
        "     |   MW   |      €/h        | "
        "    €/MWh       |     €/kWh"
    )
    print("-" * 78)

    for hour in hours:
        print(
            f"{hour:02d}:00 | "
            f"{aggregate_fixed_demand_mw[hour]:6.2f} | "
            f"{generation_cost_eur_per_hour[hour]:15.2f} | "
            f"{dynamic_price_eur_per_mwh[hour]:14.2f} | "
            f"{dynamic_price_eur_per_kwh[hour]:10.3f}"
        )

    minimum_price_hour = int(
        np.argmin(dynamic_price_eur_per_kwh)
    )

    maximum_price_hour = int(
        np.argmax(dynamic_price_eur_per_kwh)
    )

    print()
    print("Price summary")
    print(
        f"  Minimum price: "
        f"€{dynamic_price_eur_per_kwh[minimum_price_hour]:.3f}/kWh "
        f"at {minimum_price_hour:02d}:00"
    )
    print(
        f"  Maximum price: "
        f"€{dynamic_price_eur_per_kwh[maximum_price_hour]:.3f}/kWh "
        f"at {maximum_price_hour:02d}:00"
    )

    output_directory = Path("results")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        aggregate_fixed_demand_mw,
        marker="o",
    )

    plt.xlabel("Hour")
    plt.ylabel("Forecast demand (MW)")
    plt.title("Forecast Aggregate Demand")
    plt.xticks(hours)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_06_forecast_demand.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(figsize=(10, 5))

    plt.step(
        hours,
        dynamic_price_eur_per_kwh,
        where="mid",
    )

    plt.xlabel("Hour")
    plt.ylabel("Dynamic price (€/kWh)")
    plt.title("Dynamic Price Generated from Forecast Demand")
    plt.xticks(hours)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_06_generated_price.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Figures saved:")
    print("  results/step_06_forecast_demand.png")
    print("  results/step_06_generated_price.png")


if __name__ == "__main__":
    main()