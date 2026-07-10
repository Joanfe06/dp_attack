from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def calculate_responsive_demand(
    baseline_demand_kw: np.ndarray,
    price_eur_per_kwh: np.ndarray,
    reference_price: float,
    elasticity: float,
    minimum_response_factor: float = 0.60,
    maximum_response_factor: float = 1.40,
) -> np.ndarray:
    """
    Calculate demand after a consumer responds to electricity prices.

    A price above the reference price reduces demand.
    A price below the reference price increases demand.

    This is a deliberately simple model. It does not yet shift energy
    between time intervals.
    """
    if baseline_demand_kw.shape != price_eur_per_kwh.shape:
        raise ValueError("Demand and price arrays must have the same length.")

    if reference_price <= 0:
        raise ValueError("The reference price must be positive.")

    if elasticity < 0:
        raise ValueError("Elasticity cannot be negative.")

    relative_price_change = (
        price_eur_per_kwh - reference_price
    ) / reference_price

    response_factor = 1.0 - elasticity * relative_price_change

    response_factor = np.clip(
        response_factor,
        minimum_response_factor,
        maximum_response_factor,
    )

    responsive_demand_kw = baseline_demand_kw * response_factor

    return responsive_demand_kw


def calculate_energy_kwh(
    demand_kw: np.ndarray,
    interval_duration_hours: float = 1.0,
) -> float:
    """Calculate total energy consumed over all intervals."""
    return float(np.sum(demand_kw * interval_duration_hours))


def calculate_bill_eur(
    demand_kw: np.ndarray,
    price_eur_per_kwh: np.ndarray,
    interval_duration_hours: float = 1.0,
) -> float:
    """Calculate the total electricity bill."""
    return float(
        np.sum(
            demand_kw
            * interval_duration_hours
            * price_eur_per_kwh
        )
    )


def main() -> None:
    # We simulate 24 one-hour intervals.
    hours = np.arange(24)
    interval_duration_hours = 1.0

    # Baseline household demand in kW.
    # Demand is relatively low overnight and high in the evening.
    baseline_demand_kw = np.array(
        [
            1.2, 1.1, 1.0, 0.9, 0.9, 1.0,
            1.3, 1.8, 2.2, 2.0, 1.7, 1.6,
            1.5, 1.6, 1.7, 1.9, 2.3, 2.8,
            3.0, 2.7, 2.3, 1.9, 1.6, 1.4,
        ],
        dtype=float,
    )

    # A constant tariff of 0.20 €/kWh.
    flat_price_eur_per_kwh = np.full(24, 0.20)

    # A dynamic tariff.
    # It is cheap overnight and expensive during the evening peak.
    dynamic_price_eur_per_kwh = np.array(
        [
            0.14, 0.14, 0.14, 0.14, 0.14, 0.16,
            0.18, 0.22, 0.26, 0.24, 0.20, 0.18,
            0.18, 0.18, 0.20, 0.22, 0.28, 0.36,
            0.40, 0.34, 0.28, 0.22, 0.18, 0.16,
        ],
        dtype=float,
    )

    # The consumer considers 0.20 €/kWh a normal price.
    reference_price = 0.20

    # Price elasticity controls how strongly demand reacts.
    elasticity = 0.25

    responsive_demand_kw = calculate_responsive_demand(
        baseline_demand_kw=baseline_demand_kw,
        price_eur_per_kwh=dynamic_price_eur_per_kwh,
        reference_price=reference_price,
        elasticity=elasticity,
    )

    baseline_energy_kwh = calculate_energy_kwh(
        baseline_demand_kw,
        interval_duration_hours,
    )

    responsive_energy_kwh = calculate_energy_kwh(
        responsive_demand_kw,
        interval_duration_hours,
    )

    flat_price_bill_eur = calculate_bill_eur(
        baseline_demand_kw,
        flat_price_eur_per_kwh,
        interval_duration_hours,
    )

    dynamic_bill_without_response_eur = calculate_bill_eur(
        baseline_demand_kw,
        dynamic_price_eur_per_kwh,
        interval_duration_hours,
    )

    dynamic_bill_with_response_eur = calculate_bill_eur(
        responsive_demand_kw,
        dynamic_price_eur_per_kwh,
        interval_duration_hours,
    )

    print("=== Simulation results ===")
    print(f"Baseline energy: {baseline_energy_kwh:.2f} kWh")
    print(f"Responsive energy: {responsive_energy_kwh:.2f} kWh")
    print()
    print(f"Baseline peak demand: {np.max(baseline_demand_kw):.2f} kW")
    print(f"Responsive peak demand: {np.max(responsive_demand_kw):.2f} kW")
    print()
    print(f"Bill with flat price: €{flat_price_bill_eur:.2f}")
    print(
        "Bill with dynamic price and no response: "
        f"€{dynamic_bill_without_response_eur:.2f}"
    )
    print(
        "Bill with dynamic price and response: "
        f"€{dynamic_bill_with_response_eur:.2f}"
    )

    # Demand plot.
    plt.figure(figsize=(10, 5))
    plt.plot(
        hours,
        baseline_demand_kw,
        marker="o",
        label="Baseline demand",
    )
    plt.plot(
        hours,
        responsive_demand_kw,
        marker="o",
        label="Price-responsive demand",
    )
    plt.xlabel("Hour")
    plt.ylabel("Demand (kW)")
    plt.title("Baseline and Price-Responsive Demand")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "step_01_demand_response.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Price plot.
    plt.figure(figsize=(10, 4))
    plt.step(
        hours,
        dynamic_price_eur_per_kwh,
        where="mid",
        label="Dynamic price",
    )
    plt.axhline(
        reference_price,
        linestyle="--",
        label="Reference price",
    )
    plt.xlabel("Hour")
    plt.ylabel("Price (€/kWh)")
    plt.title("Dynamic Electricity Price")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "step_01_dynamic_price.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Figures saved:")
    print("  step_01_demand_response.png")
    print("  step_01_dynamic_price.png")


if __name__ == "__main__":
    main()