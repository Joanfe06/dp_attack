from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend because your environment cannot open windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def get_available_hours(
    arrival_hour: int,
    departure_hour: int,
    number_of_hours: int = 24,
) -> np.ndarray:
    """
    Return the hourly intervals during which the EV is connected.

    Examples
    --------
    Arrival at 18:00 and departure at 07:00:

        [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]

    The departure hour itself is not included. A departure at 07:00
    means that charging must be complete by the beginning of hour 7.
    """
    if not 0 <= arrival_hour < number_of_hours:
        raise ValueError("Arrival hour is outside the valid range.")

    if not 0 <= departure_hour < number_of_hours:
        raise ValueError("Departure hour is outside the valid range.")

    if arrival_hour == departure_hour:
        # Interpret equal arrival and departure as availability all day.
        return np.arange(number_of_hours)

    if arrival_hour < departure_hour:
        # Example: arrival at 10:00, departure at 17:00.
        return np.arange(arrival_hour, departure_hour)

    # Example: arrival at 18:00, departure at 07:00.
    # Availability crosses midnight.
    evening_hours = np.arange(arrival_hour, number_of_hours)
    morning_hours = np.arange(0, departure_hour)

    return np.concatenate((evening_hours, morning_hours))


def create_ev_schedule(
    price_eur_per_kwh: np.ndarray,
    available_hours: np.ndarray,
    required_energy_kwh: float,
    maximum_charging_power_kw: float,
    interval_duration_hours: float = 1.0,
    optimize_for_price: bool = False,
) -> np.ndarray:
    """
    Create an EV charging schedule.

    If optimize_for_price is False, the EV charges immediately after
    arriving.

    If optimize_for_price is True, the EV charges during the cheapest
    available intervals.
    """
    if required_energy_kwh <= 0:
        raise ValueError("Required energy must be positive.")

    if maximum_charging_power_kw <= 0:
        raise ValueError("Maximum charging power must be positive.")

    if interval_duration_hours <= 0:
        raise ValueError("Interval duration must be positive.")

    number_of_intervals = len(price_eur_per_kwh)

    if np.any(available_hours < 0) or np.any(
        available_hours >= number_of_intervals
    ):
        raise ValueError("An available hour is outside the price array.")

    maximum_possible_energy_kwh = (
        len(available_hours)
        * maximum_charging_power_kw
        * interval_duration_hours
    )

    if required_energy_kwh > maximum_possible_energy_kwh:
        raise ValueError(
            "The EV cannot receive the required energy before departure. "
            f"Required: {required_energy_kwh:.2f} kWh; "
            f"maximum possible: {maximum_possible_energy_kwh:.2f} kWh."
        )

    charging_schedule_kw = np.zeros(number_of_intervals)

    if optimize_for_price:
        # Sort available intervals by electricity price.
        #
        # kind="stable" preserves the original order when two hours have
        # the same price.
        sorting_indices = np.argsort(
            price_eur_per_kwh[available_hours],
            kind="stable",
        )
        ordered_hours = available_hours[sorting_indices]
    else:
        # The available-hours array already begins at the arrival time.
        # Therefore, this represents immediate charging.
        ordered_hours = available_hours

    remaining_energy_kwh = required_energy_kwh

    for hour in ordered_hours:
        maximum_interval_energy_kwh = (
            maximum_charging_power_kw
            * interval_duration_hours
        )

        energy_this_interval_kwh = min(
            remaining_energy_kwh,
            maximum_interval_energy_kwh,
        )

        charging_power_kw = (
            energy_this_interval_kwh
            / interval_duration_hours
        )

        charging_schedule_kw[hour] = charging_power_kw
        remaining_energy_kwh -= energy_this_interval_kwh

        if remaining_energy_kwh <= 1e-9:
            break

    if remaining_energy_kwh > 1e-9:
        raise RuntimeError(
            "The scheduling algorithm did not allocate all required energy."
        )

    return charging_schedule_kw


def calculate_energy_kwh(
    power_kw: np.ndarray,
    interval_duration_hours: float = 1.0,
) -> float:
    """Calculate energy from a power profile."""
    return float(
        np.sum(power_kw * interval_duration_hours)
    )


def calculate_bill_eur(
    demand_kw: np.ndarray,
    price_eur_per_kwh: np.ndarray,
    interval_duration_hours: float = 1.0,
) -> float:
    """Calculate the electricity bill."""
    if demand_kw.shape != price_eur_per_kwh.shape:
        raise ValueError(
            "Demand and price arrays must have the same shape."
        )

    return float(
        np.sum(
            demand_kw
            * price_eur_per_kwh
            * interval_duration_hours
        )
    )


def main() -> None:
    hours = np.arange(24)
    interval_duration_hours = 1.0

    # This is now treated as fixed household demand.
    # It does not respond to the electricity price.
    fixed_demand_kw = np.array(
        [
            1.2, 1.1, 1.0, 0.9, 0.9, 1.0,
            1.3, 1.8, 2.2, 2.0, 1.7, 1.6,
            1.5, 1.6, 1.7, 1.9, 2.3, 2.8,
            3.0, 2.7, 2.3, 1.9, 1.6, 1.4,
        ],
        dtype=float,
    )

    # The price is an input to the simulation.
    #
    # We are not yet calculating the price from demand or market
    # conditions. We will introduce that feedback later.
    dynamic_price_eur_per_kwh = np.array(
        [
            0.14, 0.14, 0.14, 0.14, 0.14, 0.16,
            0.18, 0.22, 0.26, 0.24, 0.20, 0.18,
            0.18, 0.18, 0.20, 0.22, 0.28, 0.36,
            0.40, 0.34, 0.28, 0.22, 0.18, 0.16,
        ],
        dtype=float,
    )
    
    # The legitimate price remains unchanged in the electricity system.
    legitimate_price_eur_per_kwh = dynamic_price_eur_per_kwh.copy()

    # The attacker modifies the price signal received by the EV.
    attacked_price_eur_per_kwh = legitimate_price_eur_per_kwh.copy()

    attacked_hour = 18
    attacked_price_eur_per_kwh[attacked_hour] = 0.05

    # EV parameters.
    arrival_hour = 18
    departure_hour = 7
    required_energy_kwh = 12.0
    maximum_charging_power_kw = 3.6

    available_hours = get_available_hours(
        arrival_hour=arrival_hour,
        departure_hour=departure_hour,
    )

    # Strategy 1: begin charging immediately at 18:00.
    immediate_ev_charging_kw = create_ev_schedule(
        price_eur_per_kwh=dynamic_price_eur_per_kwh,
        available_hours=available_hours,
        required_energy_kwh=required_energy_kwh,
        maximum_charging_power_kw=maximum_charging_power_kw,
        interval_duration_hours=interval_duration_hours,
        optimize_for_price=False,
    )

    # Strategy 2: charge during the cheapest available intervals.
    legitimate_price_aware_ev_charging_kw = create_ev_schedule(
        price_eur_per_kwh=legitimate_price_eur_per_kwh,
        available_hours=available_hours,
        required_energy_kwh=required_energy_kwh,
        maximum_charging_power_kw=maximum_charging_power_kw,
        interval_duration_hours=interval_duration_hours,
        optimize_for_price=True,
    )
    attacked_ev_charging_kw = create_ev_schedule(
        price_eur_per_kwh=attacked_price_eur_per_kwh,
        available_hours=available_hours,
        required_energy_kwh=required_energy_kwh,
        maximum_charging_power_kw=maximum_charging_power_kw,
        interval_duration_hours=interval_duration_hours,
        optimize_for_price=True,
    )

    total_immediate_demand_kw = (
        fixed_demand_kw
        + immediate_ev_charging_kw
    )

    total_price_aware_demand_kw = (
        fixed_demand_kw
        + legitimate_price_aware_ev_charging_kw
    )
    
    total_legitimate_price_aware_demand_kw = (
        fixed_demand_kw
        + legitimate_price_aware_ev_charging_kw
    )

    total_attacked_demand_kw = (
        fixed_demand_kw
        + attacked_ev_charging_kw
    )

    fixed_energy_kwh = calculate_energy_kwh(
        fixed_demand_kw,
        interval_duration_hours,
    )

    immediate_ev_energy_kwh = calculate_energy_kwh(
        immediate_ev_charging_kw,
        interval_duration_hours,
    )

    price_aware_ev_energy_kwh = calculate_energy_kwh(
        legitimate_price_aware_ev_charging_kw,
        interval_duration_hours,
    )

    fixed_demand_bill_eur = calculate_bill_eur(
        fixed_demand_kw,
        dynamic_price_eur_per_kwh,
        interval_duration_hours,
    )

    immediate_total_bill_eur = calculate_bill_eur(
        total_immediate_demand_kw,
        dynamic_price_eur_per_kwh,
        interval_duration_hours,
    )

    price_aware_total_bill_eur = calculate_bill_eur(
        total_price_aware_demand_kw,
        dynamic_price_eur_per_kwh,
        interval_duration_hours,
    )

    immediate_ev_cost_eur = calculate_bill_eur(
        immediate_ev_charging_kw,
        dynamic_price_eur_per_kwh,
        interval_duration_hours,
    )

    price_aware_ev_cost_eur = calculate_bill_eur(
        legitimate_price_aware_ev_charging_kw,
        dynamic_price_eur_per_kwh,
        interval_duration_hours,
    )
    
    legitimate_ev_cost_eur = calculate_bill_eur(
        legitimate_price_aware_ev_charging_kw,
        legitimate_price_eur_per_kwh,
        interval_duration_hours,
    )

    attacked_ev_perceived_cost_eur = calculate_bill_eur(
        attacked_ev_charging_kw,
        attacked_price_eur_per_kwh,
        interval_duration_hours,
    )

    attacked_ev_real_cost_eur = calculate_bill_eur(
        attacked_ev_charging_kw,
        legitimate_price_eur_per_kwh,
        interval_duration_hours,
    )

    print("=== EV charging simulation ===")
    print(f"Available hours: {available_hours.tolist()}")
    print()

    print("Energy")
    print(f"  Fixed household energy: {fixed_energy_kwh:.2f} kWh")
    print(f"  Required EV energy: {required_energy_kwh:.2f} kWh")
    print(
        "  Immediate EV energy: "
        f"{immediate_ev_energy_kwh:.2f} kWh"
    )
    print(
        "  Price-aware EV energy: "
        f"{price_aware_ev_energy_kwh:.2f} kWh"
    )
    print()

    print("Peak demand")
    print(
        "  Fixed demand only: "
        f"{np.max(fixed_demand_kw):.2f} kW"
    )
    print(
        "  With immediate charging: "
        f"{np.max(total_immediate_demand_kw):.2f} kW"
    )
    print(
        "  With price-aware charging: "
        f"{np.max(total_price_aware_demand_kw):.2f} kW"
    )
    print()

    print("Cost")
    print(
        "  Fixed household bill: "
        f"€{fixed_demand_bill_eur:.2f}"
    )
    print(
        "  Immediate EV charging cost: "
        f"€{immediate_ev_cost_eur:.2f}"
    )
    print(
        "  Price-aware EV charging cost: "
        f"€{price_aware_ev_cost_eur:.2f}"
    )
    print(
        "  Total bill with immediate charging: "
        f"€{immediate_total_bill_eur:.2f}"
    )
    print(
        "  Total bill with price-aware charging: "
        f"€{price_aware_total_bill_eur:.2f}"
    )
    print()

    print("Hourly details")
    print(
        "Hour | Price | Fixed | Immediate EV | "
        "Price-aware EV | Total price-aware"
    )
    print("-" * 78)

    for hour in hours:
        print(
            f"{hour:02d}:00 | "
            f"{dynamic_price_eur_per_kwh[hour]:5.2f} | "
            f"{fixed_demand_kw[hour]:5.2f} | "
            f"{immediate_ev_charging_kw[hour]:12.2f} | "
            f"{legitimate_price_aware_ev_charging_kw[hour]:14.2f} | "
            f"{total_price_aware_demand_kw[hour]:17.2f}"
        )

    output_directory = Path("results")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Plot 1: electricity price.
    plt.figure(figsize=(10, 4))
    plt.step(
        hours,
        dynamic_price_eur_per_kwh,
        where="mid",
        label="Dynamic price",
    )
    plt.xlabel("Hour")
    plt.ylabel("Price (€/kWh)")
    plt.title("Dynamic Electricity Price")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory / "step_02_price.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 2: EV charging schedules.
    plt.figure(figsize=(10, 5))
    plt.step(
        hours,
        immediate_ev_charging_kw,
        where="mid",
        label="Immediate EV charging",
    )
    plt.step(
        hours,
        legitimate_price_aware_ev_charging_kw,
        where="mid",
        label="Price-aware EV charging",
    )
    plt.xlabel("Hour")
    plt.ylabel("EV charging power (kW)")
    plt.title("EV Charging Strategies")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory / "step_02_ev_schedules.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 3: total household demand.
    plt.figure(figsize=(10, 5))
    plt.plot(
        hours,
        fixed_demand_kw,
        marker="o",
        label="Fixed demand only",
    )
    plt.plot(
        hours,
        total_immediate_demand_kw,
        marker="o",
        label="Fixed + immediate EV",
    )
    plt.plot(
        hours,
        total_price_aware_demand_kw,
        marker="o",
        label="Fixed + price-aware EV",
    )
    plt.xlabel("Hour")
    plt.ylabel("Total demand (kW)")
    plt.title("Effect of EV Charging on Total Demand")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory / "step_02_total_demand.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print()
    print("Figures saved in the results directory:")
    print("  results/step_02_price.png")
    print("  results/step_02_ev_schedules.png")
    print("  results/step_02_total_demand.png")
    
    print()
    print("=== False-price attack ===")
    print(
        f"Legitimate price at {attacked_hour:02d}:00: "
        f"€{legitimate_price_eur_per_kwh[attacked_hour]:.2f}/kWh"
    )
    print(
        f"Attacked price at {attacked_hour:02d}:00: "
        f"€{attacked_price_eur_per_kwh[attacked_hour]:.2f}/kWh"
    )
    print()

    print("EV charging cost")
    print(
        f"  Legitimate price-aware schedule: "
        f"€{legitimate_ev_cost_eur:.2f}"
    )
    print(
        f"  Cost perceived under attack: "
        f"€{attacked_ev_perceived_cost_eur:.2f}"
    )
    print(
        f"  Real cost under legitimate settlement: "
        f"€{attacked_ev_real_cost_eur:.2f}"
    )
    print()

    print("Peak total demand")
    print(
        f"  Legitimate price-aware schedule: "
        f"{np.max(total_legitimate_price_aware_demand_kw):.2f} kW"
    )
    print(
        f"  Schedule produced under attack: "
        f"{np.max(total_attacked_demand_kw):.2f} kW"
    )
    print()
    print("Charging schedule comparison")
    print("Hour | Legitimate price | Attacked price | Normal EV | Attacked EV")
    print("-" * 75)

    for hour in hours:
        print(
            f"{hour:02d}:00 | "
            f"{legitimate_price_eur_per_kwh[hour]:16.2f} | "
            f"{attacked_price_eur_per_kwh[hour]:14.2f} | "
            f"{legitimate_price_aware_ev_charging_kw[hour]:9.2f} | "
            f"{attacked_ev_charging_kw[hour]:11.2f}"
        )
    
    plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        total_legitimate_price_aware_demand_kw,
        marker="o",
        label="Legitimate price-aware demand",
    )

    plt.plot(
        hours,
        total_attacked_demand_kw,
        marker="o",
        label="Demand under false-price attack",
    )

    plt.xlabel("Hour")
    plt.ylabel("Total demand (kW)")
    plt.title("Effect of a False-Price Signal on EV Charging")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory / "step_03_false_price_attack.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == "__main__":
    main()