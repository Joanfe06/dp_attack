from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from step_02_ev_charging import (
    calculate_energy_kwh,
    create_ev_schedule,
    get_available_hours,
)


def kw_to_mw(power_kw: np.ndarray | float) -> np.ndarray | float:
    """Convert power from kW to MW."""
    return power_kw / 1000.0


def main() -> None:
    hours = np.arange(24)
    interval_duration_hours = 1.0

    # Fixed demand of one representative household.
    fixed_demand_per_household_kw = np.array(
        [
            1.2, 1.1, 1.0, 0.9, 0.9, 1.0,
            1.3, 1.8, 2.2, 2.0, 1.7, 1.6,
            1.5, 1.6, 1.7, 1.9, 2.3, 2.8,
            3.0, 2.7, 2.3, 1.9, 1.6, 1.4,
        ],
        dtype=float,
    )

    legitimate_price_eur_per_kwh = np.array(
        [
            0.14, 0.14, 0.14, 0.14, 0.14, 0.16,
            0.18, 0.22, 0.26, 0.24, 0.20, 0.18,
            0.18, 0.18, 0.20, 0.22, 0.28, 0.36,
            0.40, 0.34, 0.28, 0.22, 0.18, 0.16,
        ],
        dtype=float,
    )

    # The attacker modifies only the price received by some EVs.
    attacked_price_eur_per_kwh = (
        legitimate_price_eur_per_kwh.copy()
    )

    attacked_hour = 18
    attacked_price_eur_per_kwh[attacked_hour] = 0.05

    # Population parameters.
    number_of_households = 10_000
    ev_adoption_rate = 0.30
    attacked_ev_fraction = 0.25

    number_of_evs = int(
        round(number_of_households * ev_adoption_rate)
    )

    number_of_attacked_evs = int(
        round(number_of_evs * attacked_ev_fraction)
    )

    number_of_unattacked_evs = (
        number_of_evs - number_of_attacked_evs
    )

    # All EVs still use the same technical parameters for now.
    arrival_hour = 18
    departure_hour = 7
    required_energy_kwh = 12.0
    maximum_charging_power_kw = 3.6

    available_hours = get_available_hours(
        arrival_hour=arrival_hour,
        departure_hour=departure_hour,
    )

    # Schedule followed by an EV receiving legitimate prices.
    legitimate_ev_schedule_kw = create_ev_schedule(
        price_eur_per_kwh=legitimate_price_eur_per_kwh,
        available_hours=available_hours,
        required_energy_kwh=required_energy_kwh,
        maximum_charging_power_kw=maximum_charging_power_kw,
        interval_duration_hours=interval_duration_hours,
        optimize_for_price=True,
    )

    # Schedule followed by an EV receiving the false price.
    attacked_ev_schedule_kw = create_ev_schedule(
        price_eur_per_kwh=attacked_price_eur_per_kwh,
        available_hours=available_hours,
        required_energy_kwh=required_energy_kwh,
        maximum_charging_power_kw=maximum_charging_power_kw,
        interval_duration_hours=interval_duration_hours,
        optimize_for_price=True,
    )

    # Aggregate fixed demand from every household.
    aggregate_fixed_demand_kw = (
        fixed_demand_per_household_kw
        * number_of_households
    )

    # No-attack scenario:
    # every EV receives the legitimate price.
    aggregate_legitimate_ev_demand_kw = (
        legitimate_ev_schedule_kw
        * number_of_evs
    )

    # Attack scenario:
    # unattacked EVs follow the legitimate schedule;
    # attacked EVs follow the manipulated schedule.
    aggregate_attack_ev_demand_kw = (
        legitimate_ev_schedule_kw
        * number_of_unattacked_evs
        + attacked_ev_schedule_kw
        * number_of_attacked_evs
    )

    total_legitimate_demand_kw = (
        aggregate_fixed_demand_kw
        + aggregate_legitimate_ev_demand_kw
    )

    total_attack_demand_kw = (
        aggregate_fixed_demand_kw
        + aggregate_attack_ev_demand_kw
    )

    # Energy must remain equal in both scenarios.
    legitimate_ev_energy_mwh = (
        calculate_energy_kwh(
            aggregate_legitimate_ev_demand_kw,
            interval_duration_hours,
        )
        / 1000.0
    )

    attack_ev_energy_mwh = (
        calculate_energy_kwh(
            aggregate_attack_ev_demand_kw,
            interval_duration_hours,
        )
        / 1000.0
    )

    legitimate_peak_mw = float(
        np.max(kw_to_mw(total_legitimate_demand_kw))
    )

    attack_peak_mw = float(
        np.max(kw_to_mw(total_attack_demand_kw))
    )

    legitimate_demand_at_attacked_hour_mw = float(
        kw_to_mw(
            total_legitimate_demand_kw[attacked_hour]
        )
    )

    attack_demand_at_attacked_hour_mw = float(
        kw_to_mw(
            total_attack_demand_kw[attacked_hour]
        )
    )

    attack_increase_mw = (
        attack_demand_at_attacked_hour_mw
        - legitimate_demand_at_attacked_hour_mw
    )

    attack_increase_percent = (
        100.0
        * attack_increase_mw
        / legitimate_demand_at_attacked_hour_mw
    )

    print("=== Population-level false-price attack ===")
    print()
    print("Population")
    print(f"  Households: {number_of_households:,}")
    print(f"  EV adoption rate: {ev_adoption_rate:.0%}")
    print(f"  Total EVs: {number_of_evs:,}")
    print(
        f"  Attacked EV fraction: "
        f"{attacked_ev_fraction:.0%}"
    )
    print(
        f"  EVs receiving false prices: "
        f"{number_of_attacked_evs:,}"
    )
    print(
        f"  EVs receiving legitimate prices: "
        f"{number_of_unattacked_evs:,}"
    )

    print()
    print("EV energy")
    print(
        f"  No-attack scenario: "
        f"{legitimate_ev_energy_mwh:.2f} MWh"
    )
    print(
        f"  Attack scenario: "
        f"{attack_ev_energy_mwh:.2f} MWh"
    )

    print()
    print("Peak aggregate demand")
    print(
        f"  No-attack scenario: "
        f"{legitimate_peak_mw:.2f} MW"
    )
    print(
        f"  Attack scenario: "
        f"{attack_peak_mw:.2f} MW"
    )

    print()
    print(f"Effect at {attacked_hour:02d}:00")
    print(
        f"  Legitimate demand: "
        f"{legitimate_demand_at_attacked_hour_mw:.2f} MW"
    )
    print(
        f"  Demand under attack: "
        f"{attack_demand_at_attacked_hour_mw:.2f} MW"
    )
    print(
        f"  Absolute increase: "
        f"{attack_increase_mw:.2f} MW"
    )
    print(
        f"  Relative increase: "
        f"{attack_increase_percent:.2f}%"
    )

    print()
    print("Hourly aggregate demand")
    print(
        "Hour | Fixed demand | No attack | Under attack | Difference"
    )
    print("-" * 68)

    for hour in hours:
        fixed_mw = kw_to_mw(
            aggregate_fixed_demand_kw[hour]
        )

        legitimate_mw = kw_to_mw(
            total_legitimate_demand_kw[hour]
        )

        attacked_mw = kw_to_mw(
            total_attack_demand_kw[hour]
        )

        difference_mw = attacked_mw - legitimate_mw

        print(
            f"{hour:02d}:00 | "
            f"{fixed_mw:12.2f} | "
            f"{legitimate_mw:9.2f} | "
            f"{attacked_mw:12.2f} | "
            f"{difference_mw:10.2f}"
        )

    output_directory = Path("results")
    output_directory.mkdir(parents=True, exist_ok=True)

    # First figure: complete demand profiles.
    plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        kw_to_mw(aggregate_fixed_demand_kw),
        marker="o",
        label="Fixed demand only",
    )

    plt.plot(
        hours,
        kw_to_mw(total_legitimate_demand_kw),
        marker="o",
        label="All EVs receive legitimate prices",
    )

    plt.plot(
        hours,
        kw_to_mw(total_attack_demand_kw),
        marker="o",
        label="25% of EVs receive false prices",
    )

    plt.xlabel("Hour")
    plt.ylabel("Aggregate demand (MW)")
    plt.title("Population-Level False-Price Attack")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory / "step_04_population_demand.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Second figure: attack-induced difference.
    attack_difference_mw = kw_to_mw(
        total_attack_demand_kw
        - total_legitimate_demand_kw
    )

    plt.figure(figsize=(10, 4))

    plt.bar(
        hours,
        attack_difference_mw,
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.xlabel("Hour")
    plt.ylabel("Attack-induced demand change (MW)")
    plt.title("Demand Shift Caused by the False-Price Signal")
    plt.xticks(hours)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_directory / "step_04_attack_difference.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Figures saved:")
    print("  results/step_04_population_demand.png")
    print("  results/step_04_attack_difference.png")


if __name__ == "__main__":
    main()