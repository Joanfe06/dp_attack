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


def kw_to_mw(
    power_kw: np.ndarray | float,
) -> np.ndarray | float:
    """Convert kW to MW."""
    return power_kw / 1000.0


def main() -> None:
    hours = np.arange(24)
    interval_duration_hours = 1.0

    # A fixed seed makes the randomly generated population reproducible.
    random_seed = 42
    rng = np.random.default_rng(random_seed)

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

    # ------------------------------------------------------------------
    # Generate heterogeneous EV properties
    # ------------------------------------------------------------------

    # Most vehicles arrive around 18:30.
    # Values are rounded to complete hours and limited to 15:00–23:00.
    arrival_hours = np.clip(
        np.rint(
            rng.normal(
                loc=18.5,
                scale=1.5,
                size=number_of_evs,
            )
        ),
        15,
        23,
    ).astype(int)

    # Most vehicles depart around 07:00.
    # Departure times are limited to 05:00–10:00.
    departure_hours = np.clip(
        np.rint(
            rng.normal(
                loc=7.0,
                scale=1.0,
                size=number_of_evs,
            )
        ),
        5,
        10,
    ).astype(int)

    # Different charger powers.
    charging_power_options_kw = np.array(
        [3.6, 7.2, 11.0]
    )

    charging_power_probabilities = np.array(
        [0.60, 0.30, 0.10]
    )

    maximum_charging_powers_kw = rng.choice(
        charging_power_options_kw,
        size=number_of_evs,
        p=charging_power_probabilities,
    )

    # Required energy is centered around 12 kWh, but differs per EV.
    raw_required_energies_kwh = np.clip(
        rng.normal(
            loc=12.0,
            scale=4.0,
            size=number_of_evs,
        ),
        4.0,
        24.0,
    )

    # Randomly select the EVs that receive the false price.
    receives_false_price = (
        rng.random(number_of_evs)
        < attacked_ev_fraction
    )

    # ------------------------------------------------------------------
    # Aggregate schedules
    # ------------------------------------------------------------------

    aggregate_normal_ev_demand_kw = np.zeros(24)
    aggregate_attacked_ev_demand_kw = np.zeros(24)

    number_receiving_false_price = int(
        np.sum(receives_false_price)
    )

    number_attacked_and_connected = 0
    number_changed_schedules = 0

    preview_records: list[tuple] = []

    for ev_index in range(number_of_evs):
        arrival_hour = int(arrival_hours[ev_index])
        departure_hour = int(departure_hours[ev_index])

        maximum_charging_power_kw = float(
            maximum_charging_powers_kw[ev_index]
        )

        available_hours = get_available_hours(
            arrival_hour=arrival_hour,
            departure_hour=departure_hour,
        )

        maximum_possible_energy_kwh = (
            len(available_hours)
            * maximum_charging_power_kw
            * interval_duration_hours
        )

        # Guarantee that the requested energy is feasible.
        required_energy_kwh = min(
            float(raw_required_energies_kwh[ev_index]),
            maximum_possible_energy_kwh,
        )

        normal_schedule_kw = create_ev_schedule(
            price_eur_per_kwh=legitimate_price_eur_per_kwh,
            available_hours=available_hours,
            required_energy_kwh=required_energy_kwh,
            maximum_charging_power_kw=maximum_charging_power_kw,
            interval_duration_hours=interval_duration_hours,
            optimize_for_price=True,
        )

        if receives_false_price[ev_index]:
            attacked_schedule_kw = create_ev_schedule(
                price_eur_per_kwh=attacked_price_eur_per_kwh,
                available_hours=available_hours,
                required_energy_kwh=required_energy_kwh,
                maximum_charging_power_kw=maximum_charging_power_kw,
                interval_duration_hours=interval_duration_hours,
                optimize_for_price=True,
            )
        else:
            attacked_schedule_kw = normal_schedule_kw.copy()

        is_connected_at_attacked_hour = (
            attacked_hour in available_hours
        )

        schedule_changed = not np.allclose(
            normal_schedule_kw,
            attacked_schedule_kw,
        )

        if (
            receives_false_price[ev_index]
            and is_connected_at_attacked_hour
        ):
            number_attacked_and_connected += 1

        if schedule_changed:
            number_changed_schedules += 1

        aggregate_normal_ev_demand_kw += normal_schedule_kw
        aggregate_attacked_ev_demand_kw += attacked_schedule_kw

        # Save the first ten EVs for inspection.
        if ev_index < 10:
            preview_records.append(
                (
                    ev_index,
                    arrival_hour,
                    departure_hour,
                    required_energy_kwh,
                    maximum_charging_power_kw,
                    bool(receives_false_price[ev_index]),
                    is_connected_at_attacked_hour,
                    schedule_changed,
                )
            )

    # ------------------------------------------------------------------
    # Add the fixed household demand
    # ------------------------------------------------------------------

    aggregate_fixed_demand_kw = (
        fixed_demand_per_household_kw
        * number_of_households
    )

    total_normal_demand_kw = (
        aggregate_fixed_demand_kw
        + aggregate_normal_ev_demand_kw
    )

    total_attacked_demand_kw = (
        aggregate_fixed_demand_kw
        + aggregate_attacked_ev_demand_kw
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    normal_ev_energy_mwh = (
        calculate_energy_kwh(
            aggregate_normal_ev_demand_kw,
            interval_duration_hours,
        )
        / 1000.0
    )

    attacked_ev_energy_mwh = (
        calculate_energy_kwh(
            aggregate_attacked_ev_demand_kw,
            interval_duration_hours,
        )
        / 1000.0
    )

    attack_energy_difference_mwh = (
        calculate_energy_kwh(
            aggregate_attacked_ev_demand_kw
            - aggregate_normal_ev_demand_kw,
            interval_duration_hours,
        )
        / 1000.0
    )

    if not np.isclose(
        normal_ev_energy_mwh,
        attacked_ev_energy_mwh,
        atol=1e-8,
    ):
        raise RuntimeError(
            "The attack changed total EV energy. "
            "Energy should only be shifted in time."
        )

    normal_peak_hour = int(
        np.argmax(total_normal_demand_kw)
    )

    attacked_peak_hour = int(
        np.argmax(total_attacked_demand_kw)
    )

    normal_peak_mw = float(
        kw_to_mw(
            total_normal_demand_kw[normal_peak_hour]
        )
    )

    attacked_peak_mw = float(
        kw_to_mw(
            total_attacked_demand_kw[attacked_peak_hour]
        )
    )

    normal_demand_at_attack_hour_mw = float(
        kw_to_mw(
            total_normal_demand_kw[attacked_hour]
        )
    )

    attacked_demand_at_attack_hour_mw = float(
        kw_to_mw(
            total_attacked_demand_kw[attacked_hour]
        )
    )

    increase_at_attack_hour_mw = (
        attacked_demand_at_attack_hour_mw
        - normal_demand_at_attack_hour_mw
    )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------

    print("=== Heterogeneous EV population ===")
    print()

    print("Population")
    print(f"  Random seed: {random_seed}")
    print(f"  Households: {number_of_households:,}")
    print(f"  EVs: {number_of_evs:,}")
    print(
        f"  EVs receiving false prices: "
        f"{number_receiving_false_price:,}"
    )
    print(
        f"  Attacked EVs connected at {attacked_hour:02d}:00: "
        f"{number_attacked_and_connected:,}"
    )
    print(
        f"  EV schedules actually changed: "
        f"{number_changed_schedules:,}"
    )

    print()
    print("EV energy")
    print(
        f"  Normal scenario: "
        f"{normal_ev_energy_mwh:.4f} MWh"
    )
    print(
        f"  Attack scenario: "
        f"{attacked_ev_energy_mwh:.4f} MWh"
    )
    print(
        f"  Net energy difference: "
        f"{attack_energy_difference_mwh:.10f} MWh"
    )

    print()
    print("Aggregate peak demand")
    print(
        f"  Normal scenario: "
        f"{normal_peak_mw:.2f} MW "
        f"at {normal_peak_hour:02d}:00"
    )
    print(
        f"  Attack scenario: "
        f"{attacked_peak_mw:.2f} MW "
        f"at {attacked_peak_hour:02d}:00"
    )

    print()
    print(f"Effect at {attacked_hour:02d}:00")
    print(
        f"  Normal demand: "
        f"{normal_demand_at_attack_hour_mw:.2f} MW"
    )
    print(
        f"  Demand under attack: "
        f"{attacked_demand_at_attack_hour_mw:.2f} MW"
    )
    print(
        f"  Increase caused by attack: "
        f"{increase_at_attack_hour_mw:.2f} MW"
    )

    print()
    print("First ten EVs")
    print(
        "ID | Arrival | Departure | Energy | Power | "
        "Attacked | Connected at 18 | Changed"
    )
    print("-" * 87)

    for record in preview_records:
        (
            ev_index,
            arrival_hour,
            departure_hour,
            required_energy_kwh,
            maximum_power_kw,
            attacked,
            connected,
            changed,
        ) = record

        print(
            f"{ev_index:02d} | "
            f"{arrival_hour:02d}:00   | "
            f"{departure_hour:02d}:00     | "
            f"{required_energy_kwh:6.2f} | "
            f"{maximum_power_kw:5.1f} | "
            f"{str(attacked):8s} | "
            f"{str(connected):15s} | "
            f"{str(changed):7s}"
        )

    print()
    print("Hourly aggregate demand")
    print(
        "Hour | Normal EV | Attacked EV | "
        "Normal total | Attacked total | Difference"
    )
    print("-" * 82)

    for hour in hours:
        normal_ev_mw = kw_to_mw(
            aggregate_normal_ev_demand_kw[hour]
        )

        attacked_ev_mw = kw_to_mw(
            aggregate_attacked_ev_demand_kw[hour]
        )

        normal_total_mw = kw_to_mw(
            total_normal_demand_kw[hour]
        )

        attacked_total_mw = kw_to_mw(
            total_attacked_demand_kw[hour]
        )

        difference_mw = (
            attacked_total_mw - normal_total_mw
        )

        print(
            f"{hour:02d}:00 | "
            f"{normal_ev_mw:9.2f} | "
            f"{attacked_ev_mw:11.2f} | "
            f"{normal_total_mw:12.2f} | "
            f"{attacked_total_mw:14.2f} | "
            f"{difference_mw:10.2f}"
        )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    output_directory = Path("results")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        kw_to_mw(aggregate_normal_ev_demand_kw),
        marker="o",
        label="Normal EV demand",
    )

    plt.plot(
        hours,
        kw_to_mw(aggregate_attacked_ev_demand_kw),
        marker="o",
        label="EV demand under attack",
    )

    plt.xlabel("Hour")
    plt.ylabel("EV demand (MW)")
    plt.title("Heterogeneous EV Population")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_05_heterogeneous_ev_demand.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        kw_to_mw(total_normal_demand_kw),
        marker="o",
        label="Normal total demand",
    )

    plt.plot(
        hours,
        kw_to_mw(total_attacked_demand_kw),
        marker="o",
        label="Total demand under attack",
    )

    plt.xlabel("Hour")
    plt.ylabel("Aggregate demand (MW)")
    plt.title("Aggregate Effect of the False-Price Attack")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_05_total_demand.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    attack_difference_mw = kw_to_mw(
        total_attacked_demand_kw
        - total_normal_demand_kw
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
    plt.ylabel("Demand change (MW)")
    plt.title("Demand Shift Caused by the False Price")
    plt.xticks(hours)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_05_attack_difference.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Figures saved:")
    print(
        "  results/"
        "step_05_heterogeneous_ev_demand.png"
    )
    print(
        "  results/"
        "step_05_total_demand.png"
    )
    print(
        "  results/"
        "step_05_attack_difference.png"
    )


if __name__ == "__main__":
    main()