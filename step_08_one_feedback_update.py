from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend because the environment cannot open windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from step_02_ev_charging import (
    calculate_energy_kwh,
    create_ev_schedule,
    get_available_hours,
)

from step_06_generate_dynamic_price import (
    calculate_generation_cost_eur_per_hour,
    calculate_marginal_price_eur_per_mwh,
    eur_per_mwh_to_eur_per_kwh,
)


def kw_to_mw(
    power_kw: np.ndarray | float,
) -> np.ndarray | float:
    """Convert power from kW to MW."""
    return power_kw / 1000.0


def main() -> None:
    hours = np.arange(24)
    interval_duration_hours = 1.0

    # Using a fixed seed makes the randomly generated EV population
    # identical every time the program runs.
    random_seed = 42
    rng = np.random.default_rng(random_seed)

    # ============================================================
    # 1. Fixed household demand
    # ============================================================

    fixed_demand_per_household_kw = np.array(
        [
            1.2, 1.1, 1.0, 0.9, 0.9, 1.0,
            1.3, 1.8, 2.2, 2.0, 1.7, 1.6,
            1.5, 1.6, 1.7, 1.9, 2.3, 2.8,
            3.0, 2.7, 2.3, 1.9, 1.6, 1.4,
        ],
        dtype=float,
    )

    number_of_households = 10_000

    aggregate_fixed_demand_kw = (
        fixed_demand_per_household_kw
        * number_of_households
    )

    aggregate_fixed_demand_mw = kw_to_mw(
        aggregate_fixed_demand_kw
    )

    # ============================================================
    # 2. Generate the initial dynamic price
    # ============================================================

    # Generation-cost model:
    #
    # C(D) = aD² + bD
    #
    # Marginal price:
    #
    # p(D) = 2aD + b
    #
    # These parameters are illustrative rather than calibrated
    # using data from a real electricity market.
    quadratic_coefficient = 6.190476
    linear_coefficient = 28.571432

    legitimate_price_eur_per_mwh = (
        calculate_marginal_price_eur_per_mwh(
            demand_mw=aggregate_fixed_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
        )
    )

    legitimate_price_eur_per_kwh = (
        eur_per_mwh_to_eur_per_kwh(
            legitimate_price_eur_per_mwh
        )
    )

    # ============================================================
    # 3. Create the false-price signal
    # ============================================================

    attacked_price_eur_per_kwh = (
        legitimate_price_eur_per_kwh.copy()
    )

    attacked_hour = 18

    # The real price at 18:00 is approximately €0.40/kWh.
    # Attacked EVs are falsely told that it is €0.05/kWh.
    attacked_price_eur_per_kwh[attacked_hour] = 0.05

    # ============================================================
    # 4. Generate a heterogeneous EV population
    # ============================================================

    ev_adoption_rate = 0.30
    attacked_ev_fraction = 0.25

    number_of_evs = int(
        round(number_of_households * ev_adoption_rate)
    )

    # Arrival times are centered around 18:30.
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

    # Departure times are centered around 07:00.
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

    charging_power_options_kw = np.array(
        [3.6, 7.2, 11.0],
        dtype=float,
    )

    charging_power_probabilities = np.array(
        [0.60, 0.30, 0.10],
        dtype=float,
    )

    maximum_charging_powers_kw = rng.choice(
        charging_power_options_kw,
        size=number_of_evs,
        p=charging_power_probabilities,
    )

    # Required charging energy differs between EVs.
    raw_required_energies_kwh = np.clip(
        rng.normal(
            loc=12.0,
            scale=4.0,
            size=number_of_evs,
        ),
        4.0,
        24.0,
    )

    # Each EV independently has a 25% probability of receiving
    # the false price.
    receives_false_price = (
        rng.random(number_of_evs)
        < attacked_ev_fraction
    )

    # ============================================================
    # 5. Calculate and aggregate EV charging schedules
    # ============================================================

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

        # Prevent the randomly generated required energy from
        # exceeding what the EV can physically receive before departure.
        required_energy_kwh = min(
            float(raw_required_energies_kwh[ev_index]),
            maximum_possible_energy_kwh,
        )

        # Normal schedule: the EV receives the legitimate price.
        normal_schedule_kw = create_ev_schedule(
            price_eur_per_kwh=legitimate_price_eur_per_kwh,
            available_hours=available_hours,
            required_energy_kwh=required_energy_kwh,
            maximum_charging_power_kw=maximum_charging_power_kw,
            interval_duration_hours=interval_duration_hours,
            optimize_for_price=True,
        )

        # Attack schedule: only selected EVs receive the false price.
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

        # Save the first ten EVs so their properties can be inspected.
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

    # ============================================================
    # 6. Calculate realized total demand
    # ============================================================

    total_normal_demand_kw = (
        aggregate_fixed_demand_kw
        + aggregate_normal_ev_demand_kw
    )

    total_attacked_demand_kw = (
        aggregate_fixed_demand_kw
        + aggregate_attacked_ev_demand_kw
    )

    realized_normal_demand_mw = kw_to_mw(
        total_normal_demand_kw
    )

    realized_attacked_demand_mw = kw_to_mw(
        total_attacked_demand_kw
    )

    # ============================================================
    # 7. Perform one price-feedback update
    # ============================================================

    # The initial price was based on fixed demand only.
    #
    # These updated prices are calculated after EV demand has
    # been added to the system.
    real_time_normal_price_eur_per_mwh = (
        calculate_marginal_price_eur_per_mwh(
            demand_mw=realized_normal_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
        )
    )

    real_time_attacked_price_eur_per_mwh = (
        calculate_marginal_price_eur_per_mwh(
            demand_mw=realized_attacked_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
        )
    )

    real_time_normal_price_eur_per_kwh = (
        eur_per_mwh_to_eur_per_kwh(
            real_time_normal_price_eur_per_mwh
        )
    )

    real_time_attacked_price_eur_per_kwh = (
        eur_per_mwh_to_eur_per_kwh(
            real_time_attacked_price_eur_per_mwh
        )
    )

    # ============================================================
    # 8. Calculate generation costs
    # ============================================================

    normal_generation_cost_eur_per_hour = (
        calculate_generation_cost_eur_per_hour(
            demand_mw=realized_normal_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
        )
    )

    attacked_generation_cost_eur_per_hour = (
        calculate_generation_cost_eur_per_hour(
            demand_mw=realized_attacked_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
        )
    )

    normal_daily_generation_cost_eur = float(
        np.sum(
            normal_generation_cost_eur_per_hour
            * interval_duration_hours
        )
    )

    attacked_daily_generation_cost_eur = float(
        np.sum(
            attacked_generation_cost_eur_per_hour
            * interval_duration_hours
        )
    )

    attack_cost_increase_eur = (
        attacked_daily_generation_cost_eur
        - normal_daily_generation_cost_eur
    )

    attack_cost_increase_percent = (
        100.0
        * attack_cost_increase_eur
        / normal_daily_generation_cost_eur
    )

    # ============================================================
    # 9. Validate energy conservation
    # ============================================================

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

    energy_difference_mwh = (
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
            "The attack changed the total EV energy. "
            "The energy should only be shifted between hours."
        )

    # ============================================================
    # 10. Calculate peak-demand metrics
    # ============================================================

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

    relative_increase_at_attack_hour = (
        100.0
        * increase_at_attack_hour_mw
        / normal_demand_at_attack_hour_mw
    )

    # ============================================================
    # 11. Print the simulation results
    # ============================================================

    print("=== One dynamic-price feedback update ===")
    print()

    print("Initial generated price")
    print(
        f"  Minimum price: "
        f"€{np.min(legitimate_price_eur_per_kwh):.3f}/kWh"
    )
    print(
        f"  Maximum price: "
        f"€{np.max(legitimate_price_eur_per_kwh):.3f}/kWh"
    )
    print(
        f"  Legitimate price at {attacked_hour:02d}:00: "
        f"€{legitimate_price_eur_per_kwh[attacked_hour]:.3f}/kWh"
    )
    print(
        f"  False price shown to attacked EVs: "
        f"€{attacked_price_eur_per_kwh[attacked_hour]:.3f}/kWh"
    )

    print()
    print("Population")
    print(f"  Random seed: {random_seed}")
    print(f"  Households: {number_of_households:,}")
    print(f"  EV adoption rate: {ev_adoption_rate:.0%}")
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
        f"{energy_difference_mwh:.10f} MWh"
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
        f"  Absolute increase: "
        f"{increase_at_attack_hour_mw:.2f} MW"
    )
    print(
        f"  Relative increase: "
        f"{relative_increase_at_attack_hour:.2f}%"
    )

    print()
    print("Updated price at the attacked hour")
    print(
        f"  Initial forecast price: "
        f"€{legitimate_price_eur_per_kwh[attacked_hour]:.3f}/kWh"
    )
    print(
        f"  Updated normal price: "
        f"€{real_time_normal_price_eur_per_kwh[attacked_hour]:.3f}/kWh"
    )
    print(
        f"  Updated price under attack: "
        f"€{real_time_attacked_price_eur_per_kwh[attacked_hour]:.3f}/kWh"
    )

    print()
    print("Daily generation cost")
    print(
        f"  Normal EV scheduling: "
        f"€{normal_daily_generation_cost_eur:.2f}"
    )
    print(
        f"  EV scheduling under attack: "
        f"€{attacked_daily_generation_cost_eur:.2f}"
    )
    print(
        f"  Increase caused by attack: "
        f"€{attack_cost_increase_eur:.2f}"
    )
    print(
        f"  Relative cost increase: "
        f"{attack_cost_increase_percent:.4f}%"
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
    print("Hourly demand and updated prices")
    print(
        "Hour | Normal demand | Attack demand | "
        "Initial price | Updated normal | Updated attack"
    )
    print("-" * 98)

    for hour in hours:
        print(
            f"{hour:02d}:00 | "
            f"{realized_normal_demand_mw[hour]:13.2f} | "
            f"{realized_attacked_demand_mw[hour]:13.2f} | "
            f"{legitimate_price_eur_per_kwh[hour]:13.3f} | "
            f"{real_time_normal_price_eur_per_kwh[hour]:14.3f} | "
            f"{real_time_attacked_price_eur_per_kwh[hour]:14.3f}"
        )

    # ============================================================
    # 12. Create figures
    # ============================================================

    output_directory = Path("results")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Figure 1: legitimate and false price signals.
    plt.figure(figsize=(10, 5))

    plt.step(
        hours,
        legitimate_price_eur_per_kwh,
        where="mid",
        label="Legitimate initial price",
    )

    plt.step(
        hours,
        attacked_price_eur_per_kwh,
        where="mid",
        label="Price seen by attacked EVs",
    )

    plt.xlabel("Hour")
    plt.ylabel("Price (€/kWh)")
    plt.title("Legitimate and False Price Signals")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_08_price_signals.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Figure 2: aggregate EV demand.
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
    plt.title("Aggregate EV Demand")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_08_ev_demand.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Figure 3: total demand.
    plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        realized_normal_demand_mw,
        marker="o",
        label="Normal total demand",
    )

    plt.plot(
        hours,
        realized_attacked_demand_mw,
        marker="o",
        label="Total demand under attack",
    )

    plt.xlabel("Hour")
    plt.ylabel("Aggregate demand (MW)")
    plt.title("Realized Aggregate Demand")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_08_total_demand.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Figure 4: initial and updated prices.
    plt.figure(figsize=(10, 5))

    plt.step(
        hours,
        legitimate_price_eur_per_kwh,
        where="mid",
        label="Initial price from fixed-demand forecast",
    )

    plt.step(
        hours,
        real_time_normal_price_eur_per_kwh,
        where="mid",
        label="Updated price after normal EV response",
    )

    plt.step(
        hours,
        real_time_attacked_price_eur_per_kwh,
        where="mid",
        label="Updated price after attacked EV response",
    )

    plt.xlabel("Hour")
    plt.ylabel("Marginal price (€/kWh)")
    plt.title("One Dynamic-Price Feedback Update")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_08_feedback_prices.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Figure 5: attack-induced demand change.
    attack_difference_mw = (
        realized_attacked_demand_mw
        - realized_normal_demand_mw
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
        / "step_08_attack_difference.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Figures saved:")
    print("  results/step_08_price_signals.png")
    print("  results/step_08_ev_demand.png")
    print("  results/step_08_total_demand.png")
    print("  results/step_08_feedback_prices.png")
    print("  results/step_08_attack_difference.png")


if __name__ == "__main__":
    main()