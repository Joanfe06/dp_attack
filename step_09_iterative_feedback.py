from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from step_02_ev_charging import (
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
    number_of_iterations = 10

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
    ev_adoption_rate = 0.30

    number_of_evs = int(
        round(number_of_households * ev_adoption_rate)
    )

    aggregate_fixed_demand_kw = (
        fixed_demand_per_household_kw
        * number_of_households
    )

    aggregate_fixed_demand_mw = kw_to_mw(
        aggregate_fixed_demand_kw
    )

    # ============================================================
    # 2. Initial dynamic price
    # ============================================================

    # Generation cost:
    #
    # C(D) = aD² + bD
    #
    # Marginal price:
    #
    # p(D) = 2aD + b
    quadratic_coefficient = 6.190476
    linear_coefficient = 28.571432

    initial_price_eur_per_mwh = (
        calculate_marginal_price_eur_per_mwh(
            demand_mw=aggregate_fixed_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
        )
    )

    initial_price_eur_per_kwh = (
        eur_per_mwh_to_eur_per_kwh(
            initial_price_eur_per_mwh
        )
    )

    # ============================================================
    # 3. Generate the heterogeneous EV population once
    # ============================================================

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

    raw_required_energies_kwh = np.clip(
        rng.normal(
            loc=12.0,
            scale=4.0,
            size=number_of_evs,
        ),
        4.0,
        24.0,
    )

    # Precompute availability and feasible energy once.
    available_hours_by_ev: list[np.ndarray] = []
    required_energies_kwh = np.zeros(
        number_of_evs,
        dtype=float,
    )

    for ev_index in range(number_of_evs):
        available_hours = get_available_hours(
            arrival_hour=int(arrival_hours[ev_index]),
            departure_hour=int(departure_hours[ev_index]),
        )

        maximum_possible_energy_kwh = (
            len(available_hours)
            * float(maximum_charging_powers_kw[ev_index])
            * interval_duration_hours
        )

        available_hours_by_ev.append(available_hours)

        required_energies_kwh[ev_index] = min(
            float(raw_required_energies_kwh[ev_index]),
            maximum_possible_energy_kwh,
        )

    expected_total_ev_energy_mwh = (
        float(np.sum(required_energies_kwh))
        / 1000.0
    )

    # ============================================================
    # 4. Iterative price-demand feedback
    # ============================================================

    current_price_eur_per_kwh = (
        initial_price_eur_per_kwh.copy()
    )

    previous_total_demand_mw: np.ndarray | None = None

    price_used_history: list[np.ndarray] = []
    updated_price_history: list[np.ndarray] = []
    ev_demand_history_mw: list[np.ndarray] = []
    total_demand_history_mw: list[np.ndarray] = []

    max_price_changes: list[float] = []
    max_demand_changes: list[float] = []
    peak_demands_mw: list[float] = []
    peak_hours: list[int] = []
    daily_generation_costs_eur: list[float] = []
    two_cycle_flags: list[bool] = []

    print("=== Naive iterative dynamic-pricing feedback ===")
    print()
    print(f"Iterations: {number_of_iterations}")
    print(f"Random seed: {random_seed}")
    print(f"Households: {number_of_households:,}")
    print(f"EV adoption rate: {ev_adoption_rate:.0%}")
    print(f"EVs: {number_of_evs:,}")
    print(
        "Expected EV energy in every iteration: "
        f"{expected_total_ev_energy_mwh:.4f} MWh"
    )

    print()
    print(
        "Iter | Cheapest price | EV peak | Total peak | "
        "Max Δprice | Max Δdemand | Two-cycle"
    )
    print("-" * 103)

    for iteration_index in range(number_of_iterations):
        aggregate_ev_demand_kw = np.zeros(
            24,
            dtype=float,
        )

        # Every EV independently reschedules according to the
        # current price profile.
        for ev_index in range(number_of_evs):
            schedule_kw = create_ev_schedule(
                price_eur_per_kwh=current_price_eur_per_kwh,
                available_hours=available_hours_by_ev[ev_index],
                required_energy_kwh=float(
                    required_energies_kwh[ev_index]
                ),
                maximum_charging_power_kw=float(
                    maximum_charging_powers_kw[ev_index]
                ),
                interval_duration_hours=interval_duration_hours,
                optimize_for_price=True,
            )

            aggregate_ev_demand_kw += schedule_kw

        aggregate_ev_demand_mw = kw_to_mw(
            aggregate_ev_demand_kw
        )

        total_demand_mw = (
            aggregate_fixed_demand_mw
            + aggregate_ev_demand_mw
        )

        # The operator calculates the next price from the demand
        # produced by the current price.
        updated_price_eur_per_mwh = (
            calculate_marginal_price_eur_per_mwh(
                demand_mw=total_demand_mw,
                quadratic_coefficient=quadratic_coefficient,
                linear_coefficient=linear_coefficient,
            )
        )

        updated_price_eur_per_kwh = (
            eur_per_mwh_to_eur_per_kwh(
                updated_price_eur_per_mwh
            )
        )

        max_price_change = float(
            np.max(
                np.abs(
                    updated_price_eur_per_kwh
                    - current_price_eur_per_kwh
                )
            )
        )

        if previous_total_demand_mw is None:
            max_demand_change = float("nan")
        else:
            max_demand_change = float(
                np.max(
                    np.abs(
                        total_demand_mw
                        - previous_total_demand_mw
                    )
                )
            )

        ev_energy_mwh = float(
            np.sum(
                aggregate_ev_demand_mw
                * interval_duration_hours
            )
        )

        if not np.isclose(
            ev_energy_mwh,
            expected_total_ev_energy_mwh,
            atol=1e-8,
        ):
            raise RuntimeError(
                "EV energy changed during the iterative simulation."
            )

        generation_cost_eur = float(
            np.sum(
                calculate_generation_cost_eur_per_hour(
                    demand_mw=total_demand_mw,
                    quadratic_coefficient=quadratic_coefficient,
                    linear_coefficient=linear_coefficient,
                )
                * interval_duration_hours
            )
        )

        price_used_history.append(
            current_price_eur_per_kwh.copy()
        )

        updated_price_history.append(
            updated_price_eur_per_kwh.copy()
        )

        ev_demand_history_mw.append(
            aggregate_ev_demand_mw.copy()
        )

        total_demand_history_mw.append(
            total_demand_mw.copy()
        )

        max_price_changes.append(max_price_change)
        max_demand_changes.append(max_demand_change)

        peak_demands_mw.append(
            float(np.max(total_demand_mw))
        )

        peak_hours.append(
            int(np.argmax(total_demand_mw))
        )

        daily_generation_costs_eur.append(
            generation_cost_eur
        )

        # Compare this demand with the demand two iterations earlier.
        # Equality indicates a period-two oscillation:
        #
        # A -> B -> A -> B
        two_cycle_detected = False

        if iteration_index >= 2:
            two_cycle_detected = np.allclose(
                total_demand_mw,
                total_demand_history_mw[
                    iteration_index - 2
                ],
                atol=1e-8,
            )

        two_cycle_flags.append(two_cycle_detected)

        cheapest_hour = int(
            np.argmin(current_price_eur_per_kwh)
        )

        ev_peak_hour = int(
            np.argmax(aggregate_ev_demand_mw)
        )

        total_peak_hour = int(
            np.argmax(total_demand_mw)
        )

        if np.isnan(max_demand_change):
            demand_change_text = "n/a"
        else:
            demand_change_text = (
                f"{max_demand_change:.2f} MW"
            )

        print(
            f"{iteration_index + 1:4d} | "
            f"{cheapest_hour:02d}:00 "
            f"({current_price_eur_per_kwh[cheapest_hour]:.3f}) | "
            f"{ev_peak_hour:02d}:00 "
            f"({aggregate_ev_demand_mw[ev_peak_hour]:5.2f} MW) | "
            f"{total_peak_hour:02d}:00 "
            f"({total_demand_mw[total_peak_hour]:5.2f} MW) | "
            f"{max_price_change:10.3f} | "
            f"{demand_change_text:>13s} | "
            f"{str(two_cycle_detected):9s}"
        )

        # Undamped update:
        #
        # p^(k+1) = p_new
        previous_total_demand_mw = total_demand_mw.copy()
        current_price_eur_per_kwh = (
            updated_price_eur_per_kwh.copy()
        )

    # ============================================================
    # 5. Final summary
    # ============================================================

    final_demand_change = max_demand_changes[-1]

    print()
    print("Final status")
    print(
        "  Final maximum price change: "
        f"€{max_price_changes[-1]:.4f}/kWh"
    )

    if np.isnan(final_demand_change):
        print("  Final maximum demand change: n/a")
    else:
        print(
            "  Final maximum demand change: "
            f"{final_demand_change:.4f} MW"
        )

    print(
        "  A two-cycle was detected: "
        f"{any(two_cycle_flags)}"
    )

    print(
        "  Final peak demand: "
        f"{peak_demands_mw[-1]:.2f} MW "
        f"at {peak_hours[-1]:02d}:00"
    )

    print(
        "  Final daily generation cost: "
        f"€{daily_generation_costs_eur[-1]:.2f}"
    )

    print()
    print("Hourly profile in the final iteration")
    print(
        "Hour | Price used | EV demand | Total demand | Updated price"
    )
    print("-" * 72)

    final_iteration_index = number_of_iterations - 1

    for hour in hours:
        print(
            f"{hour:02d}:00 | "
            f"{price_used_history[final_iteration_index][hour]:10.3f} | "
            f"{ev_demand_history_mw[final_iteration_index][hour]:9.2f} | "
            f"{total_demand_history_mw[final_iteration_index][hour]:12.2f} | "
            f"{updated_price_history[final_iteration_index][hour]:13.3f}"
        )

    # ============================================================
    # 6. Figures
    # ============================================================

    output_directory = Path("results")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_iterations = sorted(
        {
            0,
            1,
            2,
            number_of_iterations - 1,
        }
    )

    # Price profiles.
    plt.figure(figsize=(10, 5))

    plt.step(
        hours,
        initial_price_eur_per_kwh,
        where="mid",
        label="Initial price",
    )

    for index in selected_iterations:
        plt.step(
            hours,
            updated_price_history[index],
            where="mid",
            label=(
                "Updated price after iteration "
                f"{index + 1}"
            ),
        )

    plt.xlabel("Hour")
    plt.ylabel("Price (€/kWh)")
    plt.title("Undamped Price Iterations")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_09_price_iterations.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Demand profiles.
    plt.figure(figsize=(10, 5))

    for index in selected_iterations:
        plt.plot(
            hours,
            total_demand_history_mw[index],
            marker="o",
            label=f"Iteration {index + 1}",
        )

    plt.xlabel("Hour")
    plt.ylabel("Total demand (MW)")
    plt.title("Undamped Demand Iterations")
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_09_demand_iterations.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    iteration_numbers = np.arange(
        1,
        number_of_iterations + 1,
    )

    # Maximum price change per iteration.
    plt.figure(figsize=(10, 4))

    plt.plot(
        iteration_numbers,
        max_price_changes,
        marker="o",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Maximum price change (€/kWh)")
    plt.title("Price-Update Magnitude")
    plt.xticks(iteration_numbers)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_09_price_change.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Maximum demand change per iteration.
    demand_changes_for_plot = np.array(
        max_demand_changes,
        dtype=float,
    )

    demand_changes_for_plot[0] = np.nan

    plt.figure(figsize=(10, 4))

    plt.plot(
        iteration_numbers,
        demand_changes_for_plot,
        marker="o",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Maximum demand change (MW)")
    plt.title("Demand-Update Magnitude")
    plt.xticks(iteration_numbers)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_09_demand_change.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Peak demand by iteration.
    plt.figure(figsize=(10, 4))

    plt.plot(
        iteration_numbers,
        peak_demands_mw,
        marker="o",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Peak demand (MW)")
    plt.title("Peak Demand Across Iterations")
    plt.xticks(iteration_numbers)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_09_peak_demand.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Figures saved:")
    print("  results/step_09_price_iterations.png")
    print("  results/step_09_demand_iterations.png")
    print("  results/step_09_price_change.png")
    print("  results/step_09_demand_change.png")
    print("  results/step_09_peak_demand.png")


if __name__ == "__main__":
    main()