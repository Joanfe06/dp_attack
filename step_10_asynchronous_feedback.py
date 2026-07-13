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

    # Feedback parameters.
    maximum_iterations = 100
    price_damping_factor = 0.20
    ev_rescheduling_fraction = 0.10

    # Convergence is declared only if both conditions remain satisfied
    # for several consecutive iterations.
    convergence_price_tolerance_eur_per_kwh = 1e-4
    convergence_demand_tolerance_mw = 0.01
    convergence_patience = 5

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
    # 4. Initial schedules
    # ============================================================

    # Each row stores the 24-hour charging schedule of one EV.
    current_ev_schedules_kw = np.zeros(
        (number_of_evs, 24),
        dtype=float,
    )

    for ev_index in range(number_of_evs):
        current_ev_schedules_kw[ev_index] = (
            create_ev_schedule(
                price_eur_per_kwh=initial_price_eur_per_kwh,
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
        )

    aggregate_ev_demand_kw = np.sum(
        current_ev_schedules_kw,
        axis=0,
    )

    previous_total_demand_mw = (
        aggregate_fixed_demand_mw
        + kw_to_mw(aggregate_ev_demand_kw)
    )

    current_price_eur_per_kwh = (
        initial_price_eur_per_kwh.copy()
    )

    # Reschedule exactly this many EVs per iteration.
    evs_selected_per_iteration = max(
        1,
        int(round(
            number_of_evs * ev_rescheduling_fraction
        )),
    )

    # ============================================================
    # 5. Histories and convergence state
    # ============================================================

    price_history: list[np.ndarray] = [
        current_price_eur_per_kwh.copy()
    ]
    total_demand_history_mw: list[np.ndarray] = [
        previous_total_demand_mw.copy()
    ]

    raw_price_change_history: list[float] = []
    applied_price_change_history: list[float] = []
    demand_change_history_mw: list[float] = []
    changed_schedules_history: list[int] = []
    peak_demand_history_mw: list[float] = []
    peak_hour_history: list[int] = []
    generation_cost_history_eur: list[float] = []

    consecutive_converged_iterations = 0
    converged = False
    completed_iterations = 0

    # ============================================================
    # 6. Partial asynchronous feedback
    # ============================================================

    print(
        "=== Damped feedback with partial EV rescheduling ==="
    )
    print()
    print(f"Maximum iterations: {maximum_iterations}")
    print(
        f"Price damping factor: "
        f"{price_damping_factor:.2f}"
    )
    print(
        f"EV rescheduling fraction: "
        f"{ev_rescheduling_fraction:.0%}"
    )
    print(
        f"EVs selected per iteration: "
        f"{evs_selected_per_iteration:,}"
    )
    print(f"Random seed: {random_seed}")
    print(f"Households: {number_of_households:,}")
    print(f"EVs: {number_of_evs:,}")
    print(
        "Expected EV energy in every iteration: "
        f"{expected_total_ev_energy_mwh:.4f} MWh"
    )

    print()
    print(
        "Iter | Changed EVs | Cheapest hour | Total peak | "
        "Raw Δprice | Applied Δprice | Max Δdemand"
    )
    print("-" * 112)

    for iteration_index in range(maximum_iterations):
        selected_ev_indices = rng.choice(
            number_of_evs,
            size=evs_selected_per_iteration,
            replace=False,
        )

        number_of_changed_schedules = 0

        # Only the selected EVs reconsider their schedules.
        for ev_index in selected_ev_indices:
            previous_schedule_kw = (
                current_ev_schedules_kw[ev_index].copy()
            )

            new_schedule_kw = create_ev_schedule(
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

            if not np.allclose(
                previous_schedule_kw,
                new_schedule_kw,
                atol=1e-12,
            ):
                number_of_changed_schedules += 1

                # Update the aggregate without summing all EVs again.
                aggregate_ev_demand_kw -= previous_schedule_kw
                aggregate_ev_demand_kw += new_schedule_kw

                current_ev_schedules_kw[ev_index] = (
                    new_schedule_kw
                )

        aggregate_ev_demand_mw = kw_to_mw(
            aggregate_ev_demand_kw
        )

        total_demand_mw = (
            aggregate_fixed_demand_mw
            + aggregate_ev_demand_mw
        )

        raw_price_eur_per_mwh = (
            calculate_marginal_price_eur_per_mwh(
                demand_mw=total_demand_mw,
                quadratic_coefficient=quadratic_coefficient,
                linear_coefficient=linear_coefficient,
            )
        )

        raw_price_eur_per_kwh = (
            eur_per_mwh_to_eur_per_kwh(
                raw_price_eur_per_mwh
            )
        )

        # Damped price update:
        #
        # p_next = (1 - alpha) p_current + alpha p_raw
        next_price_eur_per_kwh = (
            (1.0 - price_damping_factor)
            * current_price_eur_per_kwh
            + price_damping_factor
            * raw_price_eur_per_kwh
        )

        raw_price_change = float(
            np.max(
                np.abs(
                    raw_price_eur_per_kwh
                    - current_price_eur_per_kwh
                )
            )
        )

        applied_price_change = float(
            np.max(
                np.abs(
                    next_price_eur_per_kwh
                    - current_price_eur_per_kwh
                )
            )
        )

        maximum_demand_change_mw = float(
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

        daily_generation_cost_eur = float(
            np.sum(
                calculate_generation_cost_eur_per_hour(
                    demand_mw=total_demand_mw,
                    quadratic_coefficient=quadratic_coefficient,
                    linear_coefficient=linear_coefficient,
                )
                * interval_duration_hours
            )
        )

        cheapest_hour = int(
            np.argmin(current_price_eur_per_kwh)
        )

        peak_hour = int(
            np.argmax(total_demand_mw)
        )

        peak_demand_mw = float(
            total_demand_mw[peak_hour]
        )

        raw_price_change_history.append(
            raw_price_change
        )
        applied_price_change_history.append(
            applied_price_change
        )
        demand_change_history_mw.append(
            maximum_demand_change_mw
        )
        changed_schedules_history.append(
            number_of_changed_schedules
        )
        peak_demand_history_mw.append(
            peak_demand_mw
        )
        peak_hour_history.append(
            peak_hour
        )
        generation_cost_history_eur.append(
            daily_generation_cost_eur
        )

        price_history.append(
            next_price_eur_per_kwh.copy()
        )
        total_demand_history_mw.append(
            total_demand_mw.copy()
        )

        print(
            f"{iteration_index + 1:4d} | "
            f"{number_of_changed_schedules:11d} | "
            f"{cheapest_hour:02d}:00 "
            f"({current_price_eur_per_kwh[cheapest_hour]:.3f}) | "
            f"{peak_hour:02d}:00 "
            f"({peak_demand_mw:5.2f} MW) | "
            f"{raw_price_change:10.4f} | "
            f"{applied_price_change:14.4f} | "
            f"{maximum_demand_change_mw:12.4f}"
        )

        # Require both price and demand to remain below their
        # respective thresholds for several iterations.
        if (
            applied_price_change
            <= convergence_price_tolerance_eur_per_kwh
            and maximum_demand_change_mw
            <= convergence_demand_tolerance_mw
        ):
            consecutive_converged_iterations += 1
        else:
            consecutive_converged_iterations = 0

        completed_iterations = iteration_index + 1

        previous_total_demand_mw = (
            total_demand_mw.copy()
        )
        current_price_eur_per_kwh = (
            next_price_eur_per_kwh.copy()
        )

        if (
            consecutive_converged_iterations
            >= convergence_patience
        ):
            converged = True
            break

    # ============================================================
    # 7. Final summary
    # ============================================================

    final_total_demand_mw = (
        total_demand_history_mw[-1]
    )
    final_price_eur_per_kwh = price_history[-1]

    final_peak_hour = int(
        np.argmax(final_total_demand_mw)
    )

    final_peak_demand_mw = float(
        final_total_demand_mw[final_peak_hour]
    )

    print()
    print("Final status")
    print(
        f"  Completed iterations: "
        f"{completed_iterations}"
    )
    print(f"  Converged: {converged}")
    print(
        "  Final raw price discrepancy: "
        f"€{raw_price_change_history[-1]:.6f}/kWh"
    )
    print(
        "  Final applied price change: "
        f"€{applied_price_change_history[-1]:.6f}/kWh"
    )
    print(
        "  Final maximum demand change: "
        f"{demand_change_history_mw[-1]:.6f} MW"
    )
    print(
        "  EV schedules changed in final iteration: "
        f"{changed_schedules_history[-1]:,}"
    )
    print(
        "  Final peak demand: "
        f"{final_peak_demand_mw:.2f} MW "
        f"at {final_peak_hour:02d}:00"
    )
    print(
        "  Final daily generation cost: "
        f"€{generation_cost_history_eur[-1]:.2f}"
    )

    print()
    print("Final hourly profile")
    print(
        "Hour | Final price | EV demand | Total demand"
    )
    print("-" * 55)

    final_aggregate_ev_demand_mw = (
        final_total_demand_mw
        - aggregate_fixed_demand_mw
    )

    for hour in hours:
        print(
            f"{hour:02d}:00 | "
            f"{final_price_eur_per_kwh[hour]:11.3f} | "
            f"{final_aggregate_ev_demand_mw[hour]:9.2f} | "
            f"{final_total_demand_mw[hour]:12.2f}"
        )

    # ============================================================
    # 8. Figures
    # ============================================================

    output_directory = Path("results")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    iteration_numbers = np.arange(
        1,
        completed_iterations + 1,
    )

    # Price profiles at selected iterations.
    selected_price_indices = sorted(
        {
            0,
            min(1, completed_iterations),
            min(5, completed_iterations),
            min(20, completed_iterations),
            completed_iterations,
        }
    )

    plt.figure(figsize=(10, 5))

    for history_index in selected_price_indices:
        if history_index == 0:
            label = "Initial price"
        else:
            label = (
                f"Price after iteration "
                f"{history_index}"
            )

        plt.step(
            hours,
            price_history[history_index],
            where="mid",
            label=label,
        )

    plt.xlabel("Hour")
    plt.ylabel("Price (€/kWh)")
    plt.title(
        "Price Evolution with Partial EV Rescheduling"
    )
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_10_price_evolution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Demand profiles at selected iterations.
    selected_demand_indices = sorted(
        {
            0,
            min(1, completed_iterations),
            min(5, completed_iterations),
            min(20, completed_iterations),
            completed_iterations,
        }
    )

    plt.figure(figsize=(10, 5))

    for history_index in selected_demand_indices:
        if history_index == 0:
            label = "Initial EV response"
        else:
            label = (
                f"Demand after iteration "
                f"{history_index}"
            )

        plt.plot(
            hours,
            total_demand_history_mw[history_index],
            marker="o",
            label=label,
        )

    plt.xlabel("Hour")
    plt.ylabel("Total demand (MW)")
    plt.title(
        "Demand Evolution with Partial EV Rescheduling"
    )
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_10_demand_evolution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Raw and applied price changes.
    plt.figure(figsize=(10, 4))

    plt.plot(
        iteration_numbers,
        raw_price_change_history,
        marker="o",
        label="Raw price discrepancy",
    )

    plt.plot(
        iteration_numbers,
        applied_price_change_history,
        marker="o",
        label="Applied damped price change",
    )

    plt.axhline(
        convergence_price_tolerance_eur_per_kwh,
        linestyle="--",
        label="Price convergence tolerance",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Maximum price change (€/kWh)")
    plt.title("Price-Update Magnitudes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_10_price_changes.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Maximum demand change.
    plt.figure(figsize=(10, 4))

    plt.plot(
        iteration_numbers,
        demand_change_history_mw,
        marker="o",
    )

    plt.axhline(
        convergence_demand_tolerance_mw,
        linestyle="--",
        label="Demand convergence tolerance",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Maximum demand change (MW)")
    plt.title("Demand-Update Magnitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_10_demand_changes.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Number of EVs that changed schedules.
    plt.figure(figsize=(10, 4))

    plt.plot(
        iteration_numbers,
        changed_schedules_history,
        marker="o",
    )

    plt.xlabel("Iteration")
    plt.ylabel("EVs changing schedule")
    plt.title("Schedule Changes per Iteration")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_10_changed_schedules.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Peak demand across iterations.
    plt.figure(figsize=(10, 4))

    plt.plot(
        iteration_numbers,
        peak_demand_history_mw,
        marker="o",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Peak demand (MW)")
    plt.title("Peak Demand Across Iterations")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_10_peak_demand.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Figures saved:")
    print("  results/step_10_price_evolution.png")
    print("  results/step_10_demand_evolution.png")
    print("  results/step_10_price_changes.png")
    print("  results/step_10_demand_changes.png")
    print("  results/step_10_changed_schedules.png")
    print("  results/step_10_peak_demand.png")


if __name__ == "__main__":
    main()
