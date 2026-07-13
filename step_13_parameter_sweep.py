from __future__ import annotations

import csv
import json
from dataclasses import dataclass
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


@dataclass
class SimulationState:
    """Mutable state of the price-demand feedback simulation."""

    ev_schedules_kw: np.ndarray
    aggregate_ev_demand_kw: np.ndarray
    total_demand_mw: np.ndarray
    price_eur_per_kwh: np.ndarray

    def clone(self) -> "SimulationState":
        """Return a deep copy suitable for an independent scenario."""
        return SimulationState(
            ev_schedules_kw=self.ev_schedules_kw.copy(),
            aggregate_ev_demand_kw=self.aggregate_ev_demand_kw.copy(),
            total_demand_mw=self.total_demand_mw.copy(),
            price_eur_per_kwh=self.price_eur_per_kwh.copy(),
        )


def kw_to_mw(
    power_kw: np.ndarray | float,
) -> np.ndarray | float:
    """Convert power from kW to MW."""
    return power_kw / 1000.0


def calculate_schedule_cost_eur(
    schedule_kw: np.ndarray,
    price_eur_per_kwh: np.ndarray,
    interval_duration_hours: float,
) -> float:
    """Calculate the cost of one EV charging schedule."""
    if schedule_kw.shape != price_eur_per_kwh.shape:
        raise ValueError(
            "The charging schedule and price arrays must have the same shape."
        )

    return float(
        np.sum(
            schedule_kw
            * price_eur_per_kwh
            * interval_duration_hours
        )
    )


def calculate_price_eur_per_kwh(
    demand_mw: np.ndarray,
    quadratic_coefficient: float,
    linear_coefficient: float,
) -> np.ndarray:
    """Calculate the marginal price associated with aggregate demand."""
    price_eur_per_mwh = calculate_marginal_price_eur_per_mwh(
        demand_mw=demand_mw,
        quadratic_coefficient=quadratic_coefficient,
        linear_coefficient=linear_coefficient,
    )

    return eur_per_mwh_to_eur_per_kwh(
        price_eur_per_mwh
    )


def calculate_daily_generation_cost_eur(
    demand_mw: np.ndarray,
    quadratic_coefficient: float,
    linear_coefficient: float,
    interval_duration_hours: float,
) -> float:
    """Calculate total generation cost across all time intervals."""
    hourly_cost_eur = calculate_generation_cost_eur_per_hour(
        demand_mw=demand_mw,
        quadratic_coefficient=quadratic_coefficient,
        linear_coefficient=linear_coefficient,
    )

    return float(
        np.sum(
            hourly_cost_eur
            * interval_duration_hours
        )
    )


def perform_feedback_iteration(
    *,
    state: SimulationState,
    selected_ev_indices: np.ndarray,
    aggregate_fixed_demand_mw: np.ndarray,
    available_hours_by_ev: list[np.ndarray],
    required_energies_kwh: np.ndarray,
    maximum_charging_powers_kw: np.ndarray,
    interval_duration_hours: float,
    price_damping_factor: float,
    minimum_saving_to_reschedule_eur: float,
    quadratic_coefficient: float,
    linear_coefficient: float,
    attack_is_active: bool,
    receives_false_price: np.ndarray,
    attacked_hour: int,
    false_price_eur_per_kwh: float,
    unique_attacked_selected: np.ndarray | None = None,
    unique_attacked_rescheduled: np.ndarray | None = None,
    unique_attacked_moved_into_target: np.ndarray | None = None,
) -> dict[str, float | int]:
    """
    Execute one partial asynchronous feedback iteration.

    Only the selected EVs reconsider their schedules. An attacked EV uses
    the forged target-hour price while the attack is active. A schedule
    change is accepted only when its perceived saving reaches the
    configured hysteresis threshold.
    """
    previous_total_demand_mw = state.total_demand_mw.copy()

    candidate_changes = 0
    accepted_changes = 0
    attacked_accepted_changes = 0
    attacked_moved_into_target_changes = 0

    for ev_index_value in selected_ev_indices:
        ev_index = int(ev_index_value)

        is_attacked_ev = bool(
            attack_is_active
            and receives_false_price[ev_index]
        )

        if (
            is_attacked_ev
            and unique_attacked_selected is not None
        ):
            unique_attacked_selected[ev_index] = True

        previous_schedule_kw = (
            state.ev_schedules_kw[ev_index].copy()
        )

        perceived_price_eur_per_kwh = (
            state.price_eur_per_kwh.copy()
        )

        if is_attacked_ev:
            perceived_price_eur_per_kwh[
                attacked_hour
            ] = false_price_eur_per_kwh

        candidate_schedule_kw = create_ev_schedule(
            price_eur_per_kwh=perceived_price_eur_per_kwh,
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

        schedule_is_different = not np.allclose(
            previous_schedule_kw,
            candidate_schedule_kw,
            atol=1e-12,
        )

        if not schedule_is_different:
            continue

        candidate_changes += 1

        current_schedule_cost_eur = (
            calculate_schedule_cost_eur(
                schedule_kw=previous_schedule_kw,
                price_eur_per_kwh=perceived_price_eur_per_kwh,
                interval_duration_hours=interval_duration_hours,
            )
        )

        candidate_schedule_cost_eur = (
            calculate_schedule_cost_eur(
                schedule_kw=candidate_schedule_kw,
                price_eur_per_kwh=perceived_price_eur_per_kwh,
                interval_duration_hours=interval_duration_hours,
            )
        )

        expected_saving_eur = (
            current_schedule_cost_eur
            - candidate_schedule_cost_eur
        )

        if (
            expected_saving_eur + 1e-12
            < minimum_saving_to_reschedule_eur
        ):
            continue

        accepted_changes += 1

        moved_into_target = (
            candidate_schedule_kw[attacked_hour]
            > previous_schedule_kw[attacked_hour] + 1e-12
        )

        if is_attacked_ev:
            attacked_accepted_changes += 1

            if unique_attacked_rescheduled is not None:
                unique_attacked_rescheduled[ev_index] = True

            if moved_into_target:
                attacked_moved_into_target_changes += 1

                if (
                    unique_attacked_moved_into_target
                    is not None
                ):
                    unique_attacked_moved_into_target[
                        ev_index
                    ] = True

        state.aggregate_ev_demand_kw -= (
            previous_schedule_kw
        )

        state.aggregate_ev_demand_kw += (
            candidate_schedule_kw
        )

        state.ev_schedules_kw[ev_index] = (
            candidate_schedule_kw
        )

    aggregate_ev_demand_mw = kw_to_mw(
        state.aggregate_ev_demand_kw
    )

    state.total_demand_mw = (
        aggregate_fixed_demand_mw
        + aggregate_ev_demand_mw
    )

    raw_price_eur_per_kwh = (
        calculate_price_eur_per_kwh(
            demand_mw=state.total_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
        )
    )

    next_price_eur_per_kwh = (
        (1.0 - price_damping_factor)
        * state.price_eur_per_kwh
        + price_damping_factor
        * raw_price_eur_per_kwh
    )

    raw_price_discrepancy = float(
        np.max(
            np.abs(
                raw_price_eur_per_kwh
                - state.price_eur_per_kwh
            )
        )
    )

    applied_price_change = float(
        np.max(
            np.abs(
                next_price_eur_per_kwh
                - state.price_eur_per_kwh
            )
        )
    )

    maximum_demand_change_mw = float(
        np.max(
            np.abs(
                state.total_demand_mw
                - previous_total_demand_mw
            )
        )
    )

    state.price_eur_per_kwh = (
        next_price_eur_per_kwh
    )

    daily_generation_cost_eur = (
        calculate_daily_generation_cost_eur(
            demand_mw=state.total_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
            interval_duration_hours=interval_duration_hours,
        )
    )

    return {
        "candidate_changes": candidate_changes,
        "accepted_changes": accepted_changes,
        "attacked_accepted_changes": attacked_accepted_changes,
        "attacked_moved_into_target_changes": (
            attacked_moved_into_target_changes
        ),
        "raw_price_discrepancy": raw_price_discrepancy,
        "applied_price_change": applied_price_change,
        "maximum_demand_change_mw": maximum_demand_change_mw,
        "target_demand_mw": float(
            state.total_demand_mw[attacked_hour]
        ),
        "target_legitimate_price_eur_per_kwh": float(
            state.price_eur_per_kwh[attacked_hour]
        ),
        "daily_peak_demand_mw": float(
            np.max(state.total_demand_mw)
        ),
        "daily_generation_cost_eur": (
            daily_generation_cost_eur
        ),
    }


def write_results_csv(
    output_path: Path,
    records: list[dict[str, object]],
) -> None:
    """Write all sweep results to a CSV file."""
    if not records:
        raise ValueError("No result records were produced.")

    fieldnames = list(records[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records)


def build_result_matrix(
    records: list[dict[str, object]],
    attacked_fractions: np.ndarray,
    false_prices: np.ndarray,
    metric_name: str,
) -> np.ndarray:
    """Build a two-dimensional matrix for one sweep metric."""
    matrix = np.zeros(
        (
            len(attacked_fractions),
            len(false_prices),
        ),
        dtype=float,
    )

    record_lookup = {
        (
            float(record["attacked_fraction"]),
            float(
                record[
                    "false_price_eur_per_kwh"
                ]
            ),
        ): record
        for record in records
    }

    for row_index, attacked_fraction in enumerate(
        attacked_fractions
    ):
        for column_index, false_price in enumerate(
            false_prices
        ):
            record = record_lookup[
                (
                    float(attacked_fraction),
                    float(false_price),
                )
            ]

            matrix[
                row_index,
                column_index,
            ] = float(record[metric_name])

    return matrix


def save_heatmap(
    *,
    matrix: np.ndarray,
    attacked_fractions: np.ndarray,
    false_prices: np.ndarray,
    title: str,
    colorbar_label: str,
    output_path: Path,
    value_format: str = ".2f",
) -> None:
    """Save one annotated parameter-sweep heatmap."""
    plt.figure(figsize=(9, 6))

    image = plt.imshow(
        matrix,
        aspect="auto",
        origin="lower",
    )

    plt.colorbar(
        image,
        label=colorbar_label,
    )

    plt.xticks(
        np.arange(len(false_prices)),
        [
            f"{price:.2f}"
            for price in false_prices
        ],
    )

    plt.yticks(
        np.arange(len(attacked_fractions)),
        [
            f"{fraction:.0%}"
            for fraction in attacked_fractions
        ],
    )

    plt.xlabel(
        "Forged price at 18:00 (€/kWh)"
    )
    plt.ylabel("Attacked EV fraction")
    plt.title(title)

    for row_index in range(matrix.shape[0]):
        for column_index in range(
            matrix.shape[1]
        ):
            plt.text(
                column_index,
                row_index,
                format(
                    matrix[
                        row_index,
                        column_index,
                    ],
                    value_format,
                ),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    hours = np.arange(24)
    interval_duration_hours = 1.0

    # ============================================================
    # 1. Experiment configuration
    # ============================================================

    random_seed_population = 42
    random_seed_baseline_updates = 1001
    random_seed_attack_updates = 2001
    random_seed_recovery_updates = 3001
    random_seed_attack_order = 4001

    number_of_households = 10_000
    ev_adoption_rate = 0.30

    price_damping_factor = 0.20
    ev_rescheduling_fraction = 0.10
    minimum_saving_to_reschedule_eur = 0.10

    baseline_maximum_iterations = 150
    attack_iterations = 20
    recovery_maximum_iterations = 100

    convergence_price_tolerance_eur_per_kwh = 1e-4
    convergence_demand_tolerance_mw = 0.01
    convergence_patience = 5

    attacked_hour = 18

    attacked_fractions = np.array(
        [
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            1.00,
        ],
        dtype=float,
    )

    false_prices_eur_per_kwh = np.array(
        [
            0.00,
            0.05,
            0.10,
            0.20,
            0.30,
        ],
        dtype=float,
    )

    quadratic_coefficient = 6.190476
    linear_coefficient = 28.571432

    output_directory = Path("results")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # 2. Fixed household demand
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

    aggregate_fixed_demand_kw = (
        fixed_demand_per_household_kw
        * number_of_households
    )

    aggregate_fixed_demand_mw = kw_to_mw(
        aggregate_fixed_demand_kw
    )

    initial_price_eur_per_kwh = (
        calculate_price_eur_per_kwh(
            demand_mw=aggregate_fixed_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
        )
    )

    # ============================================================
    # 3. Heterogeneous EV population
    # ============================================================

    rng_population = np.random.default_rng(
        random_seed_population
    )

    number_of_evs = int(
        round(
            number_of_households
            * ev_adoption_rate
        )
    )

    arrival_hours = np.clip(
        np.rint(
            rng_population.normal(
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
            rng_population.normal(
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

    maximum_charging_powers_kw = (
        rng_population.choice(
            charging_power_options_kw,
            size=number_of_evs,
            p=charging_power_probabilities,
        )
    )

    raw_required_energies_kwh = np.clip(
        rng_population.normal(
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
            arrival_hour=int(
                arrival_hours[ev_index]
            ),
            departure_hour=int(
                departure_hours[ev_index]
            ),
        )

        maximum_possible_energy_kwh = (
            len(available_hours)
            * float(
                maximum_charging_powers_kw[
                    ev_index
                ]
            )
            * interval_duration_hours
        )

        available_hours_by_ev.append(
            available_hours
        )

        required_energies_kwh[ev_index] = min(
            float(
                raw_required_energies_kwh[
                    ev_index
                ]
            ),
            maximum_possible_energy_kwh,
        )

    expected_total_ev_energy_mwh = (
        float(
            np.sum(required_energies_kwh)
        )
        / 1000.0
    )

    connected_at_target = np.array(
        [
            attacked_hour
            in available_hours_by_ev[ev_index]
            for ev_index in range(number_of_evs)
        ],
        dtype=bool,
    )

    evs_selected_per_iteration = max(
        1,
        int(
            round(
                number_of_evs
                * ev_rescheduling_fraction
            )
        ),
    )

    # ============================================================
    # 4. Precompute identical update selections for comparability
    # ============================================================

    rng_baseline_updates = np.random.default_rng(
        random_seed_baseline_updates
    )
    rng_attack_updates = np.random.default_rng(
        random_seed_attack_updates
    )
    rng_recovery_updates = np.random.default_rng(
        random_seed_recovery_updates
    )
    rng_attack_order = np.random.default_rng(
        random_seed_attack_order
    )

    baseline_update_plan = [
        rng_baseline_updates.choice(
            number_of_evs,
            size=evs_selected_per_iteration,
            replace=False,
        )
        for _ in range(
            baseline_maximum_iterations
        )
    ]

    attack_update_plan = [
        rng_attack_updates.choice(
            number_of_evs,
            size=evs_selected_per_iteration,
            replace=False,
        )
        for _ in range(attack_iterations)
    ]

    recovery_update_plan = [
        rng_recovery_updates.choice(
            number_of_evs,
            size=evs_selected_per_iteration,
            replace=False,
        )
        for _ in range(
            recovery_maximum_iterations
        )
    ]

    # Nested attack sets make fraction comparisons easier:
    # every 50% attack set contains the corresponding 25% set.
    attack_order = rng_attack_order.permutation(
        number_of_evs
    )

    # ============================================================
    # 5. Create and converge the common baseline
    # ============================================================

    initial_ev_schedules_kw = np.zeros(
        (number_of_evs, 24),
        dtype=float,
    )

    for ev_index in range(number_of_evs):
        initial_ev_schedules_kw[
            ev_index
        ] = create_ev_schedule(
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

    initial_aggregate_ev_demand_kw = np.sum(
        initial_ev_schedules_kw,
        axis=0,
    )

    baseline_state = SimulationState(
        ev_schedules_kw=initial_ev_schedules_kw,
        aggregate_ev_demand_kw=initial_aggregate_ev_demand_kw,
        total_demand_mw=(
            aggregate_fixed_demand_mw
            + kw_to_mw(
                initial_aggregate_ev_demand_kw
            )
        ),
        price_eur_per_kwh=(
            initial_price_eur_per_kwh.copy()
        ),
    )

    no_attacked_evs = np.zeros(
        number_of_evs,
        dtype=bool,
    )

    baseline_converged = False
    baseline_completed_iterations = 0
    consecutive_converged_iterations = 0

    print("=== Parameter sweep ===")
    print()
    print("Common baseline")
    print(
        f"  Households: "
        f"{number_of_households:,}"
    )
    print(f"  EVs: {number_of_evs:,}")
    print(
        f"  Total EV energy: "
        f"{expected_total_ev_energy_mwh:.4f} MWh"
    )
    print(
        f"  EVs updated per iteration: "
        f"{evs_selected_per_iteration:,}"
    )
    print(
        f"  Price damping: "
        f"{price_damping_factor:.2f}"
    )
    print(
        f"  Rescheduling hysteresis: "
        f"€{minimum_saving_to_reschedule_eur:.2f}"
    )

    for iteration_index, selected_indices in enumerate(
        baseline_update_plan
    ):
        result = perform_feedback_iteration(
            state=baseline_state,
            selected_ev_indices=selected_indices,
            aggregate_fixed_demand_mw=aggregate_fixed_demand_mw,
            available_hours_by_ev=available_hours_by_ev,
            required_energies_kwh=required_energies_kwh,
            maximum_charging_powers_kw=maximum_charging_powers_kw,
            interval_duration_hours=interval_duration_hours,
            price_damping_factor=price_damping_factor,
            minimum_saving_to_reschedule_eur=(
                minimum_saving_to_reschedule_eur
            ),
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
            attack_is_active=False,
            receives_false_price=no_attacked_evs,
            attacked_hour=attacked_hour,
            false_price_eur_per_kwh=0.0,
        )

        baseline_completed_iterations = (
            iteration_index + 1
        )

        if (
            float(
                result[
                    "applied_price_change"
                ]
            )
            <= convergence_price_tolerance_eur_per_kwh
            and float(
                result[
                    "maximum_demand_change_mw"
                ]
            )
            <= convergence_demand_tolerance_mw
        ):
            consecutive_converged_iterations += 1
        else:
            consecutive_converged_iterations = 0

        if (
            consecutive_converged_iterations
            >= convergence_patience
        ):
            baseline_converged = True
            break

    if not baseline_converged:
        raise RuntimeError(
            "The common no-attack baseline did not converge. "
            "Increase baseline_maximum_iterations or revise the "
            "feedback parameters before running the sweep."
        )

    baseline_target_demand_mw = float(
        baseline_state.total_demand_mw[
            attacked_hour
        ]
    )

    baseline_daily_peak_mw = float(
        np.max(
            baseline_state.total_demand_mw
        )
    )

    baseline_target_price_eur_per_kwh = float(
        baseline_state.price_eur_per_kwh[
            attacked_hour
        ]
    )

    baseline_generation_cost_eur = (
        calculate_daily_generation_cost_eur(
            demand_mw=baseline_state.total_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
            interval_duration_hours=interval_duration_hours,
        )
    )

    print(
        f"  Converged after "
        f"{baseline_completed_iterations} iterations"
    )
    print(
        f"  Demand at {attacked_hour:02d}:00: "
        f"{baseline_target_demand_mw:.2f} MW"
    )
    print(
        f"  Price at {attacked_hour:02d}:00: "
        f"€{baseline_target_price_eur_per_kwh:.3f}/kWh"
    )
    print(
        f"  Daily generation cost: "
        f"€{baseline_generation_cost_eur:.2f}"
    )

    # ============================================================
    # 6. Run every attack configuration
    # ============================================================

    result_records: list[dict[str, object]] = []

    total_scenarios = (
        len(attacked_fractions)
        * len(false_prices_eur_per_kwh)
    )
    scenario_number = 0

    print()
    print(
        "Scenario | Attacked | False price | "
        "Unique rescheduled | Max Δ18 | Max cost Δ | "
        "Recovery"
    )
    print("-" * 103)

    for attacked_fraction in attacked_fractions:
        number_of_attacked_evs = int(
            round(
                number_of_evs
                * float(attacked_fraction)
            )
        )

        attacked_indices = attack_order[
            :number_of_attacked_evs
        ]

        receives_false_price = np.zeros(
            number_of_evs,
            dtype=bool,
        )
        receives_false_price[
            attacked_indices
        ] = True

        attacked_connected_at_target = int(
            np.sum(
                receives_false_price
                & connected_at_target
            )
        )

        for false_price_eur_per_kwh in (
            false_prices_eur_per_kwh
        ):
            scenario_number += 1

            scenario_state = baseline_state.clone()

            unique_attacked_selected = np.zeros(
                number_of_evs,
                dtype=bool,
            )
            unique_attacked_rescheduled = np.zeros(
                number_of_evs,
                dtype=bool,
            )
            unique_attacked_moved_into_target = np.zeros(
                number_of_evs,
                dtype=bool,
            )

            maximum_target_demand_mw = (
                baseline_target_demand_mw
            )
            maximum_target_price_eur_per_kwh = (
                baseline_target_price_eur_per_kwh
            )
            maximum_daily_peak_mw = (
                baseline_daily_peak_mw
            )
            maximum_attack_generation_cost_eur = (
                baseline_generation_cost_eur
            )

            total_attack_accepted_changes = 0
            total_attacked_accepted_changes = 0
            total_moved_into_target_changes = 0

            # -----------------------------
            # Attack phase
            # -----------------------------
            for selected_indices in attack_update_plan:
                result = perform_feedback_iteration(
                    state=scenario_state,
                    selected_ev_indices=selected_indices,
                    aggregate_fixed_demand_mw=aggregate_fixed_demand_mw,
                    available_hours_by_ev=available_hours_by_ev,
                    required_energies_kwh=required_energies_kwh,
                    maximum_charging_powers_kw=maximum_charging_powers_kw,
                    interval_duration_hours=interval_duration_hours,
                    price_damping_factor=price_damping_factor,
                    minimum_saving_to_reschedule_eur=(
                        minimum_saving_to_reschedule_eur
                    ),
                    quadratic_coefficient=quadratic_coefficient,
                    linear_coefficient=linear_coefficient,
                    attack_is_active=True,
                    receives_false_price=receives_false_price,
                    attacked_hour=attacked_hour,
                    false_price_eur_per_kwh=float(
                        false_price_eur_per_kwh
                    ),
                    unique_attacked_selected=unique_attacked_selected,
                    unique_attacked_rescheduled=(
                        unique_attacked_rescheduled
                    ),
                    unique_attacked_moved_into_target=(
                        unique_attacked_moved_into_target
                    ),
                )

                total_attack_accepted_changes += int(
                    result[
                        "accepted_changes"
                    ]
                )
                total_attacked_accepted_changes += int(
                    result[
                        "attacked_accepted_changes"
                    ]
                )
                total_moved_into_target_changes += int(
                    result[
                        "attacked_moved_into_target_changes"
                    ]
                )

                maximum_target_demand_mw = max(
                    maximum_target_demand_mw,
                    float(
                        result[
                            "target_demand_mw"
                        ]
                    ),
                )

                maximum_target_price_eur_per_kwh = max(
                    maximum_target_price_eur_per_kwh,
                    float(
                        result[
                            "target_legitimate_price_eur_per_kwh"
                        ]
                    ),
                )

                maximum_daily_peak_mw = max(
                    maximum_daily_peak_mw,
                    float(
                        result[
                            "daily_peak_demand_mw"
                        ]
                    ),
                )

                maximum_attack_generation_cost_eur = max(
                    maximum_attack_generation_cost_eur,
                    float(
                        result[
                            "daily_generation_cost_eur"
                        ]
                    ),
                )

            attack_end_target_demand_mw = float(
                scenario_state.total_demand_mw[
                    attacked_hour
                ]
            )

            attack_end_target_price_eur_per_kwh = float(
                scenario_state.price_eur_per_kwh[
                    attacked_hour
                ]
            )

            attack_end_daily_peak_mw = float(
                np.max(
                    scenario_state.total_demand_mw
                )
            )

            attack_end_generation_cost_eur = (
                calculate_daily_generation_cost_eur(
                    demand_mw=scenario_state.total_demand_mw,
                    quadratic_coefficient=quadratic_coefficient,
                    linear_coefficient=linear_coefficient,
                    interval_duration_hours=interval_duration_hours,
                )
            )

            # -----------------------------
            # Recovery phase
            # -----------------------------
            recovery_converged = False
            recovery_iterations = 0
            consecutive_converged_iterations = 0

            for (
                recovery_iteration_index,
                selected_indices,
            ) in enumerate(recovery_update_plan):
                result = perform_feedback_iteration(
                    state=scenario_state,
                    selected_ev_indices=selected_indices,
                    aggregate_fixed_demand_mw=aggregate_fixed_demand_mw,
                    available_hours_by_ev=available_hours_by_ev,
                    required_energies_kwh=required_energies_kwh,
                    maximum_charging_powers_kw=maximum_charging_powers_kw,
                    interval_duration_hours=interval_duration_hours,
                    price_damping_factor=price_damping_factor,
                    minimum_saving_to_reschedule_eur=(
                        minimum_saving_to_reschedule_eur
                    ),
                    quadratic_coefficient=quadratic_coefficient,
                    linear_coefficient=linear_coefficient,
                    attack_is_active=False,
                    receives_false_price=receives_false_price,
                    attacked_hour=attacked_hour,
                    false_price_eur_per_kwh=float(
                        false_price_eur_per_kwh
                    ),
                )

                recovery_iterations = (
                    recovery_iteration_index + 1
                )

                if (
                    float(
                        result[
                            "applied_price_change"
                        ]
                    )
                    <= convergence_price_tolerance_eur_per_kwh
                    and float(
                        result[
                            "maximum_demand_change_mw"
                        ]
                    )
                    <= convergence_demand_tolerance_mw
                ):
                    consecutive_converged_iterations += 1
                else:
                    consecutive_converged_iterations = 0

                if (
                    consecutive_converged_iterations
                    >= convergence_patience
                ):
                    recovery_converged = True
                    break

            recovered_target_demand_mw = float(
                scenario_state.total_demand_mw[
                    attacked_hour
                ]
            )

            recovered_generation_cost_eur = (
                calculate_daily_generation_cost_eur(
                    demand_mw=scenario_state.total_demand_mw,
                    quadratic_coefficient=quadratic_coefficient,
                    linear_coefficient=linear_coefficient,
                    interval_duration_hours=interval_duration_hours,
                )
            )

            final_distance_from_baseline_mw = float(
                np.max(
                    np.abs(
                        scenario_state.total_demand_mw
                        - baseline_state.total_demand_mw
                    )
                )
            )

            unique_attacked_selected_count = int(
                np.sum(
                    unique_attacked_selected
                )
            )

            unique_attacked_rescheduled_count = int(
                np.sum(
                    unique_attacked_rescheduled
                )
            )

            unique_moved_into_target_count = int(
                np.sum(
                    unique_attacked_moved_into_target
                )
            )

            maximum_target_increase_mw = (
                maximum_target_demand_mw
                - baseline_target_demand_mw
            )

            maximum_target_increase_percent = (
                100.0
                * maximum_target_increase_mw
                / baseline_target_demand_mw
            )

            attack_end_target_increase_mw = (
                attack_end_target_demand_mw
                - baseline_target_demand_mw
            )

            attack_end_target_increase_percent = (
                100.0
                * attack_end_target_increase_mw
                / baseline_target_demand_mw
            )

            maximum_generation_cost_increase_eur = (
                maximum_attack_generation_cost_eur
                - baseline_generation_cost_eur
            )

            attack_end_generation_cost_increase_eur = (
                attack_end_generation_cost_eur
                - baseline_generation_cost_eur
            )

            recovered_generation_cost_difference_eur = (
                recovered_generation_cost_eur
                - baseline_generation_cost_eur
            )

            result_record: dict[str, object] = {
                "attacked_fraction": float(
                    attacked_fraction
                ),
                "false_price_eur_per_kwh": float(
                    false_price_eur_per_kwh
                ),
                "attacked_evs": number_of_attacked_evs,
                "attacked_evs_connected_at_target": (
                    attacked_connected_at_target
                ),
                "unique_attacked_evs_selected": (
                    unique_attacked_selected_count
                ),
                "unique_attacked_evs_rescheduled": (
                    unique_attacked_rescheduled_count
                ),
                "unique_attacked_evs_moved_into_target": (
                    unique_moved_into_target_count
                ),
                "total_attack_accepted_changes": (
                    total_attack_accepted_changes
                ),
                "total_attacked_accepted_changes": (
                    total_attacked_accepted_changes
                ),
                "total_moved_into_target_changes": (
                    total_moved_into_target_changes
                ),
                "baseline_target_demand_mw": (
                    baseline_target_demand_mw
                ),
                "maximum_target_demand_mw": (
                    maximum_target_demand_mw
                ),
                "maximum_target_increase_mw": (
                    maximum_target_increase_mw
                ),
                "maximum_target_increase_percent": (
                    maximum_target_increase_percent
                ),
                "attack_end_target_demand_mw": (
                    attack_end_target_demand_mw
                ),
                "attack_end_target_increase_mw": (
                    attack_end_target_increase_mw
                ),
                "attack_end_target_increase_percent": (
                    attack_end_target_increase_percent
                ),
                "baseline_target_price_eur_per_kwh": (
                    baseline_target_price_eur_per_kwh
                ),
                "maximum_target_legitimate_price_eur_per_kwh": (
                    maximum_target_price_eur_per_kwh
                ),
                "attack_end_target_legitimate_price_eur_per_kwh": (
                    attack_end_target_price_eur_per_kwh
                ),
                "baseline_daily_peak_mw": (
                    baseline_daily_peak_mw
                ),
                "maximum_daily_peak_mw": (
                    maximum_daily_peak_mw
                ),
                "attack_end_daily_peak_mw": (
                    attack_end_daily_peak_mw
                ),
                "baseline_daily_generation_cost_eur": (
                    baseline_generation_cost_eur
                ),
                "maximum_attack_daily_generation_cost_eur": (
                    maximum_attack_generation_cost_eur
                ),
                "maximum_generation_cost_increase_eur": (
                    maximum_generation_cost_increase_eur
                ),
                "attack_end_generation_cost_eur": (
                    attack_end_generation_cost_eur
                ),
                "attack_end_generation_cost_increase_eur": (
                    attack_end_generation_cost_increase_eur
                ),
                "recovery_converged": recovery_converged,
                "recovery_iterations": recovery_iterations,
                "recovered_target_demand_mw": (
                    recovered_target_demand_mw
                ),
                "final_distance_from_baseline_mw": (
                    final_distance_from_baseline_mw
                ),
                "recovered_generation_cost_eur": (
                    recovered_generation_cost_eur
                ),
                "recovered_generation_cost_difference_eur": (
                    recovered_generation_cost_difference_eur
                ),
            }

            result_records.append(
                result_record
            )

            print(
                f"{scenario_number:3d}/"
                f"{total_scenarios:<3d} | "
                f"{attacked_fraction:7.0%} | "
                f"€{false_price_eur_per_kwh:10.2f} | "
                f"{unique_attacked_rescheduled_count:18d} | "
                f"{maximum_target_increase_mw:7.2f} MW | "
                f"€{maximum_generation_cost_increase_eur:9.2f} | "
                f"{recovery_iterations:3d} "
                f"({'yes' if recovery_converged else 'no'})"
            )

    # ============================================================
    # 7. Save tabular results and configuration
    # ============================================================

    csv_output_path = (
        output_directory
        / "step_13_parameter_sweep_results.csv"
    )

    write_results_csv(
        output_path=csv_output_path,
        records=result_records,
    )

    configuration = {
        "random_seeds": {
            "population": random_seed_population,
            "baseline_updates": (
                random_seed_baseline_updates
            ),
            "attack_updates": (
                random_seed_attack_updates
            ),
            "recovery_updates": (
                random_seed_recovery_updates
            ),
            "attack_order": (
                random_seed_attack_order
            ),
        },
        "number_of_households": (
            number_of_households
        ),
        "number_of_evs": number_of_evs,
        "ev_adoption_rate": ev_adoption_rate,
        "price_damping_factor": (
            price_damping_factor
        ),
        "ev_rescheduling_fraction": (
            ev_rescheduling_fraction
        ),
        "minimum_saving_to_reschedule_eur": (
            minimum_saving_to_reschedule_eur
        ),
        "baseline_maximum_iterations": (
            baseline_maximum_iterations
        ),
        "baseline_completed_iterations": (
            baseline_completed_iterations
        ),
        "attack_iterations": attack_iterations,
        "recovery_maximum_iterations": (
            recovery_maximum_iterations
        ),
        "attacked_hour": attacked_hour,
        "attacked_fractions": (
            attacked_fractions.tolist()
        ),
        "false_prices_eur_per_kwh": (
            false_prices_eur_per_kwh.tolist()
        ),
        "baseline_target_demand_mw": (
            baseline_target_demand_mw
        ),
        "baseline_target_price_eur_per_kwh": (
            baseline_target_price_eur_per_kwh
        ),
        "baseline_daily_peak_mw": (
            baseline_daily_peak_mw
        ),
        "baseline_generation_cost_eur": (
            baseline_generation_cost_eur
        ),
    }

    config_output_path = (
        output_directory
        / "step_13_parameter_sweep_config.json"
    )

    config_output_path.write_text(
        json.dumps(
            configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # 8. Heatmaps
    # ============================================================

    heatmap_metrics = [
        (
            "maximum_target_increase_mw",
            "Maximum Demand Increase at 18:00",
            "Demand increase (MW)",
            "step_13_heatmap_demand_increase.png",
            ".2f",
        ),
        (
            "maximum_target_increase_percent",
            "Maximum Relative Demand Increase at 18:00",
            "Demand increase (%)",
            "step_13_heatmap_demand_increase_percent.png",
            ".2f",
        ),
        (
            "unique_attacked_evs_rescheduled",
            "Unique Attacked EVs That Rescheduled",
            "Unique EVs",
            "step_13_heatmap_unique_rescheduled.png",
            ".0f",
        ),
        (
            "unique_attacked_evs_moved_into_target",
            "Unique Attacked EVs That Moved Charging into 18:00",
            "Unique EVs",
            "step_13_heatmap_moved_into_target.png",
            ".0f",
        ),
        (
            "maximum_target_legitimate_price_eur_per_kwh",
            "Maximum Legitimate Price at 18:00 During Attack",
            "Legitimate price (€/kWh)",
            "step_13_heatmap_legitimate_price.png",
            ".3f",
        ),
        (
            "maximum_generation_cost_increase_eur",
            "Maximum Daily Generation-Cost Increase",
            "Cost increase (€)",
            "step_13_heatmap_generation_cost.png",
            ".0f",
        ),
        (
            "maximum_daily_peak_mw",
            "Maximum Daily Peak Demand",
            "Peak demand (MW)",
            "step_13_heatmap_daily_peak.png",
            ".2f",
        ),
        (
            "recovery_iterations",
            "Recovery Iterations",
            "Iterations",
            "step_13_heatmap_recovery_iterations.png",
            ".0f",
        ),
        (
            "final_distance_from_baseline_mw",
            "Final Demand-Profile Distance from Baseline",
            "Maximum distance (MW)",
            "step_13_heatmap_final_distance.png",
            ".3f",
        ),
    ]

    for (
        metric_name,
        title,
        colorbar_label,
        filename,
        value_format,
    ) in heatmap_metrics:
        matrix = build_result_matrix(
            records=result_records,
            attacked_fractions=attacked_fractions,
            false_prices=false_prices_eur_per_kwh,
            metric_name=metric_name,
        )

        save_heatmap(
            matrix=matrix,
            attacked_fractions=attacked_fractions,
            false_prices=false_prices_eur_per_kwh,
            title=title,
            colorbar_label=colorbar_label,
            output_path=(
                output_directory / filename
            ),
            value_format=value_format,
        )

    # ============================================================
    # 9. Line plots
    # ============================================================

    # Demand increase versus attack fraction.
    plt.figure(figsize=(10, 5))

    for false_price in false_prices_eur_per_kwh:
        matching_records = sorted(
            [
                record
                for record in result_records
                if np.isclose(
                    float(
                        record[
                            "false_price_eur_per_kwh"
                        ]
                    ),
                    float(false_price),
                )
            ],
            key=lambda record: float(
                record[
                    "attacked_fraction"
                ]
            ),
        )

        plt.plot(
            [
                100.0
                * float(
                    record[
                        "attacked_fraction"
                    ]
                )
                for record in matching_records
            ],
            [
                float(
                    record[
                        "maximum_target_increase_mw"
                    ]
                )
                for record in matching_records
            ],
            marker="o",
            label=(
                f"False price €{false_price:.2f}/kWh"
            ),
        )

    plt.xlabel("Attacked EV fraction (%)")
    plt.ylabel(
        "Maximum demand increase at 18:00 (MW)"
    )
    plt.title(
        "Attack Impact versus Compromised Population"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_13_demand_increase_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Generation-cost increase versus attack fraction.
    plt.figure(figsize=(10, 5))

    for false_price in false_prices_eur_per_kwh:
        matching_records = sorted(
            [
                record
                for record in result_records
                if np.isclose(
                    float(
                        record[
                            "false_price_eur_per_kwh"
                        ]
                    ),
                    float(false_price),
                )
            ],
            key=lambda record: float(
                record[
                    "attacked_fraction"
                ]
            ),
        )

        plt.plot(
            [
                100.0
                * float(
                    record[
                        "attacked_fraction"
                    ]
                )
                for record in matching_records
            ],
            [
                float(
                    record[
                        "maximum_generation_cost_increase_eur"
                    ]
                )
                for record in matching_records
            ],
            marker="o",
            label=(
                f"False price €{false_price:.2f}/kWh"
            ),
        )

    plt.xlabel("Attacked EV fraction (%)")
    plt.ylabel(
        "Maximum daily generation-cost increase (€)"
    )
    plt.title(
        "Economic Impact versus Compromised Population"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_13_generation_cost_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print()
    print("Results saved:")
    print(
        "  results/"
        "step_13_parameter_sweep_results.csv"
    )
    print(
        "  results/"
        "step_13_parameter_sweep_config.json"
    )
    print(
        "  results/step_13_heatmap_*.png"
    )
    print(
        "  results/"
        "step_13_demand_increase_curves.png"
    )
    print(
        "  results/"
        "step_13_generation_cost_curves.png"
    )


if __name__ == "__main__":
    main()
