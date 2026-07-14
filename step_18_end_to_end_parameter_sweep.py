from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# Use a non-interactive backend because the script runs without a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from step_17_end_to_end_regional_attack import (
        REGION_A,
        REGION_B,
        NUMBER_OF_REGIONS,
        FeedbackState,
        build_three_bus_network,
        calculate_price_eur_per_kwh,
        kw_to_mw,
        perform_feedback_iteration,
        run_power_flow,
    )
except ImportError as exc:
    raise SystemExit(
        "Could not import Step 17.\n"
        "Place this file in the same directory as:\n"
        "    step_17_end_to_end_regional_attack.py\n"
        "and ensure that pandapower is installed."
    ) from exc

from step_02_ev_charging import (
    create_ev_schedule,
    get_available_hours,
)


def evaluate_target_hour_power_flow(
    *,
    fixed_demand_mw_by_region: np.ndarray,
    total_demand_mw_by_region: np.ndarray,
    attacked_hour: int,
    fixed_load_power_factor: float,
    ev_power_factor: float,
    minimum_voltage_limit_pu: float,
    maximum_voltage_limit_pu: float,
) -> dict[str, object]:
    """
    Run one AC power flow at the attacked hour.

    Fixed demand and EV demand are represented as separate load elements so
    that they can use different power factors.
    """
    ev_demand_mw_by_region = (
        total_demand_mw_by_region
        - fixed_demand_mw_by_region
    )

    network = build_three_bus_network(
        region_a_fixed_p_mw=float(
            fixed_demand_mw_by_region[
                REGION_A,
                attacked_hour,
            ]
        ),
        region_a_ev_p_mw=float(
            ev_demand_mw_by_region[
                REGION_A,
                attacked_hour,
            ]
        ),
        region_b_fixed_p_mw=float(
            fixed_demand_mw_by_region[
                REGION_B,
                attacked_hour,
            ]
        ),
        region_b_ev_p_mw=float(
            ev_demand_mw_by_region[
                REGION_B,
                attacked_hour,
            ]
        ),
        fixed_load_power_factor=fixed_load_power_factor,
        ev_power_factor=ev_power_factor,
    )

    converged = run_power_flow(network)

    if not converged:
        return {
            "power_flow_converged": False,
            "minimum_voltage_pu": np.nan,
            "bus_1_voltage_pu": np.nan,
            "bus_2_voltage_pu": np.nan,
            "line_0_loading_percent": np.nan,
            "line_1_loading_percent": np.nan,
            "maximum_line_loading_percent": np.nan,
            "total_line_losses_mw": np.nan,
            "voltage_violation": False,
            "line_overload": False,
        }

    voltage_violation = bool(
        np.any(
            (
                network.res_bus["vm_pu"]
                < minimum_voltage_limit_pu
            )
            | (
                network.res_bus["vm_pu"]
                > maximum_voltage_limit_pu
            )
        )
    )

    line_overload = bool(
        np.any(
            network.res_line[
                "loading_percent"
            ]
            > 100.0
        )
    )

    return {
        "power_flow_converged": True,
        "minimum_voltage_pu": float(
            network.res_bus["vm_pu"].min()
        ),
        "bus_1_voltage_pu": float(
            network.res_bus.at[
                1,
                "vm_pu",
            ]
        ),
        "bus_2_voltage_pu": float(
            network.res_bus.at[
                2,
                "vm_pu",
            ]
        ),
        "line_0_loading_percent": float(
            network.res_line.at[
                0,
                "loading_percent",
            ]
        ),
        "line_1_loading_percent": float(
            network.res_line.at[
                1,
                "loading_percent",
            ]
        ),
        "maximum_line_loading_percent": float(
            network.res_line[
                "loading_percent"
            ].max()
        ),
        "total_line_losses_mw": float(
            network.res_line[
                "pl_mw"
            ].sum()
        ),
        "voltage_violation": voltage_violation,
        "line_overload": line_overload,
    }


def build_metric_matrix(
    *,
    results: pd.DataFrame,
    attacked_fractions: np.ndarray,
    false_prices_eur_per_kwh: np.ndarray,
    attack_duration_iterations: int,
    metric_name: str,
) -> np.ndarray:
    """Build one fraction-by-price matrix for a fixed attack duration."""
    duration_results = results[
        results[
            "attack_duration_iterations"
        ]
        == attack_duration_iterations
    ]

    matrix = np.full(
        (
            len(attacked_fractions),
            len(false_prices_eur_per_kwh),
        ),
        np.nan,
        dtype=float,
    )

    for row_index, attacked_fraction in enumerate(
        attacked_fractions
    ):
        for column_index, false_price in enumerate(
            false_prices_eur_per_kwh
        ):
            matching_rows = duration_results[
                np.isclose(
                    duration_results[
                        "attacked_region_b_ev_fraction"
                    ],
                    attacked_fraction,
                )
                & np.isclose(
                    duration_results[
                        "false_price_eur_per_kwh"
                    ],
                    false_price,
                )
            ]

            if matching_rows.empty:
                continue

            matrix[
                row_index,
                column_index,
            ] = float(
                matching_rows.iloc[0][
                    metric_name
                ]
            )

    return matrix


def save_heatmap(
    *,
    matrix: np.ndarray,
    attacked_fractions: np.ndarray,
    false_prices_eur_per_kwh: np.ndarray,
    title: str,
    colorbar_label: str,
    output_path: Path,
    value_format: str,
) -> None:
    """Save one annotated parameter heatmap."""
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
        np.arange(
            len(false_prices_eur_per_kwh)
        ),
        [
            f"{value:.2f}"
            for value in false_prices_eur_per_kwh
        ],
    )

    plt.yticks(
        np.arange(
            len(attacked_fractions)
        ),
        [
            f"{value:.0%}"
            for value in attacked_fractions
        ],
    )

    plt.xlabel(
        "Forged price at 18:00 (€/kWh)"
    )
    plt.ylabel(
        "Attacked fraction of Region-B EVs"
    )
    plt.title(title)

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index,
            ]

            if np.isnan(value):
                text = "n/a"
            else:
                text = format(
                    value,
                    value_format,
                )

            plt.text(
                column_index,
                row_index,
                text,
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


def build_duration_threshold_summary(
    *,
    results: pd.DataFrame,
    attacked_fractions: np.ndarray,
    false_prices_eur_per_kwh: np.ndarray,
) -> pd.DataFrame:
    """
    Find the first sampled attack duration that causes each violation.

    NaN means that the violation was not observed for any tested duration.
    """
    records: list[dict[str, object]] = []

    for attacked_fraction in attacked_fractions:
        for false_price in (
            false_prices_eur_per_kwh
        ):
            scenario_results = results[
                np.isclose(
                    results[
                        "attacked_region_b_ev_fraction"
                    ],
                    attacked_fraction,
                )
                & np.isclose(
                    results[
                        "false_price_eur_per_kwh"
                    ],
                    false_price,
                )
            ].sort_values(
                "attack_duration_iterations"
            )

            voltage_rows = scenario_results[
                scenario_results[
                    "maximum_voltage_violation"
                ]
            ]

            overload_rows = scenario_results[
                scenario_results[
                    "maximum_line_overload"
                ]
            ]

            nonconvergent_rows = (
                scenario_results[
                    ~scenario_results[
                        "maximum_power_flow_converged"
                    ]
                ]
            )

            first_voltage_duration = (
                float(
                    voltage_rows[
                        "attack_duration_iterations"
                    ].min()
                )
                if not voltage_rows.empty
                else float("nan")
            )

            first_overload_duration = (
                float(
                    overload_rows[
                        "attack_duration_iterations"
                    ].min()
                )
                if not overload_rows.empty
                else float("nan")
            )

            first_nonconvergence_duration = (
                float(
                    nonconvergent_rows[
                        "attack_duration_iterations"
                    ].min()
                )
                if not nonconvergent_rows.empty
                else float("nan")
            )

            records.append(
                {
                    "attacked_region_b_ev_fraction": float(
                        attacked_fraction
                    ),
                    "false_price_eur_per_kwh": float(
                        false_price
                    ),
                    "first_sampled_voltage_violation_duration": (
                        first_voltage_duration
                    ),
                    "first_sampled_line_overload_duration": (
                        first_overload_duration
                    ),
                    "first_sampled_nonconvergence_duration": (
                        first_nonconvergence_duration
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def main() -> None:
    hours = np.arange(24)
    interval_duration_hours = 1.0

    # ============================================================
    # 1. Experiment configuration
    # ============================================================

    random_seed_population = 42
    random_seed_region_assignment = 3001
    random_seed_baseline_updates = 1001
    random_seed_attack_updates = 2001
    random_seed_attack_order = 4001

    number_of_households = 10_000
    households_per_region = (
        number_of_households // 2
    )
    ev_adoption_rate = 0.30

    price_damping_factor = 0.20
    ev_rescheduling_fraction = 0.10
    minimum_saving_to_reschedule_eur = 0.10

    baseline_maximum_iterations = 150

    convergence_price_tolerance_eur_per_kwh = 1e-4
    convergence_demand_tolerance_mw = 0.01
    convergence_patience = 5

    attacked_hour = 18

    attacked_region_b_ev_fractions = np.array(
        [
            0.25,
            0.50,
            0.75,
            1.00,
        ],
        dtype=float,
    )

    false_prices_eur_per_kwh = np.array(
        [
            0.05,
            0.10,
            0.20,
            0.30,
        ],
        dtype=float,
    )

    attack_durations_iterations = np.array(
        [
            5,
            10,
            20,
            40,
        ],
        dtype=int,
    )

    maximum_attack_iterations = int(
        np.max(
            attack_durations_iterations
        )
    )

    quadratic_coefficient = 6.190476
    linear_coefficient = 28.571432

    fixed_load_power_factor = 0.95
    ev_power_factor = 0.99

    minimum_voltage_limit_pu = 0.95
    maximum_voltage_limit_pu = 1.05

    output_directory = Path("results")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # 2. Regional fixed-demand profiles
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

    aggregate_fixed_demand_kw_by_region = np.vstack(
        [
            fixed_demand_per_household_kw
            * households_per_region,
            fixed_demand_per_household_kw
            * households_per_region,
        ]
    )

    aggregate_fixed_demand_mw_by_region = kw_to_mw(
        aggregate_fixed_demand_kw_by_region
    )

    total_fixed_system_demand_mw = np.sum(
        aggregate_fixed_demand_mw_by_region,
        axis=0,
    )

    initial_legitimate_price_eur_per_kwh = (
        calculate_price_eur_per_kwh(
            total_system_demand_mw=(
                total_fixed_system_demand_mw
            ),
            quadratic_coefficient=(
                quadratic_coefficient
            ),
            linear_coefficient=(
                linear_coefficient
            ),
        )
    )

    # ============================================================
    # 3. Generate the regional EV population once
    # ============================================================

    rng_population = np.random.default_rng(
        random_seed_population
    )

    rng_region_assignment = np.random.default_rng(
        random_seed_region_assignment
    )

    number_of_evs = int(
        round(
            number_of_households
            * ev_adoption_rate
        )
    )

    evs_per_region = number_of_evs // 2

    region_by_ev = np.array(
        [REGION_A] * evs_per_region
        + [REGION_B]
        * (
            number_of_evs
            - evs_per_region
        ),
        dtype=int,
    )

    rng_region_assignment.shuffle(
        region_by_ev
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
        [
            3.6,
            7.2,
            11.0,
        ],
        dtype=float,
    )

    charging_power_probabilities = np.array(
        [
            0.60,
            0.30,
            0.10,
        ],
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

    available_hours_by_ev: list[
        np.ndarray
    ] = []

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

        required_energies_kwh[
            ev_index
        ] = min(
            float(
                raw_required_energies_kwh[
                    ev_index
                ]
            ),
            maximum_possible_energy_kwh,
        )

    region_b_ev_indices = np.flatnonzero(
        region_by_ev == REGION_B
    )

    rng_attack_order = np.random.default_rng(
        random_seed_attack_order
    )

    nested_region_b_attack_order = (
        rng_attack_order.permutation(
            region_b_ev_indices
        )
    )

    connected_at_target = np.array(
        [
            attacked_hour
            in available_hours_by_ev[
                ev_index
            ]
            for ev_index in range(
                number_of_evs
            )
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
    # 4. Create and converge one common baseline
    # ============================================================

    initial_ev_schedules_kw = np.zeros(
        (
            number_of_evs,
            24,
        ),
        dtype=float,
    )

    initial_aggregate_ev_demand_kw_by_region = (
        np.zeros(
            (
                NUMBER_OF_REGIONS,
                24,
            ),
            dtype=float,
        )
    )

    for ev_index in range(number_of_evs):
        schedule_kw = create_ev_schedule(
            price_eur_per_kwh=(
                initial_legitimate_price_eur_per_kwh
            ),
            available_hours=(
                available_hours_by_ev[
                    ev_index
                ]
            ),
            required_energy_kwh=float(
                required_energies_kwh[
                    ev_index
                ]
            ),
            maximum_charging_power_kw=float(
                maximum_charging_powers_kw[
                    ev_index
                ]
            ),
            interval_duration_hours=(
                interval_duration_hours
            ),
            optimize_for_price=True,
        )

        initial_ev_schedules_kw[
            ev_index
        ] = schedule_kw

        initial_aggregate_ev_demand_kw_by_region[
            int(region_by_ev[ev_index])
        ] += schedule_kw

    baseline_state = FeedbackState(
        ev_schedules_kw=(
            initial_ev_schedules_kw
        ),
        aggregate_ev_demand_kw_by_region=(
            initial_aggregate_ev_demand_kw_by_region
        ),
        total_demand_mw_by_region=(
            aggregate_fixed_demand_mw_by_region
            + kw_to_mw(
                initial_aggregate_ev_demand_kw_by_region
            )
        ),
        legitimate_price_eur_per_kwh=(
            initial_legitimate_price_eur_per_kwh.copy()
        ),
    )

    rng_baseline_updates = np.random.default_rng(
        random_seed_baseline_updates
    )

    no_attack_mask = np.zeros(
        number_of_evs,
        dtype=bool,
    )

    consecutive_converged_iterations = 0
    baseline_converged = False
    baseline_completed_iterations = 0

    print(
        "=== Step 18: end-to-end regional attack parameter sweep ==="
    )
    print()
    print("Converging the common no-attack baseline...")

    for iteration_index in range(
        baseline_maximum_iterations
    ):
        selected_indices = (
            rng_baseline_updates.choice(
                number_of_evs,
                size=evs_selected_per_iteration,
                replace=False,
            )
        )

        result = perform_feedback_iteration(
            state=baseline_state,
            selected_ev_indices=selected_indices,
            region_by_ev=region_by_ev,
            aggregate_fixed_demand_mw_by_region=(
                aggregate_fixed_demand_mw_by_region
            ),
            available_hours_by_ev=(
                available_hours_by_ev
            ),
            required_energies_kwh=(
                required_energies_kwh
            ),
            maximum_charging_powers_kw=(
                maximum_charging_powers_kw
            ),
            interval_duration_hours=(
                interval_duration_hours
            ),
            price_damping_factor=(
                price_damping_factor
            ),
            minimum_saving_to_reschedule_eur=(
                minimum_saving_to_reschedule_eur
            ),
            quadratic_coefficient=(
                quadratic_coefficient
            ),
            linear_coefficient=(
                linear_coefficient
            ),
            attack_is_active=False,
            receives_false_price=(
                no_attack_mask
            ),
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
                    "maximum_regional_demand_change_mw"
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
            "The common regional baseline did not converge."
        )

    converged_baseline_state = (
        baseline_state.clone()
    )

    baseline_physical = (
        evaluate_target_hour_power_flow(
            fixed_demand_mw_by_region=(
                aggregate_fixed_demand_mw_by_region
            ),
            total_demand_mw_by_region=(
                converged_baseline_state
                .total_demand_mw_by_region
            ),
            attacked_hour=attacked_hour,
            fixed_load_power_factor=(
                fixed_load_power_factor
            ),
            ev_power_factor=(
                ev_power_factor
            ),
            minimum_voltage_limit_pu=(
                minimum_voltage_limit_pu
            ),
            maximum_voltage_limit_pu=(
                maximum_voltage_limit_pu
            ),
        )
    )

    baseline_region_b_target_demand_mw = float(
        converged_baseline_state
        .total_demand_mw_by_region[
            REGION_B,
            attacked_hour,
        ]
    )

    baseline_total_target_demand_mw = float(
        np.sum(
            converged_baseline_state
            .total_demand_mw_by_region[
                :,
                attacked_hour,
            ]
        )
    )

    print(
        f"Baseline converged after "
        f"{baseline_completed_iterations} iterations."
    )
    print(
        f"Baseline Region-B demand at "
        f"{attacked_hour:02d}:00: "
        f"{baseline_region_b_target_demand_mw:.2f} MW"
    )
    print(
        f"Baseline minimum voltage: "
        f"{float(baseline_physical['minimum_voltage_pu']):.4f} p.u."
    )
    print(
        f"Baseline Line 1 loading: "
        f"{float(baseline_physical['line_1_loading_percent']):.2f}%"
    )

    # Use the same asynchronous update selections in every scenario.
    rng_attack_updates = np.random.default_rng(
        random_seed_attack_updates
    )

    attack_update_plan = [
        rng_attack_updates.choice(
            number_of_evs,
            size=evs_selected_per_iteration,
            replace=False,
        )
        for _ in range(
            maximum_attack_iterations
        )
    ]

    # ============================================================
    # 5. Run the complete three-dimensional sweep
    # ============================================================

    records: list[dict[str, object]] = []

    number_of_scenarios = (
        len(attacked_region_b_ev_fractions)
        * len(false_prices_eur_per_kwh)
        * len(attack_durations_iterations)
    )

    scenario_counter = 0

    print()
    print(
        "Scenario | Fraction | False price | Duration | "
        "Max ΔRegion-B | Min voltage | Line 1 | Violation"
    )
    print("-" * 112)

    for attacked_fraction in (
        attacked_region_b_ev_fractions
    ):
        number_of_attacked_region_b_evs = int(
            round(
                len(region_b_ev_indices)
                * float(attacked_fraction)
            )
        )

        attacked_indices = (
            nested_region_b_attack_order[
                :number_of_attacked_region_b_evs
            ]
        )

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

        for false_price in (
            false_prices_eur_per_kwh
        ):
            # Run one 40-iteration trajectory and take snapshots at
            # 5, 10, 20, and 40 iterations. This keeps scenarios with
            # the same fraction and price directly comparable.
            scenario_state = (
                converged_baseline_state.clone()
            )

            unique_attacked_rescheduled = np.zeros(
                number_of_evs,
                dtype=bool,
            )

            unique_attacked_moved_into_target = np.zeros(
                number_of_evs,
                dtype=bool,
            )

            maximum_region_b_target_demand_mw = (
                baseline_region_b_target_demand_mw
            )

            maximum_total_target_demand_mw = (
                baseline_total_target_demand_mw
            )

            maximum_legitimate_target_price = float(
                converged_baseline_state
                .legitimate_price_eur_per_kwh[
                    attacked_hour
                ]
            )

            maximum_state = (
                converged_baseline_state.clone()
            )

            for iteration_index in range(
                maximum_attack_iterations
            ):
                result = perform_feedback_iteration(
                    state=scenario_state,
                    selected_ev_indices=(
                        attack_update_plan[
                            iteration_index
                        ]
                    ),
                    region_by_ev=region_by_ev,
                    aggregate_fixed_demand_mw_by_region=(
                        aggregate_fixed_demand_mw_by_region
                    ),
                    available_hours_by_ev=(
                        available_hours_by_ev
                    ),
                    required_energies_kwh=(
                        required_energies_kwh
                    ),
                    maximum_charging_powers_kw=(
                        maximum_charging_powers_kw
                    ),
                    interval_duration_hours=(
                        interval_duration_hours
                    ),
                    price_damping_factor=(
                        price_damping_factor
                    ),
                    minimum_saving_to_reschedule_eur=(
                        minimum_saving_to_reschedule_eur
                    ),
                    quadratic_coefficient=(
                        quadratic_coefficient
                    ),
                    linear_coefficient=(
                        linear_coefficient
                    ),
                    attack_is_active=True,
                    receives_false_price=(
                        receives_false_price
                    ),
                    attacked_hour=attacked_hour,
                    false_price_eur_per_kwh=float(
                        false_price
                    ),
                    unique_attacked_rescheduled=(
                        unique_attacked_rescheduled
                    ),
                    unique_attacked_moved_into_target=(
                        unique_attacked_moved_into_target
                    ),
                )

                current_region_b_target_demand_mw = float(
                    result[
                        "region_b_target_demand_mw"
                    ]
                )

                current_total_target_demand_mw = float(
                    result[
                        "total_target_demand_mw"
                    ]
                )

                current_legitimate_target_price = float(
                    result[
                        "legitimate_target_price_eur_per_kwh"
                    ]
                )

                if (
                    current_region_b_target_demand_mw
                    > maximum_region_b_target_demand_mw
                ):
                    maximum_region_b_target_demand_mw = (
                        current_region_b_target_demand_mw
                    )
                    maximum_total_target_demand_mw = (
                        current_total_target_demand_mw
                    )
                    maximum_legitimate_target_price = (
                        current_legitimate_target_price
                    )
                    maximum_state = (
                        scenario_state.clone()
                    )

                completed_iterations = (
                    iteration_index + 1
                )

                if (
                    completed_iterations
                    not in attack_durations_iterations
                ):
                    continue

                scenario_counter += 1

                end_physical = (
                    evaluate_target_hour_power_flow(
                        fixed_demand_mw_by_region=(
                            aggregate_fixed_demand_mw_by_region
                        ),
                        total_demand_mw_by_region=(
                            scenario_state
                            .total_demand_mw_by_region
                        ),
                        attacked_hour=attacked_hour,
                        fixed_load_power_factor=(
                            fixed_load_power_factor
                        ),
                        ev_power_factor=(
                            ev_power_factor
                        ),
                        minimum_voltage_limit_pu=(
                            minimum_voltage_limit_pu
                        ),
                        maximum_voltage_limit_pu=(
                            maximum_voltage_limit_pu
                        ),
                    )
                )

                maximum_physical = (
                    evaluate_target_hour_power_flow(
                        fixed_demand_mw_by_region=(
                            aggregate_fixed_demand_mw_by_region
                        ),
                        total_demand_mw_by_region=(
                            maximum_state
                            .total_demand_mw_by_region
                        ),
                        attacked_hour=attacked_hour,
                        fixed_load_power_factor=(
                            fixed_load_power_factor
                        ),
                        ev_power_factor=(
                            ev_power_factor
                        ),
                        minimum_voltage_limit_pu=(
                            minimum_voltage_limit_pu
                        ),
                        maximum_voltage_limit_pu=(
                            maximum_voltage_limit_pu
                        ),
                    )
                )

                end_region_b_target_demand_mw = float(
                    scenario_state
                    .total_demand_mw_by_region[
                        REGION_B,
                        attacked_hour,
                    ]
                )

                end_total_target_demand_mw = float(
                    np.sum(
                        scenario_state
                        .total_demand_mw_by_region[
                            :,
                            attacked_hour,
                        ]
                    )
                )

                maximum_region_b_increment_mw = (
                    maximum_region_b_target_demand_mw
                    - baseline_region_b_target_demand_mw
                )

                end_region_b_increment_mw = (
                    end_region_b_target_demand_mw
                    - baseline_region_b_target_demand_mw
                )

                unique_rescheduled_count = int(
                    np.sum(
                        unique_attacked_rescheduled
                    )
                )

                unique_moved_into_target_count = int(
                    np.sum(
                        unique_attacked_moved_into_target
                    )
                )

                record = {
                    "attacked_region_b_ev_fraction": float(
                        attacked_fraction
                    ),
                    "false_price_eur_per_kwh": float(
                        false_price
                    ),
                    "attack_duration_iterations": int(
                        completed_iterations
                    ),
                    "attacked_region_b_evs": int(
                        number_of_attacked_region_b_evs
                    ),
                    "attacked_region_b_evs_connected_at_target": int(
                        attacked_connected_at_target
                    ),
                    "unique_attacked_evs_rescheduled": (
                        unique_rescheduled_count
                    ),
                    "unique_attacked_evs_moved_into_target": (
                        unique_moved_into_target_count
                    ),
                    "baseline_region_b_target_demand_mw": (
                        baseline_region_b_target_demand_mw
                    ),
                    "maximum_region_b_target_demand_mw": (
                        maximum_region_b_target_demand_mw
                    ),
                    "maximum_region_b_increment_mw": (
                        maximum_region_b_increment_mw
                    ),
                    "maximum_region_b_increment_percent": (
                        100.0
                        * maximum_region_b_increment_mw
                        / baseline_region_b_target_demand_mw
                    ),
                    "end_region_b_target_demand_mw": (
                        end_region_b_target_demand_mw
                    ),
                    "end_region_b_increment_mw": (
                        end_region_b_increment_mw
                    ),
                    "baseline_total_target_demand_mw": (
                        baseline_total_target_demand_mw
                    ),
                    "maximum_total_target_demand_mw": (
                        maximum_total_target_demand_mw
                    ),
                    "end_total_target_demand_mw": (
                        end_total_target_demand_mw
                    ),
                    "maximum_legitimate_target_price_eur_per_kwh": (
                        maximum_legitimate_target_price
                    ),
                    "baseline_minimum_voltage_pu": float(
                        baseline_physical[
                            "minimum_voltage_pu"
                        ]
                    ),
                    "maximum_minimum_voltage_pu": float(
                        maximum_physical[
                            "minimum_voltage_pu"
                        ]
                    ),
                    "end_minimum_voltage_pu": float(
                        end_physical[
                            "minimum_voltage_pu"
                        ]
                    ),
                    "baseline_line_0_loading_percent": float(
                        baseline_physical[
                            "line_0_loading_percent"
                        ]
                    ),
                    "maximum_line_0_loading_percent": float(
                        maximum_physical[
                            "line_0_loading_percent"
                        ]
                    ),
                    "end_line_0_loading_percent": float(
                        end_physical[
                            "line_0_loading_percent"
                        ]
                    ),
                    "baseline_line_1_loading_percent": float(
                        baseline_physical[
                            "line_1_loading_percent"
                        ]
                    ),
                    "maximum_line_1_loading_percent": float(
                        maximum_physical[
                            "line_1_loading_percent"
                        ]
                    ),
                    "end_line_1_loading_percent": float(
                        end_physical[
                            "line_1_loading_percent"
                        ]
                    ),
                    "baseline_line_losses_mw": float(
                        baseline_physical[
                            "total_line_losses_mw"
                        ]
                    ),
                    "maximum_line_losses_mw": float(
                        maximum_physical[
                            "total_line_losses_mw"
                        ]
                    ),
                    "end_line_losses_mw": float(
                        end_physical[
                            "total_line_losses_mw"
                        ]
                    ),
                    "maximum_voltage_violation": bool(
                        maximum_physical[
                            "voltage_violation"
                        ]
                    ),
                    "maximum_line_overload": bool(
                        maximum_physical[
                            "line_overload"
                        ]
                    ),
                    "maximum_power_flow_converged": bool(
                        maximum_physical[
                            "power_flow_converged"
                        ]
                    ),
                    "end_voltage_violation": bool(
                        end_physical[
                            "voltage_violation"
                        ]
                    ),
                    "end_line_overload": bool(
                        end_physical[
                            "line_overload"
                        ]
                    ),
                    "end_power_flow_converged": bool(
                        end_physical[
                            "power_flow_converged"
                        ]
                    ),
                }

                records.append(record)

                violation_text = (
                    "V"
                    if record[
                        "maximum_voltage_violation"
                    ]
                    else "-"
                )

                overload_text = (
                    "L"
                    if record[
                        "maximum_line_overload"
                    ]
                    else "-"
                )

                print(
                    f"{scenario_counter:3d}/"
                    f"{number_of_scenarios:<3d} | "
                    f"{attacked_fraction:7.0%} | "
                    f"€{false_price:10.2f} | "
                    f"{completed_iterations:8d} | "
                    f"{maximum_region_b_increment_mw:12.2f} MW | "
                    f"{float(maximum_physical['minimum_voltage_pu']):11.4f} | "
                    f"{float(maximum_physical['line_1_loading_percent']):6.2f}% | "
                    f"{violation_text}{overload_text}"
                )

    results = pd.DataFrame(
        records
    )

    duration_threshold_summary = (
        build_duration_threshold_summary(
            results=results,
            attacked_fractions=(
                attacked_region_b_ev_fractions
            ),
            false_prices_eur_per_kwh=(
                false_prices_eur_per_kwh
            ),
        )
    )

    # ============================================================
    # 6. Save numerical results
    # ============================================================

    results.to_csv(
        output_directory
        / "step_18_end_to_end_sweep_results.csv",
        index=False,
    )

    duration_threshold_summary.to_csv(
        output_directory
        / "step_18_duration_threshold_summary.csv",
        index=False,
    )

    configuration = {
        "number_of_households": (
            number_of_households
        ),
        "households_per_region": (
            households_per_region
        ),
        "number_of_evs": number_of_evs,
        "region_b_evs": int(
            len(region_b_ev_indices)
        ),
        "attacked_region_b_ev_fractions": (
            attacked_region_b_ev_fractions.tolist()
        ),
        "false_prices_eur_per_kwh": (
            false_prices_eur_per_kwh.tolist()
        ),
        "attack_durations_iterations": (
            attack_durations_iterations.tolist()
        ),
        "attacked_hour": attacked_hour,
        "price_damping_factor": (
            price_damping_factor
        ),
        "ev_rescheduling_fraction": (
            ev_rescheduling_fraction
        ),
        "minimum_saving_to_reschedule_eur": (
            minimum_saving_to_reschedule_eur
        ),
        "fixed_load_power_factor": (
            fixed_load_power_factor
        ),
        "ev_power_factor": ev_power_factor,
        "minimum_voltage_limit_pu": (
            minimum_voltage_limit_pu
        ),
        "maximum_voltage_limit_pu": (
            maximum_voltage_limit_pu
        ),
        "baseline_completed_iterations": (
            baseline_completed_iterations
        ),
        "baseline_region_b_target_demand_mw": (
            baseline_region_b_target_demand_mw
        ),
        "baseline_total_target_demand_mw": (
            baseline_total_target_demand_mw
        ),
        "baseline_minimum_voltage_pu": float(
            baseline_physical[
                "minimum_voltage_pu"
            ]
        ),
        "baseline_line_0_loading_percent": float(
            baseline_physical[
                "line_0_loading_percent"
            ]
        ),
        "baseline_line_1_loading_percent": float(
            baseline_physical[
                "line_1_loading_percent"
            ]
        ),
        "random_seeds": {
            "population": (
                random_seed_population
            ),
            "region_assignment": (
                random_seed_region_assignment
            ),
            "baseline_updates": (
                random_seed_baseline_updates
            ),
            "attack_updates": (
                random_seed_attack_updates
            ),
            "attack_order": (
                random_seed_attack_order
            ),
        },
    }

    (
        output_directory
        / "step_18_end_to_end_sweep_config.json"
    ).write_text(
        json.dumps(
            configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # 7. Save heatmaps for each tested duration
    # ============================================================

    for attack_duration in (
        attack_durations_iterations
    ):
        duration = int(
            attack_duration
        )

        demand_matrix = build_metric_matrix(
            results=results,
            attacked_fractions=(
                attacked_region_b_ev_fractions
            ),
            false_prices_eur_per_kwh=(
                false_prices_eur_per_kwh
            ),
            attack_duration_iterations=(
                duration
            ),
            metric_name=(
                "maximum_region_b_increment_mw"
            ),
        )

        voltage_matrix = build_metric_matrix(
            results=results,
            attacked_fractions=(
                attacked_region_b_ev_fractions
            ),
            false_prices_eur_per_kwh=(
                false_prices_eur_per_kwh
            ),
            attack_duration_iterations=(
                duration
            ),
            metric_name=(
                "maximum_minimum_voltage_pu"
            ),
        )

        line_1_matrix = build_metric_matrix(
            results=results,
            attacked_fractions=(
                attacked_region_b_ev_fractions
            ),
            false_prices_eur_per_kwh=(
                false_prices_eur_per_kwh
            ),
            attack_duration_iterations=(
                duration
            ),
            metric_name=(
                "maximum_line_1_loading_percent"
            ),
        )

        losses_matrix = build_metric_matrix(
            results=results,
            attacked_fractions=(
                attacked_region_b_ev_fractions
            ),
            false_prices_eur_per_kwh=(
                false_prices_eur_per_kwh
            ),
            attack_duration_iterations=(
                duration
            ),
            metric_name=(
                "maximum_line_losses_mw"
            ),
        )

        save_heatmap(
            matrix=demand_matrix,
            attacked_fractions=(
                attacked_region_b_ev_fractions
            ),
            false_prices_eur_per_kwh=(
                false_prices_eur_per_kwh
            ),
            title=(
                "Maximum Region-B Demand Increase "
                f"after {duration} Attack Iterations"
            ),
            colorbar_label="Demand increase (MW)",
            output_path=(
                output_directory
                / (
                    "step_18_heatmap_demand_"
                    f"{duration}_iterations.png"
                )
            ),
            value_format=".2f",
        )

        save_heatmap(
            matrix=voltage_matrix,
            attacked_fractions=(
                attacked_region_b_ev_fractions
            ),
            false_prices_eur_per_kwh=(
                false_prices_eur_per_kwh
            ),
            title=(
                "Minimum Voltage "
                f"after {duration} Attack Iterations"
            ),
            colorbar_label="Minimum voltage (p.u.)",
            output_path=(
                output_directory
                / (
                    "step_18_heatmap_voltage_"
                    f"{duration}_iterations.png"
                )
            ),
            value_format=".4f",
        )

        save_heatmap(
            matrix=line_1_matrix,
            attacked_fractions=(
                attacked_region_b_ev_fractions
            ),
            false_prices_eur_per_kwh=(
                false_prices_eur_per_kwh
            ),
            title=(
                "Maximum Line 1 Loading "
                f"after {duration} Attack Iterations"
            ),
            colorbar_label="Line 1 loading (%)",
            output_path=(
                output_directory
                / (
                    "step_18_heatmap_line_1_"
                    f"{duration}_iterations.png"
                )
            ),
            value_format=".1f",
        )

        save_heatmap(
            matrix=losses_matrix,
            attacked_fractions=(
                attacked_region_b_ev_fractions
            ),
            false_prices_eur_per_kwh=(
                false_prices_eur_per_kwh
            ),
            title=(
                "Maximum Active Line Losses "
                f"after {duration} Attack Iterations"
            ),
            colorbar_label="Active losses (MW)",
            output_path=(
                output_directory
                / (
                    "step_18_heatmap_losses_"
                    f"{duration}_iterations.png"
                )
            ),
            value_format=".3f",
        )

    # ============================================================
    # 8. Save duration curves for representative attacks
    # ============================================================

    plt.figure(figsize=(10, 5))

    for attacked_fraction in (
        attacked_region_b_ev_fractions
    ):
        matching_results = results[
            np.isclose(
                results[
                    "attacked_region_b_ev_fraction"
                ],
                attacked_fraction,
            )
            & np.isclose(
                results[
                    "false_price_eur_per_kwh"
                ],
                0.05,
            )
        ].sort_values(
            "attack_duration_iterations"
        )

        plt.plot(
            matching_results[
                "attack_duration_iterations"
            ],
            matching_results[
                "maximum_region_b_increment_mw"
            ],
            marker="o",
            label=(
                f"{attacked_fraction:.0%} attacked"
            ),
        )

    plt.xlabel("Attack duration (feedback iterations)")
    plt.ylabel(
        "Maximum Region-B demand increase (MW)"
    )
    plt.title(
        "Demand Impact versus Attack Duration "
        "for a €0.05/kWh Forged Price"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_18_duration_demand_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(figsize=(10, 5))

    for attacked_fraction in (
        attacked_region_b_ev_fractions
    ):
        matching_results = results[
            np.isclose(
                results[
                    "attacked_region_b_ev_fraction"
                ],
                attacked_fraction,
            )
            & np.isclose(
                results[
                    "false_price_eur_per_kwh"
                ],
                0.05,
            )
        ].sort_values(
            "attack_duration_iterations"
        )

        plt.plot(
            matching_results[
                "attack_duration_iterations"
            ],
            matching_results[
                "maximum_line_1_loading_percent"
            ],
            marker="o",
            label=(
                f"{attacked_fraction:.0%} attacked"
            ),
        )

    plt.axhline(
        100.0,
        linestyle="--",
        label="Thermal limit",
    )

    plt.xlabel("Attack duration (feedback iterations)")
    plt.ylabel("Maximum Line 1 loading (%)")
    plt.title(
        "Physical Impact versus Attack Duration "
        "for a €0.05/kWh Forged Price"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_18_duration_line_1_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Files saved:")
    print(
        "  results/"
        "step_18_end_to_end_sweep_results.csv"
    )
    print(
        "  results/"
        "step_18_duration_threshold_summary.csv"
    )
    print(
        "  results/"
        "step_18_end_to_end_sweep_config.json"
    )
    print(
        "  results/"
        "step_18_heatmap_*.png"
    )
    print(
        "  results/"
        "step_18_duration_demand_curves.png"
    )
    print(
        "  results/"
        "step_18_duration_line_1_curves.png"
    )


if __name__ == "__main__":
    main()
