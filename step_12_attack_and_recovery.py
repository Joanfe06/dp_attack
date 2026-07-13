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


def calculate_schedule_cost_eur(
    schedule_kw: np.ndarray,
    price_eur_per_kwh: np.ndarray,
    interval_duration_hours: float,
) -> float:
    """Calculate the cost of one EV charging schedule."""
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
    """Calculate marginal price from aggregate demand."""
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
    """Calculate total generation cost over the simulated day."""
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


def main() -> None:
    hours = np.arange(24)
    interval_duration_hours = 1.0

    # ============================================================
    # 1. Simulation parameters
    # ============================================================

    random_seed = 42
    rng = np.random.default_rng(random_seed)

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

    # False-price attack parameters.
    attacked_hour = 18
    false_price_eur_per_kwh = 0.05
    attacked_ev_fraction = 0.25

    # Generation-cost model:
    #
    # C(D) = aD² + bD
    #
    # Marginal price:
    #
    # p(D) = 2aD + b
    quadratic_coefficient = 6.190476
    linear_coefficient = 28.571432

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

    # Initial price is based on fixed demand only.
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

    number_of_evs = int(
        round(
            number_of_households
            * ev_adoption_rate
        )
    )

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

    # Select exactly 25% of EVs as attack recipients.
    number_of_attacked_evs = int(
        round(
            number_of_evs
            * attacked_ev_fraction
        )
    )

    attacked_ev_indices = rng.choice(
        number_of_evs,
        size=number_of_attacked_evs,
        replace=False,
    )

    receives_false_price = np.zeros(
        number_of_evs,
        dtype=bool,
    )

    receives_false_price[
        attacked_ev_indices
    ] = True

    attacked_and_connected_at_target = int(
        np.sum(
            [
                receives_false_price[
                    ev_index
                ]
                and attacked_hour
                in available_hours_by_ev[
                    ev_index
                ]
                for ev_index in range(
                    number_of_evs
                )
            ]
        )
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
    # 4. Initial EV schedules
    # ============================================================

    current_ev_schedules_kw = np.zeros(
        (number_of_evs, 24),
        dtype=float,
    )

    for ev_index in range(number_of_evs):
        current_ev_schedules_kw[
            ev_index
        ] = create_ev_schedule(
            price_eur_per_kwh=(
                initial_price_eur_per_kwh
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

    aggregate_ev_demand_kw = np.sum(
        current_ev_schedules_kw,
        axis=0,
    )

    total_demand_mw = (
        aggregate_fixed_demand_mw
        + kw_to_mw(
            aggregate_ev_demand_kw
        )
    )

    current_price_eur_per_kwh = (
        initial_price_eur_per_kwh.copy()
    )

    # ============================================================
    # 5. Shared histories
    # ============================================================

    phase_history: list[str] = []
    phase_iteration_history: list[int] = []

    demand_at_target_history_mw: list[float] = []
    legitimate_price_at_target_history: list[float] = []
    accepted_changes_history: list[int] = []
    attacked_accepted_changes_history: list[int] = []
    maximum_demand_change_history_mw: list[float] = []
    applied_price_change_history: list[float] = []
    generation_cost_history_eur: list[float] = []

    # Save representative demand profiles.
    baseline_demand_mw: np.ndarray | None = None
    maximum_attack_demand_mw: np.ndarray | None = None
    recovered_demand_mw: np.ndarray | None = None

    maximum_attack_target_demand_mw = -np.inf

    # ============================================================
    # 6. One feedback iteration
    # ============================================================

    def perform_feedback_iteration(
        attack_is_active: bool,
    ) -> dict[str, float | int | np.ndarray]:
        nonlocal aggregate_ev_demand_kw
        nonlocal total_demand_mw
        nonlocal current_price_eur_per_kwh

        previous_total_demand_mw = (
            total_demand_mw.copy()
        )

        selected_ev_indices = rng.choice(
            number_of_evs,
            size=evs_selected_per_iteration,
            replace=False,
        )

        candidate_changes = 0
        accepted_changes = 0
        attacked_accepted_changes = 0

        for ev_index in selected_ev_indices:
            previous_schedule_kw = (
                current_ev_schedules_kw[
                    ev_index
                ].copy()
            )

            perceived_price_eur_per_kwh = (
                current_price_eur_per_kwh.copy()
            )

            if (
                attack_is_active
                and receives_false_price[
                    ev_index
                ]
            ):
                perceived_price_eur_per_kwh[
                    attacked_hour
                ] = false_price_eur_per_kwh

            candidate_schedule_kw = (
                create_ev_schedule(
                    price_eur_per_kwh=(
                        perceived_price_eur_per_kwh
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
            )

            schedule_is_different = (
                not np.allclose(
                    previous_schedule_kw,
                    candidate_schedule_kw,
                    atol=1e-12,
                )
            )

            if not schedule_is_different:
                continue

            candidate_changes += 1

            current_schedule_cost_eur = (
                calculate_schedule_cost_eur(
                    schedule_kw=(
                        previous_schedule_kw
                    ),
                    price_eur_per_kwh=(
                        perceived_price_eur_per_kwh
                    ),
                    interval_duration_hours=(
                        interval_duration_hours
                    ),
                )
            )

            candidate_schedule_cost_eur = (
                calculate_schedule_cost_eur(
                    schedule_kw=(
                        candidate_schedule_kw
                    ),
                    price_eur_per_kwh=(
                        perceived_price_eur_per_kwh
                    ),
                    interval_duration_hours=(
                        interval_duration_hours
                    ),
                )
            )

            expected_saving_eur = (
                current_schedule_cost_eur
                - candidate_schedule_cost_eur
            )

            if (
                expected_saving_eur
                + 1e-12
                < minimum_saving_to_reschedule_eur
            ):
                continue

            accepted_changes += 1

            if receives_false_price[
                ev_index
            ]:
                attacked_accepted_changes += 1

            aggregate_ev_demand_kw -= (
                previous_schedule_kw
            )

            aggregate_ev_demand_kw += (
                candidate_schedule_kw
            )

            current_ev_schedules_kw[
                ev_index
            ] = candidate_schedule_kw

        aggregate_ev_demand_mw = kw_to_mw(
            aggregate_ev_demand_kw
        )

        total_demand_mw = (
            aggregate_fixed_demand_mw
            + aggregate_ev_demand_mw
        )

        raw_price_eur_per_kwh = (
            calculate_price_eur_per_kwh(
                demand_mw=total_demand_mw,
                quadratic_coefficient=(
                    quadratic_coefficient
                ),
                linear_coefficient=(
                    linear_coefficient
                ),
            )
        )

        next_price_eur_per_kwh = (
            (1.0 - price_damping_factor)
            * current_price_eur_per_kwh
            + price_damping_factor
            * raw_price_eur_per_kwh
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
                "EV energy changed during "
                "the simulation."
            )

        daily_generation_cost_eur = (
            calculate_daily_generation_cost_eur(
                demand_mw=total_demand_mw,
                quadratic_coefficient=(
                    quadratic_coefficient
                ),
                linear_coefficient=(
                    linear_coefficient
                ),
                interval_duration_hours=(
                    interval_duration_hours
                ),
            )
        )

        current_price_eur_per_kwh = (
            next_price_eur_per_kwh
        )

        return {
            "candidate_changes": (
                candidate_changes
            ),
            "accepted_changes": (
                accepted_changes
            ),
            "attacked_accepted_changes": (
                attacked_accepted_changes
            ),
            "applied_price_change": (
                applied_price_change
            ),
            "maximum_demand_change_mw": (
                maximum_demand_change_mw
            ),
            "demand_at_target_mw": float(
                total_demand_mw[
                    attacked_hour
                ]
            ),
            "legitimate_price_at_target": float(
                current_price_eur_per_kwh[
                    attacked_hour
                ]
            ),
            "daily_generation_cost_eur": (
                daily_generation_cost_eur
            ),
            "total_demand_mw": (
                total_demand_mw.copy()
            ),
        }

    # ============================================================
    # 7. Phase A: converge the no-attack baseline
    # ============================================================

    print(
        "=== False-price attack after "
        "baseline convergence ==="
    )
    print()
    print("Configuration")
    print(
        f"  Households: "
        f"{number_of_households:,}"
    )
    print(f"  EVs: {number_of_evs:,}")
    print(
        f"  Price damping factor: "
        f"{price_damping_factor:.2f}"
    )
    print(
        f"  EVs rescheduled per iteration: "
        f"{evs_selected_per_iteration:,}"
    )
    print(
        f"  Rescheduling hysteresis: "
        f"€{minimum_saving_to_reschedule_eur:.2f}"
    )
    print(
        f"  Attacked EVs: "
        f"{number_of_attacked_evs:,}"
    )
    print(
        f"  Attacked EVs connected at "
        f"{attacked_hour:02d}:00: "
        f"{attacked_and_connected_at_target:,}"
    )
    print(
        f"  False price at "
        f"{attacked_hour:02d}:00: "
        f"€{false_price_eur_per_kwh:.2f}/kWh"
    )

    print()
    print("Phase A: baseline convergence")
    print(
        "Iter | Accepted | Demand at 18 | "
        "Price at 18 | Applied Δprice | Max Δdemand"
    )
    print("-" * 80)

    consecutive_converged_iterations = 0
    baseline_converged = False
    baseline_completed_iterations = 0

    for iteration_index in range(
        baseline_maximum_iterations
    ):
        result = perform_feedback_iteration(
            attack_is_active=False
        )

        phase_history.append("Baseline")
        phase_iteration_history.append(
            iteration_index + 1
        )
        demand_at_target_history_mw.append(
            float(
                result[
                    "demand_at_target_mw"
                ]
            )
        )
        legitimate_price_at_target_history.append(
            float(
                result[
                    "legitimate_price_at_target"
                ]
            )
        )
        accepted_changes_history.append(
            int(
                result[
                    "accepted_changes"
                ]
            )
        )
        attacked_accepted_changes_history.append(
            0
        )
        maximum_demand_change_history_mw.append(
            float(
                result[
                    "maximum_demand_change_mw"
                ]
            )
        )
        applied_price_change_history.append(
            float(
                result[
                    "applied_price_change"
                ]
            )
        )
        generation_cost_history_eur.append(
            float(
                result[
                    "daily_generation_cost_eur"
                ]
            )
        )

        print(
            f"{iteration_index + 1:4d} | "
            f"{int(result['accepted_changes']):8d} | "
            f"{float(result['demand_at_target_mw']):12.2f} | "
            f"{float(result['legitimate_price_at_target']):11.3f} | "
            f"{float(result['applied_price_change']):14.6f} | "
            f"{float(result['maximum_demand_change_mw']):12.6f}"
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

        baseline_completed_iterations = (
            iteration_index + 1
        )

        if (
            consecutive_converged_iterations
            >= convergence_patience
        ):
            baseline_converged = True
            break

    baseline_demand_mw = total_demand_mw.copy()
    baseline_price_eur_per_kwh = (
        current_price_eur_per_kwh.copy()
    )
    baseline_generation_cost_eur = (
        calculate_daily_generation_cost_eur(
            demand_mw=baseline_demand_mw,
            quadratic_coefficient=(
                quadratic_coefficient
            ),
            linear_coefficient=(
                linear_coefficient
            ),
            interval_duration_hours=(
                interval_duration_hours
            ),
        )
    )

    # ============================================================
    # 8. Phase B: activate the false-price attack
    # ============================================================

    print()
    print("Phase B: false-price attack")
    print(
        "Iter | Accepted | Attacked accepted | "
        "Demand at 18 | Legitimate price at 18 | "
        "Max Δdemand"
    )
    print("-" * 100)

    for iteration_index in range(
        attack_iterations
    ):
        result = perform_feedback_iteration(
            attack_is_active=True
        )

        phase_history.append("Attack")
        phase_iteration_history.append(
            iteration_index + 1
        )
        demand_at_target_history_mw.append(
            float(
                result[
                    "demand_at_target_mw"
                ]
            )
        )
        legitimate_price_at_target_history.append(
            float(
                result[
                    "legitimate_price_at_target"
                ]
            )
        )
        accepted_changes_history.append(
            int(
                result[
                    "accepted_changes"
                ]
            )
        )
        attacked_accepted_changes_history.append(
            int(
                result[
                    "attacked_accepted_changes"
                ]
            )
        )
        maximum_demand_change_history_mw.append(
            float(
                result[
                    "maximum_demand_change_mw"
                ]
            )
        )
        applied_price_change_history.append(
            float(
                result[
                    "applied_price_change"
                ]
            )
        )
        generation_cost_history_eur.append(
            float(
                result[
                    "daily_generation_cost_eur"
                ]
            )
        )

        current_target_demand_mw = float(
            result[
                "demand_at_target_mw"
            ]
        )

        if (
            current_target_demand_mw
            > maximum_attack_target_demand_mw
        ):
            maximum_attack_target_demand_mw = (
                current_target_demand_mw
            )

            maximum_attack_demand_mw = (
                np.asarray(
                    result[
                        "total_demand_mw"
                    ]
                ).copy()
            )

        print(
            f"{iteration_index + 1:4d} | "
            f"{int(result['accepted_changes']):8d} | "
            f"{int(result['attacked_accepted_changes']):17d} | "
            f"{current_target_demand_mw:12.2f} | "
            f"{float(result['legitimate_price_at_target']):22.3f} | "
            f"{float(result['maximum_demand_change_mw']):12.4f}"
        )

    attack_end_demand_mw = (
        total_demand_mw.copy()
    )
    attack_end_price_eur_per_kwh = (
        current_price_eur_per_kwh.copy()
    )
    attack_end_generation_cost_eur = (
        calculate_daily_generation_cost_eur(
            demand_mw=attack_end_demand_mw,
            quadratic_coefficient=(
                quadratic_coefficient
            ),
            linear_coefficient=(
                linear_coefficient
            ),
            interval_duration_hours=(
                interval_duration_hours
            ),
        )
    )

    # ============================================================
    # 9. Phase C: stop the attack and observe recovery
    # ============================================================

    print()
    print("Phase C: recovery after attack")
    print(
        "Iter | Accepted | Demand at 18 | "
        "Price at 18 | Applied Δprice | Max Δdemand"
    )
    print("-" * 80)

    consecutive_converged_iterations = 0
    recovery_converged = False
    recovery_completed_iterations = 0

    for iteration_index in range(
        recovery_maximum_iterations
    ):
        result = perform_feedback_iteration(
            attack_is_active=False
        )

        phase_history.append("Recovery")
        phase_iteration_history.append(
            iteration_index + 1
        )
        demand_at_target_history_mw.append(
            float(
                result[
                    "demand_at_target_mw"
                ]
            )
        )
        legitimate_price_at_target_history.append(
            float(
                result[
                    "legitimate_price_at_target"
                ]
            )
        )
        accepted_changes_history.append(
            int(
                result[
                    "accepted_changes"
                ]
            )
        )
        attacked_accepted_changes_history.append(
            0
        )
        maximum_demand_change_history_mw.append(
            float(
                result[
                    "maximum_demand_change_mw"
                ]
            )
        )
        applied_price_change_history.append(
            float(
                result[
                    "applied_price_change"
                ]
            )
        )
        generation_cost_history_eur.append(
            float(
                result[
                    "daily_generation_cost_eur"
                ]
            )
        )

        print(
            f"{iteration_index + 1:4d} | "
            f"{int(result['accepted_changes']):8d} | "
            f"{float(result['demand_at_target_mw']):12.2f} | "
            f"{float(result['legitimate_price_at_target']):11.3f} | "
            f"{float(result['applied_price_change']):14.6f} | "
            f"{float(result['maximum_demand_change_mw']):12.6f}"
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

        recovery_completed_iterations = (
            iteration_index + 1
        )

        if (
            consecutive_converged_iterations
            >= convergence_patience
        ):
            recovery_converged = True
            break

    recovered_demand_mw = (
        total_demand_mw.copy()
    )
    recovered_price_eur_per_kwh = (
        current_price_eur_per_kwh.copy()
    )
    recovered_generation_cost_eur = (
        calculate_daily_generation_cost_eur(
            demand_mw=recovered_demand_mw,
            quadratic_coefficient=(
                quadratic_coefficient
            ),
            linear_coefficient=(
                linear_coefficient
            ),
            interval_duration_hours=(
                interval_duration_hours
            ),
        )
    )

    # ============================================================
    # 10. Final metrics
    # ============================================================

    baseline_target_demand_mw = float(
        baseline_demand_mw[
            attacked_hour
        ]
    )

    attack_end_target_demand_mw = float(
        attack_end_demand_mw[
            attacked_hour
        ]
    )

    recovered_target_demand_mw = float(
        recovered_demand_mw[
            attacked_hour
        ]
    )

    attack_increase_at_target_mw = (
        attack_end_target_demand_mw
        - baseline_target_demand_mw
    )

    attack_increase_at_target_percent = (
        100.0
        * attack_increase_at_target_mw
        / baseline_target_demand_mw
    )

    attack_cost_increase_eur = (
        attack_end_generation_cost_eur
        - baseline_generation_cost_eur
    )

    maximum_attack_increase_mw = (
        maximum_attack_target_demand_mw
        - baseline_target_demand_mw
    )

    recovery_distance_from_baseline_mw = float(
        np.max(
            np.abs(
                recovered_demand_mw
                - baseline_demand_mw
            )
        )
    )

    print()
    print("=== Summary ===")
    print(
        f"Baseline converged: "
        f"{baseline_converged} "
        f"after "
        f"{baseline_completed_iterations} "
        f"iterations"
    )
    print(
        f"Recovery converged: "
        f"{recovery_converged} "
        f"after "
        f"{recovery_completed_iterations} "
        f"iterations"
    )
    print()
    print(
        f"Baseline demand at "
        f"{attacked_hour:02d}:00: "
        f"{baseline_target_demand_mw:.2f} MW"
    )
    print(
        f"Maximum demand during attack at "
        f"{attacked_hour:02d}:00: "
        f"{maximum_attack_target_demand_mw:.2f} MW"
    )
    print(
        f"Demand at attack end at "
        f"{attacked_hour:02d}:00: "
        f"{attack_end_target_demand_mw:.2f} MW"
    )
    print(
        f"Demand after recovery at "
        f"{attacked_hour:02d}:00: "
        f"{recovered_target_demand_mw:.2f} MW"
    )
    print(
        f"Attack-end increase: "
        f"{attack_increase_at_target_mw:.2f} MW "
        f"({attack_increase_at_target_percent:.2f}%)"
    )
    print(
        f"Maximum attack increase: "
        f"{maximum_attack_increase_mw:.2f} MW"
    )
    print()
    print(
        f"Baseline legitimate price at "
        f"{attacked_hour:02d}:00: "
        f"€{baseline_price_eur_per_kwh[attacked_hour]:.3f}/kWh"
    )
    print(
        f"Legitimate price at attack end: "
        f"€{attack_end_price_eur_per_kwh[attacked_hour]:.3f}/kWh"
    )
    print(
        f"False price shown to attacked EVs: "
        f"€{false_price_eur_per_kwh:.3f}/kWh"
    )
    print()
    print(
        f"Baseline daily generation cost: "
        f"€{baseline_generation_cost_eur:.2f}"
    )
    print(
        f"Daily generation cost at attack end: "
        f"€{attack_end_generation_cost_eur:.2f}"
    )
    print(
        f"Attack-end cost increase: "
        f"€{attack_cost_increase_eur:.2f}"
    )
    print(
        f"Recovered daily generation cost: "
        f"€{recovered_generation_cost_eur:.2f}"
    )
    print(
        f"Maximum recovered-demand distance "
        f"from baseline: "
        f"{recovery_distance_from_baseline_mw:.4f} MW"
    )

    # ============================================================
    # 11. Figures
    # ============================================================

    output_directory = Path("results")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if maximum_attack_demand_mw is None:
        maximum_attack_demand_mw = (
            attack_end_demand_mw.copy()
        )

    # Demand profiles.
    plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        baseline_demand_mw,
        marker="o",
        label="Converged baseline",
    )

    plt.plot(
        hours,
        maximum_attack_demand_mw,
        marker="o",
        label="Maximum attack impact",
    )

    plt.plot(
        hours,
        recovered_demand_mw,
        marker="o",
        label="Recovered demand",
    )

    plt.xlabel("Hour")
    plt.ylabel("Total demand (MW)")
    plt.title(
        "Baseline, False-Price Attack, and Recovery"
    )
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_12_demand_profiles.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Build a continuous x-axis across all phases.
    global_iterations = np.arange(
        1,
        len(phase_history) + 1,
    )

    baseline_end = (
        baseline_completed_iterations
    )
    attack_end = (
        baseline_completed_iterations
        + attack_iterations
    )

    # Demand at the attacked hour.
    plt.figure(figsize=(10, 5))

    plt.plot(
        global_iterations,
        demand_at_target_history_mw,
        marker="o",
    )

    plt.axvline(
        baseline_end + 0.5,
        linestyle="--",
        label="Attack starts",
    )

    plt.axvline(
        attack_end + 0.5,
        linestyle="--",
        label="Attack ends",
    )

    plt.xlabel("Feedback iteration")
    plt.ylabel(
        f"Demand at {attacked_hour:02d}:00 (MW)"
    )
    plt.title(
        "Demand Response to the False-Price Attack"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_12_target_hour_demand.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Legitimate system price at the attacked hour.
    plt.figure(figsize=(10, 5))

    plt.plot(
        global_iterations,
        legitimate_price_at_target_history,
        marker="o",
        label="Legitimate marginal price",
    )

    plt.axhline(
        false_price_eur_per_kwh,
        linestyle="--",
        label="False price seen by attacked EVs",
    )

    plt.axvline(
        baseline_end + 0.5,
        linestyle="--",
        label="Attack starts",
    )

    plt.axvline(
        attack_end + 0.5,
        linestyle="--",
        label="Attack ends",
    )

    plt.xlabel("Feedback iteration")
    plt.ylabel(
        f"Price at {attacked_hour:02d}:00 (€/kWh)"
    )
    plt.title(
        "Legitimate and Forged Prices During the Attack"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_12_target_hour_price.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Accepted schedule changes.
    plt.figure(figsize=(10, 5))

    plt.plot(
        global_iterations,
        accepted_changes_history,
        marker="o",
        label="All accepted changes",
    )

    plt.plot(
        global_iterations,
        attacked_accepted_changes_history,
        marker="o",
        label="Accepted changes by attacked EVs",
    )

    plt.axvline(
        baseline_end + 0.5,
        linestyle="--",
        label="Attack starts",
    )

    plt.axvline(
        attack_end + 0.5,
        linestyle="--",
        label="Attack ends",
    )

    plt.xlabel("Feedback iteration")
    plt.ylabel("Accepted EV schedule changes")
    plt.title(
        "Schedule Changes Across Baseline, Attack, and Recovery"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_12_schedule_changes.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Daily generation cost.
    plt.figure(figsize=(10, 5))

    plt.plot(
        global_iterations,
        generation_cost_history_eur,
        marker="o",
    )

    plt.axvline(
        baseline_end + 0.5,
        linestyle="--",
        label="Attack starts",
    )

    plt.axvline(
        attack_end + 0.5,
        linestyle="--",
        label="Attack ends",
    )

    plt.xlabel("Feedback iteration")
    plt.ylabel("Daily generation cost (€)")
    plt.title(
        "Generation-Cost Effect of the False-Price Attack"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_12_generation_cost.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Maximum demand change.
    plt.figure(figsize=(10, 5))

    plt.plot(
        global_iterations,
        maximum_demand_change_history_mw,
        marker="o",
    )

    plt.axvline(
        baseline_end + 0.5,
        linestyle="--",
        label="Attack starts",
    )

    plt.axvline(
        attack_end + 0.5,
        linestyle="--",
        label="Attack ends",
    )

    plt.xlabel("Feedback iteration")
    plt.ylabel("Maximum demand change (MW)")
    plt.title(
        "Demand Disturbance and Recovery"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_12_demand_disturbance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Figures saved:")
    print(
        "  results/"
        "step_12_demand_profiles.png"
    )
    print(
        "  results/"
        "step_12_target_hour_demand.png"
    )
    print(
        "  results/"
        "step_12_target_hour_price.png"
    )
    print(
        "  results/"
        "step_12_schedule_changes.png"
    )
    print(
        "  results/"
        "step_12_generation_cost.png"
    )
    print(
        "  results/"
        "step_12_demand_disturbance.png"
    )


if __name__ == "__main__":
    main()
