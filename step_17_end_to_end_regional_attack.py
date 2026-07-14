from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Use a non-interactive backend because the script runs without a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import pandapower as pp
    from pandapower.powerflow import LoadflowNotConverged
except ImportError as exc:
    raise SystemExit(
        "pandapower is not installed in the active virtual environment.\n"
        "Install it with:\n"
        "    pip install pandapower\n"
    ) from exc

from step_02_ev_charging import (
    create_ev_schedule,
    get_available_hours,
)

from step_06_generate_dynamic_price import (
    calculate_marginal_price_eur_per_mwh,
    eur_per_mwh_to_eur_per_kwh,
)


REGION_A = 0
REGION_B = 1
NUMBER_OF_REGIONS = 2


@dataclass
class FeedbackState:
    """Mutable regional demand-response state."""

    ev_schedules_kw: np.ndarray
    aggregate_ev_demand_kw_by_region: np.ndarray
    total_demand_mw_by_region: np.ndarray
    legitimate_price_eur_per_kwh: np.ndarray

    def clone(self) -> "FeedbackState":
        """Return a deep copy for an independent scenario."""
        return FeedbackState(
            ev_schedules_kw=self.ev_schedules_kw.copy(),
            aggregate_ev_demand_kw_by_region=(
                self.aggregate_ev_demand_kw_by_region.copy()
            ),
            total_demand_mw_by_region=(
                self.total_demand_mw_by_region.copy()
            ),
            legitimate_price_eur_per_kwh=(
                self.legitimate_price_eur_per_kwh.copy()
            ),
        )


def kw_to_mw(
    power_kw: np.ndarray | float,
) -> np.ndarray | float:
    """Convert power from kW to MW."""
    return power_kw / 1000.0


def reactive_power_from_power_factor(
    active_power_mw: float,
    power_factor: float,
) -> float:
    """Calculate inductive reactive power from active power and power factor."""
    if active_power_mw < 0:
        raise ValueError("Active power cannot be negative.")

    if not 0 < power_factor <= 1:
        raise ValueError(
            "Power factor must be greater than 0 and no greater than 1."
        )

    angle_rad = np.arccos(power_factor)

    return float(
        active_power_mw * np.tan(angle_rad)
    )


def calculate_schedule_cost_eur(
    schedule_kw: np.ndarray,
    price_eur_per_kwh: np.ndarray,
    interval_duration_hours: float,
) -> float:
    """Calculate the perceived cost of one EV charging schedule."""
    return float(
        np.sum(
            schedule_kw
            * price_eur_per_kwh
            * interval_duration_hours
        )
    )


def calculate_price_eur_per_kwh(
    total_system_demand_mw: np.ndarray,
    quadratic_coefficient: float,
    linear_coefficient: float,
) -> np.ndarray:
    """Calculate the common legitimate marginal-price profile."""
    price_eur_per_mwh = calculate_marginal_price_eur_per_mwh(
        demand_mw=total_system_demand_mw,
        quadratic_coefficient=quadratic_coefficient,
        linear_coefficient=linear_coefficient,
    )

    return eur_per_mwh_to_eur_per_kwh(
        price_eur_per_mwh
    )


def perform_feedback_iteration(
    *,
    state: FeedbackState,
    selected_ev_indices: np.ndarray,
    region_by_ev: np.ndarray,
    aggregate_fixed_demand_mw_by_region: np.ndarray,
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
    unique_attacked_rescheduled: np.ndarray | None = None,
    unique_attacked_moved_into_target: np.ndarray | None = None,
) -> dict[str, float | int]:
    """
    Perform one asynchronous price-demand feedback iteration.

    Region A always receives the legitimate price. During the attack,
    selected Region-B EVs receive a forged price at the target hour.
    """
    previous_total_demand_mw_by_region = (
        state.total_demand_mw_by_region.copy()
    )

    candidate_changes = 0
    accepted_changes = 0
    attacked_accepted_changes = 0
    attacked_moved_into_target_changes = 0

    for ev_index_value in selected_ev_indices:
        ev_index = int(ev_index_value)
        ev_region = int(region_by_ev[ev_index])

        previous_schedule_kw = (
            state.ev_schedules_kw[ev_index].copy()
        )

        perceived_price_eur_per_kwh = (
            state.legitimate_price_eur_per_kwh.copy()
        )

        is_attacked_ev = bool(
            attack_is_active
            and receives_false_price[ev_index]
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

        moved_into_target = bool(
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

        state.aggregate_ev_demand_kw_by_region[
            ev_region
        ] -= previous_schedule_kw

        state.aggregate_ev_demand_kw_by_region[
            ev_region
        ] += candidate_schedule_kw

        state.ev_schedules_kw[
            ev_index
        ] = candidate_schedule_kw

    aggregate_ev_demand_mw_by_region = kw_to_mw(
        state.aggregate_ev_demand_kw_by_region
    )

    state.total_demand_mw_by_region = (
        aggregate_fixed_demand_mw_by_region
        + aggregate_ev_demand_mw_by_region
    )

    total_system_demand_mw = np.sum(
        state.total_demand_mw_by_region,
        axis=0,
    )

    raw_legitimate_price_eur_per_kwh = (
        calculate_price_eur_per_kwh(
            total_system_demand_mw=total_system_demand_mw,
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
        )
    )

    next_legitimate_price_eur_per_kwh = (
        (1.0 - price_damping_factor)
        * state.legitimate_price_eur_per_kwh
        + price_damping_factor
        * raw_legitimate_price_eur_per_kwh
    )

    applied_price_change = float(
        np.max(
            np.abs(
                next_legitimate_price_eur_per_kwh
                - state.legitimate_price_eur_per_kwh
            )
        )
    )

    maximum_regional_demand_change_mw = float(
        np.max(
            np.abs(
                state.total_demand_mw_by_region
                - previous_total_demand_mw_by_region
            )
        )
    )

    state.legitimate_price_eur_per_kwh = (
        next_legitimate_price_eur_per_kwh
    )

    return {
        "candidate_changes": candidate_changes,
        "accepted_changes": accepted_changes,
        "attacked_accepted_changes": attacked_accepted_changes,
        "attacked_moved_into_target_changes": (
            attacked_moved_into_target_changes
        ),
        "applied_price_change": applied_price_change,
        "maximum_regional_demand_change_mw": (
            maximum_regional_demand_change_mw
        ),
        "region_a_target_demand_mw": float(
            state.total_demand_mw_by_region[
                REGION_A,
                attacked_hour,
            ]
        ),
        "region_b_target_demand_mw": float(
            state.total_demand_mw_by_region[
                REGION_B,
                attacked_hour,
            ]
        ),
        "total_target_demand_mw": float(
            total_system_demand_mw[attacked_hour]
        ),
        "legitimate_target_price_eur_per_kwh": float(
            state.legitimate_price_eur_per_kwh[
                attacked_hour
            ]
        ),
    }


def build_three_bus_network(
    *,
    region_a_fixed_p_mw: float,
    region_a_ev_p_mw: float,
    region_b_fixed_p_mw: float,
    region_b_ev_p_mw: float,
    fixed_load_power_factor: float,
    ev_power_factor: float,
) -> pp.pandapowerNet:
    """
    Build the radial 20 kV network and separate fixed and EV loads.

        Bus 0: external grid
                 |
                 | Line 0
                 |
        Bus 1: Region A
                 |
                 | Line 1
                 |
        Bus 2: Region B
    """
    network = pp.create_empty_network(
        name="End-to-end regional price attack",
        sn_mva=100.0,
        f_hz=50.0,
    )

    nominal_voltage_kv = 20.0

    bus_grid = pp.create_bus(
        network,
        vn_kv=nominal_voltage_kv,
        name="Bus 0 - External grid",
    )

    bus_region_a = pp.create_bus(
        network,
        vn_kv=nominal_voltage_kv,
        name="Bus 1 - Region A",
    )

    bus_region_b = pp.create_bus(
        network,
        vn_kv=nominal_voltage_kv,
        name="Bus 2 - Region B",
    )

    pp.create_ext_grid(
        network,
        bus=bus_grid,
        vm_pu=1.02,
        va_degree=0.0,
        name="Upstream grid",
    )

    pp.create_line_from_parameters(
        network,
        from_bus=bus_grid,
        to_bus=bus_region_a,
        length_km=5.0,
        r_ohm_per_km=0.08,
        x_ohm_per_km=0.12,
        c_nf_per_km=10.0,
        max_i_ka=1.25,
        name="Line 0 - Grid to Region A",
    )

    pp.create_line_from_parameters(
        network,
        from_bus=bus_region_a,
        to_bus=bus_region_b,
        length_km=3.0,
        r_ohm_per_km=0.08,
        x_ohm_per_km=0.12,
        c_nf_per_km=10.0,
        max_i_ka=0.65,
        name="Line 1 - Region A to Region B",
    )

    pp.create_load(
        network,
        bus=bus_region_a,
        p_mw=region_a_fixed_p_mw,
        q_mvar=reactive_power_from_power_factor(
            active_power_mw=region_a_fixed_p_mw,
            power_factor=fixed_load_power_factor,
        ),
        name="Region A fixed load",
    )

    pp.create_load(
        network,
        bus=bus_region_a,
        p_mw=region_a_ev_p_mw,
        q_mvar=reactive_power_from_power_factor(
            active_power_mw=region_a_ev_p_mw,
            power_factor=ev_power_factor,
        ),
        name="Region A EV load",
    )

    pp.create_load(
        network,
        bus=bus_region_b,
        p_mw=region_b_fixed_p_mw,
        q_mvar=reactive_power_from_power_factor(
            active_power_mw=region_b_fixed_p_mw,
            power_factor=fixed_load_power_factor,
        ),
        name="Region B fixed load",
    )

    pp.create_load(
        network,
        bus=bus_region_b,
        p_mw=region_b_ev_p_mw,
        q_mvar=reactive_power_from_power_factor(
            active_power_mw=region_b_ev_p_mw,
            power_factor=ev_power_factor,
        ),
        name="Region B EV load",
    )

    return network


def run_power_flow(
    network: pp.pandapowerNet,
) -> bool:
    """Run one balanced AC Newton-Raphson power flow."""
    try:
        pp.runpp(
            network,
            algorithm="nr",
            calculate_voltage_angles=False,
            init="flat",
            max_iteration=50,
            tolerance_mva=1e-8,
            numba=False,
        )
    except LoadflowNotConverged:
        return False

    return bool(network.converged)


def evaluate_power_flow_timeseries(
    *,
    scenario_name: str,
    fixed_demand_mw_by_region: np.ndarray,
    total_demand_mw_by_region: np.ndarray,
    fixed_load_power_factor: float,
    ev_power_factor: float,
    minimum_voltage_limit_pu: float,
    maximum_voltage_limit_pu: float,
) -> pd.DataFrame:
    """Run one power flow for every hour of one demand scenario."""
    records: list[dict[str, object]] = []

    ev_demand_mw_by_region = (
        total_demand_mw_by_region
        - fixed_demand_mw_by_region
    )

    for hour in range(24):
        network = build_three_bus_network(
            region_a_fixed_p_mw=float(
                fixed_demand_mw_by_region[
                    REGION_A,
                    hour,
                ]
            ),
            region_a_ev_p_mw=float(
                ev_demand_mw_by_region[
                    REGION_A,
                    hour,
                ]
            ),
            region_b_fixed_p_mw=float(
                fixed_demand_mw_by_region[
                    REGION_B,
                    hour,
                ]
            ),
            region_b_ev_p_mw=float(
                ev_demand_mw_by_region[
                    REGION_B,
                    hour,
                ]
            ),
            fixed_load_power_factor=(
                fixed_load_power_factor
            ),
            ev_power_factor=ev_power_factor,
        )

        converged = run_power_flow(
            network
        )

        if not converged:
            records.append(
                {
                    "scenario": scenario_name,
                    "hour": hour,
                    "converged": False,
                    "region_a_fixed_demand_mw": float(
                        fixed_demand_mw_by_region[
                            REGION_A,
                            hour,
                        ]
                    ),
                    "region_a_ev_demand_mw": float(
                        ev_demand_mw_by_region[
                            REGION_A,
                            hour,
                        ]
                    ),
                    "region_a_total_demand_mw": float(
                        total_demand_mw_by_region[
                            REGION_A,
                            hour,
                        ]
                    ),
                    "region_b_fixed_demand_mw": float(
                        fixed_demand_mw_by_region[
                            REGION_B,
                            hour,
                        ]
                    ),
                    "region_b_ev_demand_mw": float(
                        ev_demand_mw_by_region[
                            REGION_B,
                            hour,
                        ]
                    ),
                    "region_b_total_demand_mw": float(
                        total_demand_mw_by_region[
                            REGION_B,
                            hour,
                        ]
                    ),
                    "total_system_demand_mw": float(
                        np.sum(
                            total_demand_mw_by_region[
                                :,
                                hour,
                            ]
                        )
                    ),
                    "bus_0_voltage_pu": np.nan,
                    "bus_1_voltage_pu": np.nan,
                    "bus_2_voltage_pu": np.nan,
                    "minimum_voltage_pu": np.nan,
                    "line_0_loading_percent": np.nan,
                    "line_1_loading_percent": np.nan,
                    "maximum_line_loading_percent": np.nan,
                    "total_line_losses_mw": np.nan,
                    "voltage_violation": False,
                    "line_overload": False,
                }
            )
            continue

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

        records.append(
            {
                "scenario": scenario_name,
                "hour": hour,
                "converged": True,
                "region_a_fixed_demand_mw": float(
                    fixed_demand_mw_by_region[
                        REGION_A,
                        hour,
                    ]
                ),
                "region_a_ev_demand_mw": float(
                    ev_demand_mw_by_region[
                        REGION_A,
                        hour,
                    ]
                ),
                "region_a_total_demand_mw": float(
                    total_demand_mw_by_region[
                        REGION_A,
                        hour,
                    ]
                ),
                "region_b_fixed_demand_mw": float(
                    fixed_demand_mw_by_region[
                        REGION_B,
                        hour,
                    ]
                ),
                "region_b_ev_demand_mw": float(
                    ev_demand_mw_by_region[
                        REGION_B,
                        hour,
                    ]
                ),
                "region_b_total_demand_mw": float(
                    total_demand_mw_by_region[
                        REGION_B,
                        hour,
                    ]
                ),
                "total_system_demand_mw": float(
                    np.sum(
                        total_demand_mw_by_region[
                            :,
                            hour,
                        ]
                    )
                ),
                "bus_0_voltage_pu": float(
                    network.res_bus.at[
                        0,
                        "vm_pu",
                    ]
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
                "minimum_voltage_pu": float(
                    network.res_bus[
                        "vm_pu"
                    ].min()
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
        )

    return pd.DataFrame(
        records
    )


def save_figures(
    *,
    output_directory: Path,
    hours: np.ndarray,
    baseline_state: FeedbackState,
    attacked_state: FeedbackState,
    baseline_power_flow: pd.DataFrame,
    attacked_power_flow: pd.DataFrame,
    attacked_hour: int,
    minimum_voltage_limit_pu: float,
) -> None:
    """Save demand-response and grid-impact figures."""
    baseline_ev_mw_by_region = kw_to_mw(
        baseline_state.aggregate_ev_demand_kw_by_region
    )

    attacked_ev_mw_by_region = kw_to_mw(
        attacked_state.aggregate_ev_demand_kw_by_region
    )

    # Regional total demand profiles.
    plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        baseline_state.total_demand_mw_by_region[
            REGION_A
        ],
        marker="o",
        label="Region A baseline",
    )

    plt.plot(
        hours,
        baseline_state.total_demand_mw_by_region[
            REGION_B
        ],
        marker="o",
        label="Region B baseline",
    )

    plt.plot(
        hours,
        attacked_state.total_demand_mw_by_region[
            REGION_A
        ],
        marker="o",
        label="Region A during attack",
    )

    plt.plot(
        hours,
        attacked_state.total_demand_mw_by_region[
            REGION_B
        ],
        marker="o",
        label="Region B during attack",
    )

    plt.xlabel("Hour")
    plt.ylabel("Regional total demand (MW)")
    plt.title(
        "Regional Demand Before and During the Price Attack"
    )
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_17_regional_demand_profiles.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Attack-induced regional EV-demand change.
    plt.figure(figsize=(10, 5))

    plt.plot(
        hours,
        (
            attacked_ev_mw_by_region[
                REGION_A
            ]
            - baseline_ev_mw_by_region[
                REGION_A
            ]
        ),
        marker="o",
        label="Region A EV-demand change",
    )

    plt.plot(
        hours,
        (
            attacked_ev_mw_by_region[
                REGION_B
            ]
            - baseline_ev_mw_by_region[
                REGION_B
            ]
        ),
        marker="o",
        label="Region B EV-demand change",
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.xlabel("Hour")
    plt.ylabel("Attack-induced EV-demand change (MW)")
    plt.title(
        "Regional Demand Shift Caused by the Forged Price"
    )
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_17_regional_attack_difference.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Minimum voltage over the day.
    plt.figure(figsize=(10, 5))

    plt.plot(
        baseline_power_flow["hour"],
        baseline_power_flow[
            "minimum_voltage_pu"
        ],
        marker="o",
        label="Baseline",
    )

    plt.plot(
        attacked_power_flow["hour"],
        attacked_power_flow[
            "minimum_voltage_pu"
        ],
        marker="o",
        label="Attack",
    )

    plt.axhline(
        minimum_voltage_limit_pu,
        linestyle="--",
        label="Lower voltage limit",
    )

    plt.xlabel("Hour")
    plt.ylabel("Minimum voltage (p.u.)")
    plt.title(
        "Minimum Network Voltage Before and During Attack"
    )
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_17_minimum_voltage.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Line 0 loading.
    plt.figure(figsize=(10, 5))

    plt.plot(
        baseline_power_flow["hour"],
        baseline_power_flow[
            "line_0_loading_percent"
        ],
        marker="o",
        label="Line 0 baseline",
    )

    plt.plot(
        attacked_power_flow["hour"],
        attacked_power_flow[
            "line_0_loading_percent"
        ],
        marker="o",
        label="Line 0 attack",
    )

    plt.axhline(
        100.0,
        linestyle="--",
        label="Thermal limit",
    )

    plt.xlabel("Hour")
    plt.ylabel("Line 0 loading (%)")
    plt.title(
        "Upstream Line Loading Before and During Attack"
    )
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_17_line_0_loading.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Line 1 loading.
    plt.figure(figsize=(10, 5))

    plt.plot(
        baseline_power_flow["hour"],
        baseline_power_flow[
            "line_1_loading_percent"
        ],
        marker="o",
        label="Line 1 baseline",
    )

    plt.plot(
        attacked_power_flow["hour"],
        attacked_power_flow[
            "line_1_loading_percent"
        ],
        marker="o",
        label="Line 1 attack",
    )

    plt.axhline(
        100.0,
        linestyle="--",
        label="Thermal limit",
    )

    plt.xlabel("Hour")
    plt.ylabel("Line 1 loading (%)")
    plt.title(
        "Downstream Line Loading Before and During Attack"
    )
    plt.xticks(hours)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_17_line_1_loading.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Target-hour physical comparison.
    target_baseline = baseline_power_flow[
        baseline_power_flow["hour"]
        == attacked_hour
    ].iloc[0]

    target_attack = attacked_power_flow[
        attacked_power_flow["hour"]
        == attacked_hour
    ].iloc[0]

    metric_names = [
        "Minimum voltage (p.u.)",
        "Line 0 loading / 100",
        "Line 1 loading / 100",
    ]

    baseline_values = [
        float(
            target_baseline[
                "minimum_voltage_pu"
            ]
        ),
        float(
            target_baseline[
                "line_0_loading_percent"
            ]
        )
        / 100.0,
        float(
            target_baseline[
                "line_1_loading_percent"
            ]
        )
        / 100.0,
    ]

    attack_values = [
        float(
            target_attack[
                "minimum_voltage_pu"
            ]
        ),
        float(
            target_attack[
                "line_0_loading_percent"
            ]
        )
        / 100.0,
        float(
            target_attack[
                "line_1_loading_percent"
            ]
        )
        / 100.0,
    ]

    x_positions = np.arange(
        len(metric_names)
    )
    bar_width = 0.36

    plt.figure(figsize=(10, 5))

    plt.bar(
        x_positions - bar_width / 2.0,
        baseline_values,
        width=bar_width,
        label="Baseline",
    )

    plt.bar(
        x_positions + bar_width / 2.0,
        attack_values,
        width=bar_width,
        label="Attack",
    )

    plt.xticks(
        x_positions,
        metric_names,
        rotation=10,
    )

    plt.ylabel("Per-unit comparison")
    plt.title(
        f"Physical Impact at {attacked_hour:02d}:00"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_17_target_hour_physical_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    hours = np.arange(24)
    interval_duration_hours = 1.0

    # ============================================================
    # 1. Simulation configuration
    # ============================================================

    random_seed_population = 42
    random_seed_baseline_updates = 1001
    random_seed_attack_updates = 2001
    random_seed_region_assignment = 3001

    number_of_households = 10_000
    households_per_region = (
        number_of_households // 2
    )
    ev_adoption_rate = 0.30

    price_damping_factor = 0.20
    ev_rescheduling_fraction = 0.10
    minimum_saving_to_reschedule_eur = 0.10

    baseline_maximum_iterations = 150
    attack_iterations = 20

    convergence_price_tolerance_eur_per_kwh = 1e-4
    convergence_demand_tolerance_mw = 0.01
    convergence_patience = 5

    attacked_hour = 18
    attacked_region_b_ev_fraction = 1.00
    false_price_eur_per_kwh = 0.05

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
            quadratic_coefficient=quadratic_coefficient,
            linear_coefficient=linear_coefficient,
        )
    )

    # ============================================================
    # 3. Heterogeneous regional EV population
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
        [
            REGION_A
        ]
        * evs_per_region
        + [
            REGION_B
        ]
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

    # Exactly the configured fraction of Region-B EVs receive the forged price.
    region_b_ev_indices = np.flatnonzero(
        region_by_ev == REGION_B
    )

    number_of_attacked_region_b_evs = int(
        round(
            len(region_b_ev_indices)
            * attacked_region_b_ev_fraction
        )
    )

    receives_false_price = np.zeros(
        number_of_evs,
        dtype=bool,
    )

    receives_false_price[
        region_b_ev_indices[
            :number_of_attacked_region_b_evs
        ]
    ] = True

    attacked_region_b_connected_at_target = int(
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

    # ============================================================
    # 4. Initial schedules and common baseline
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

    evs_selected_per_iteration = max(
        1,
        int(
            round(
                number_of_evs
                * ev_rescheduling_fraction
            )
        ),
    )

    rng_baseline_updates = np.random.default_rng(
        random_seed_baseline_updates
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

    no_attack_mask = np.zeros(
        number_of_evs,
        dtype=bool,
    )

    baseline_history: list[
        dict[str, object]
    ] = []

    consecutive_converged_iterations = 0
    baseline_converged = False
    baseline_completed_iterations = 0

    print(
        "=== Step 17: end-to-end regional price attack ==="
    )
    print()
    print("Configuration")
    print(
        f"  Households per region: "
        f"{households_per_region:,}"
    )
    print(
        f"  EVs in Region A: "
        f"{int(np.sum(region_by_ev == REGION_A)):,}"
    )
    print(
        f"  EVs in Region B: "
        f"{int(np.sum(region_by_ev == REGION_B)):,}"
    )
    print(
        f"  Attacked Region-B EVs: "
        f"{number_of_attacked_region_b_evs:,}"
    )
    print(
        f"  Attacked Region-B EVs connected at "
        f"{attacked_hour:02d}:00: "
        f"{attacked_region_b_connected_at_target:,}"
    )
    print(
        f"  False price at "
        f"{attacked_hour:02d}:00: "
        f"€{false_price_eur_per_kwh:.2f}/kWh"
    )
    print()

    print("Phase A: converge the common legitimate baseline")

    for (
        iteration_index,
        selected_indices,
    ) in enumerate(
        baseline_update_plan
    ):
        result = perform_feedback_iteration(
            state=baseline_state,
            selected_ev_indices=selected_indices,
            region_by_ev=region_by_ev,
            aggregate_fixed_demand_mw_by_region=(
                aggregate_fixed_demand_mw_by_region
            ),
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
            receives_false_price=no_attack_mask,
            attacked_hour=attacked_hour,
            false_price_eur_per_kwh=(
                false_price_eur_per_kwh
            ),
        )

        baseline_completed_iterations = (
            iteration_index + 1
        )

        baseline_history.append(
            {
                "phase": "Baseline",
                "iteration": (
                    baseline_completed_iterations
                ),
                **result,
            }
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
            baseline_completed_iterations == 1
            or baseline_completed_iterations % 5 == 0
        ):
            print(
                f"  Iteration "
                f"{baseline_completed_iterations:3d}: "
                f"accepted="
                f"{int(result['accepted_changes']):3d}, "
                f"max regional ΔD="
                f"{float(result['maximum_regional_demand_change_mw']):.4f} MW, "
                f"applied Δprice="
                f"{float(result['applied_price_change']):.6f}"
            )

        if (
            consecutive_converged_iterations
            >= convergence_patience
        ):
            baseline_converged = True
            break

    if not baseline_converged:
        raise RuntimeError(
            "The regional baseline did not converge. "
            "Increase baseline_maximum_iterations or revise the "
            "feedback parameters."
        )

    converged_baseline_state = (
        baseline_state.clone()
    )

    print(
        f"  Baseline converged after "
        f"{baseline_completed_iterations} iterations."
    )

    # ============================================================
    # 5. Attack Region B only
    # ============================================================

    attacked_state = (
        converged_baseline_state.clone()
    )

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
            attack_iterations
        )
    ]

    unique_attacked_rescheduled = np.zeros(
        number_of_evs,
        dtype=bool,
    )

    unique_attacked_moved_into_target = np.zeros(
        number_of_evs,
        dtype=bool,
    )

    attack_history: list[
        dict[str, object]
    ] = []

    print()
    print("Phase B: attack Region B")

    for (
        iteration_index,
        selected_indices,
    ) in enumerate(
        attack_update_plan
    ):
        result = perform_feedback_iteration(
            state=attacked_state,
            selected_ev_indices=selected_indices,
            region_by_ev=region_by_ev,
            aggregate_fixed_demand_mw_by_region=(
                aggregate_fixed_demand_mw_by_region
            ),
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
            receives_false_price=(
                receives_false_price
            ),
            attacked_hour=attacked_hour,
            false_price_eur_per_kwh=(
                false_price_eur_per_kwh
            ),
            unique_attacked_rescheduled=(
                unique_attacked_rescheduled
            ),
            unique_attacked_moved_into_target=(
                unique_attacked_moved_into_target
            ),
        )

        attack_history.append(
            {
                "phase": "Attack",
                "iteration": iteration_index + 1,
                **result,
            }
        )

        print(
            f"  Iteration "
            f"{iteration_index + 1:2d}: "
            f"attacked accepted="
            f"{int(result['attacked_accepted_changes']):3d}, "
            f"Region-B demand at "
            f"{attacked_hour:02d}:00="
            f"{float(result['region_b_target_demand_mw']):.2f} MW, "
            f"legitimate price="
            f"€{float(result['legitimate_target_price_eur_per_kwh']):.3f}/kWh"
        )

    # ============================================================
    # 6. End-to-end power-flow evaluation
    # ============================================================

    baseline_power_flow = (
        evaluate_power_flow_timeseries(
            scenario_name="Baseline",
            fixed_demand_mw_by_region=(
                aggregate_fixed_demand_mw_by_region
            ),
            total_demand_mw_by_region=(
                converged_baseline_state
                .total_demand_mw_by_region
            ),
            fixed_load_power_factor=(
                fixed_load_power_factor
            ),
            ev_power_factor=ev_power_factor,
            minimum_voltage_limit_pu=(
                minimum_voltage_limit_pu
            ),
            maximum_voltage_limit_pu=(
                maximum_voltage_limit_pu
            ),
        )
    )

    attacked_power_flow = (
        evaluate_power_flow_timeseries(
            scenario_name="Attack",
            fixed_demand_mw_by_region=(
                aggregate_fixed_demand_mw_by_region
            ),
            total_demand_mw_by_region=(
                attacked_state
                .total_demand_mw_by_region
            ),
            fixed_load_power_factor=(
                fixed_load_power_factor
            ),
            ev_power_factor=ev_power_factor,
            minimum_voltage_limit_pu=(
                minimum_voltage_limit_pu
            ),
            maximum_voltage_limit_pu=(
                maximum_voltage_limit_pu
            ),
        )
    )

    combined_power_flow = pd.concat(
        [
            baseline_power_flow,
            attacked_power_flow,
        ],
        ignore_index=True,
    )

    target_baseline = baseline_power_flow[
        baseline_power_flow["hour"]
        == attacked_hour
    ].iloc[0]

    target_attack = attacked_power_flow[
        attacked_power_flow["hour"]
        == attacked_hour
    ].iloc[0]

    baseline_region_b_target_demand_mw = float(
        target_baseline[
            "region_b_total_demand_mw"
        ]
    )

    attacked_region_b_target_demand_mw = float(
        target_attack[
            "region_b_total_demand_mw"
        ]
    )

    region_b_attack_increment_mw = (
        attacked_region_b_target_demand_mw
        - baseline_region_b_target_demand_mw
    )

    unique_attacked_rescheduled_count = int(
        np.sum(
            unique_attacked_rescheduled
        )
    )

    unique_attacked_moved_into_target_count = int(
        np.sum(
            unique_attacked_moved_into_target
        )
    )

    # ============================================================
    # 7. Print summary
    # ============================================================

    print()
    print("=== End-to-end result at 18:00 ===")
    print()
    print("Demand-response layer")
    print(
        f"  Baseline Region-A demand: "
        f"{float(target_baseline['region_a_total_demand_mw']):.2f} MW"
    )
    print(
        f"  Baseline Region-B demand: "
        f"{baseline_region_b_target_demand_mw:.2f} MW"
    )
    print(
        f"  Attacked Region-A demand: "
        f"{float(target_attack['region_a_total_demand_mw']):.2f} MW"
    )
    print(
        f"  Attacked Region-B demand: "
        f"{attacked_region_b_target_demand_mw:.2f} MW"
    )
    print(
        f"  Region-B attack increment generated by EV response: "
        f"{region_b_attack_increment_mw:.2f} MW"
    )
    print(
        f"  Unique attacked EVs rescheduled: "
        f"{unique_attacked_rescheduled_count:,}"
    )
    print(
        f"  Unique attacked EVs moved charging into 18:00: "
        f"{unique_attacked_moved_into_target_count:,}"
    )

    print()
    print("Physical grid layer")
    print(
        f"  Baseline minimum voltage: "
        f"{float(target_baseline['minimum_voltage_pu']):.4f} p.u."
    )
    print(
        f"  Attack minimum voltage: "
        f"{float(target_attack['minimum_voltage_pu']):.4f} p.u."
    )
    print(
        f"  Baseline Line 0 loading: "
        f"{float(target_baseline['line_0_loading_percent']):.2f}%"
    )
    print(
        f"  Attack Line 0 loading: "
        f"{float(target_attack['line_0_loading_percent']):.2f}%"
    )
    print(
        f"  Baseline Line 1 loading: "
        f"{float(target_baseline['line_1_loading_percent']):.2f}%"
    )
    print(
        f"  Attack Line 1 loading: "
        f"{float(target_attack['line_1_loading_percent']):.2f}%"
    )
    print(
        f"  Baseline voltage violation: "
        f"{bool(target_baseline['voltage_violation'])}"
    )
    print(
        f"  Attack voltage violation: "
        f"{bool(target_attack['voltage_violation'])}"
    )
    print(
        f"  Baseline line overload: "
        f"{bool(target_baseline['line_overload'])}"
    )
    print(
        f"  Attack line overload: "
        f"{bool(target_attack['line_overload'])}"
    )
    print(
        f"  Baseline active line losses: "
        f"{float(target_baseline['total_line_losses_mw']):.4f} MW"
    )
    print(
        f"  Attack active line losses: "
        f"{float(target_attack['total_line_losses_mw']):.4f} MW"
    )

    # ============================================================
    # 8. Save tables, configuration, and figures
    # ============================================================

    feedback_history = pd.DataFrame(
        baseline_history
        + attack_history
    )

    target_hour_comparison = pd.DataFrame(
        [
            target_baseline.to_dict(),
            target_attack.to_dict(),
        ]
    )

    feedback_history.to_csv(
        output_directory
        / "step_17_feedback_history.csv",
        index=False,
    )

    combined_power_flow.to_csv(
        output_directory
        / "step_17_power_flow_timeseries.csv",
        index=False,
    )

    target_hour_comparison.to_csv(
        output_directory
        / "step_17_target_hour_comparison.csv",
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
        "evs_region_a": int(
            np.sum(
                region_by_ev == REGION_A
            )
        ),
        "evs_region_b": int(
            np.sum(
                region_by_ev == REGION_B
            )
        ),
        "attacked_region_b_ev_fraction": (
            attacked_region_b_ev_fraction
        ),
        "number_of_attacked_region_b_evs": (
            number_of_attacked_region_b_evs
        ),
        "attacked_hour": attacked_hour,
        "false_price_eur_per_kwh": (
            false_price_eur_per_kwh
        ),
        "price_damping_factor": (
            price_damping_factor
        ),
        "ev_rescheduling_fraction": (
            ev_rescheduling_fraction
        ),
        "minimum_saving_to_reschedule_eur": (
            minimum_saving_to_reschedule_eur
        ),
        "baseline_completed_iterations": (
            baseline_completed_iterations
        ),
        "attack_iterations": (
            attack_iterations
        ),
        "fixed_load_power_factor": (
            fixed_load_power_factor
        ),
        "ev_power_factor": (
            ev_power_factor
        ),
        "minimum_voltage_limit_pu": (
            minimum_voltage_limit_pu
        ),
        "maximum_voltage_limit_pu": (
            maximum_voltage_limit_pu
        ),
        "region_b_attack_increment_mw": (
            region_b_attack_increment_mw
        ),
        "unique_attacked_rescheduled": (
            unique_attacked_rescheduled_count
        ),
        "unique_attacked_moved_into_target": (
            unique_attacked_moved_into_target_count
        ),
    }

    (
        output_directory
        / "step_17_config_and_summary.json"
    ).write_text(
        json.dumps(
            configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_figures(
        output_directory=output_directory,
        hours=hours,
        baseline_state=(
            converged_baseline_state
        ),
        attacked_state=attacked_state,
        baseline_power_flow=(
            baseline_power_flow
        ),
        attacked_power_flow=(
            attacked_power_flow
        ),
        attacked_hour=attacked_hour,
        minimum_voltage_limit_pu=(
            minimum_voltage_limit_pu
        ),
    )

    print()
    print("Files saved:")
    print(
        "  results/"
        "step_17_feedback_history.csv"
    )
    print(
        "  results/"
        "step_17_power_flow_timeseries.csv"
    )
    print(
        "  results/"
        "step_17_target_hour_comparison.csv"
    )
    print(
        "  results/"
        "step_17_config_and_summary.json"
    )
    print(
        "  results/"
        "step_17_regional_demand_profiles.png"
    )
    print(
        "  results/"
        "step_17_regional_attack_difference.png"
    )
    print(
        "  results/"
        "step_17_minimum_voltage.png"
    )
    print(
        "  results/"
        "step_17_line_0_loading.png"
    )
    print(
        "  results/"
        "step_17_line_1_loading.png"
    )
    print(
        "  results/"
        "step_17_target_hour_physical_comparison.png"
    )


if __name__ == "__main__":
    main()
