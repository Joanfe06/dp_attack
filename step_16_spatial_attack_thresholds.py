from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# Use a non-interactive backend because the simulation runs without a GUI.
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


def reactive_power_from_power_factor(
    active_power_mw: float,
    power_factor: float,
) -> float:
    """
    Calculate inductive reactive power from active power and power factor.

    power_factor = cos(phi)
    Q = P * tan(phi)
    """
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


def build_three_bus_network(
    *,
    region_a_p_mw: float,
    region_b_p_mw: float,
    load_power_factor: float,
) -> pp.pandapowerNet:
    """
    Build the radial 20 kV network used in Steps 14 and 15.

        Bus 0: external grid
                 |
                 | Line 0
                 |
        Bus 1: residential region A
                 |
                 | Line 1
                 |
        Bus 2: residential region B
    """
    network = pp.create_empty_network(
        name="Three-bus spatial attack threshold sweep",
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
        name="Bus 1 - Residential region A",
    )

    bus_region_b = pp.create_bus(
        network,
        vn_kv=nominal_voltage_kv,
        name="Bus 2 - Residential region B",
    )

    pp.create_ext_grid(
        network,
        bus=bus_grid,
        vm_pu=1.02,
        va_degree=0.0,
        name="Upstream grid",
    )

    # Line 0 carries the load of both regions.
    pp.create_line_from_parameters(
        network,
        from_bus=bus_grid,
        to_bus=bus_region_a,
        length_km=5.0,
        r_ohm_per_km=0.08,
        x_ohm_per_km=0.12,
        c_nf_per_km=10.0,
        max_i_ka=1.25,
        name="Line 0 - Grid to region A",
    )

    # Line 1 carries only the load located at Region B.
    pp.create_line_from_parameters(
        network,
        from_bus=bus_region_a,
        to_bus=bus_region_b,
        length_km=3.0,
        r_ohm_per_km=0.08,
        x_ohm_per_km=0.12,
        c_nf_per_km=10.0,
        max_i_ka=0.65,
        name="Line 1 - Region A to region B",
    )

    pp.create_load(
        network,
        bus=bus_region_a,
        p_mw=region_a_p_mw,
        q_mvar=reactive_power_from_power_factor(
            active_power_mw=region_a_p_mw,
            power_factor=load_power_factor,
        ),
        name="Region A demand",
    )

    pp.create_load(
        network,
        bus=bus_region_b,
        p_mw=region_b_p_mw,
        q_mvar=reactive_power_from_power_factor(
            active_power_mw=region_b_p_mw,
            power_factor=load_power_factor,
        ),
        name="Region B demand",
    )

    return network


def run_power_flow(
    network: pp.pandapowerNet,
) -> bool:
    """
    Run a balanced AC Newton-Raphson power flow.

    Returns
    -------
    bool
        True when the power flow converges; otherwise False.
    """
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


def empty_result_record(
    *,
    attack_location: str,
    attack_increment_mw: float,
    region_a_p_mw: float,
    region_b_p_mw: float,
) -> dict[str, object]:
    """Create a result record for a non-convergent operating point."""
    return {
        "attack_location": attack_location,
        "attack_increment_mw": attack_increment_mw,
        "region_a_demand_mw": region_a_p_mw,
        "region_b_demand_mw": region_b_p_mw,
        "total_load_mw": region_a_p_mw + region_b_p_mw,
        "converged": False,
        "bus_0_voltage_pu": np.nan,
        "bus_1_voltage_pu": np.nan,
        "bus_2_voltage_pu": np.nan,
        "minimum_voltage_pu": np.nan,
        "minimum_voltage_bus": np.nan,
        "line_0_loading_percent": np.nan,
        "line_1_loading_percent": np.nan,
        "maximum_line_loading_percent": np.nan,
        "total_line_losses_mw": np.nan,
        "external_grid_active_power_mw": np.nan,
        "external_grid_reactive_power_mvar": np.nan,
        "voltage_violation": False,
        "line_overload": False,
    }


def extract_result_record(
    *,
    network: pp.pandapowerNet,
    attack_location: str,
    attack_increment_mw: float,
    region_a_p_mw: float,
    region_b_p_mw: float,
    minimum_voltage_limit_pu: float,
    maximum_voltage_limit_pu: float,
) -> dict[str, object]:
    """Extract the physical metrics for one converged operating point."""
    minimum_voltage_bus = int(
        network.res_bus["vm_pu"].idxmin()
    )

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
            network.res_line["loading_percent"]
            > 100.0
        )
    )

    return {
        "attack_location": attack_location,
        "attack_increment_mw": attack_increment_mw,
        "region_a_demand_mw": region_a_p_mw,
        "region_b_demand_mw": region_b_p_mw,
        "total_load_mw": region_a_p_mw + region_b_p_mw,
        "converged": True,
        "bus_0_voltage_pu": float(
            network.res_bus.at[0, "vm_pu"]
        ),
        "bus_1_voltage_pu": float(
            network.res_bus.at[1, "vm_pu"]
        ),
        "bus_2_voltage_pu": float(
            network.res_bus.at[2, "vm_pu"]
        ),
        "minimum_voltage_pu": float(
            network.res_bus["vm_pu"].min()
        ),
        "minimum_voltage_bus": minimum_voltage_bus,
        "line_0_loading_percent": float(
            network.res_line.at[0, "loading_percent"]
        ),
        "line_1_loading_percent": float(
            network.res_line.at[1, "loading_percent"]
        ),
        "maximum_line_loading_percent": float(
            network.res_line["loading_percent"].max()
        ),
        "total_line_losses_mw": float(
            network.res_line["pl_mw"].sum()
        ),
        "external_grid_active_power_mw": float(
            network.res_ext_grid.at[0, "p_mw"]
        ),
        "external_grid_reactive_power_mvar": float(
            network.res_ext_grid.at[0, "q_mvar"]
        ),
        "voltage_violation": voltage_violation,
        "line_overload": line_overload,
    }


def first_threshold(
    *,
    location_results: pd.DataFrame,
    condition_column: str,
) -> float:
    """
    Return the first sampled attack increment satisfying one condition.

    NaN is returned if the condition never occurs in the sweep.
    """
    matching_rows = location_results[
        location_results[condition_column]
    ]

    if matching_rows.empty:
        return float("nan")

    return float(
        matching_rows[
            "attack_increment_mw"
        ].min()
    )


def first_nonconvergence_threshold(
    location_results: pd.DataFrame,
) -> float:
    """Return the first sampled increment where power flow does not converge."""
    nonconvergent_rows = location_results[
        ~location_results["converged"]
    ]

    if nonconvergent_rows.empty:
        return float("nan")

    return float(
        nonconvergent_rows[
            "attack_increment_mw"
        ].min()
    )


def build_threshold_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Create one threshold row for each attack location."""
    summary_records: list[dict[str, object]] = []

    for attack_location in [
        "Region A",
        "Region B",
    ]:
        location_results = results[
            results["attack_location"]
            == attack_location
        ].sort_values(
            "attack_increment_mw"
        )

        summary_records.append(
            {
                "attack_location": attack_location,
                "first_voltage_violation_mw": first_threshold(
                    location_results=location_results,
                    condition_column="voltage_violation",
                ),
                "first_line_overload_mw": first_threshold(
                    location_results=location_results,
                    condition_column="line_overload",
                ),
                "first_nonconvergence_mw": (
                    first_nonconvergence_threshold(
                        location_results
                    )
                ),
                "lowest_converged_voltage_pu": float(
                    location_results.loc[
                        location_results["converged"],
                        "minimum_voltage_pu",
                    ].min()
                ),
                "highest_converged_line_loading_percent": float(
                    location_results.loc[
                        location_results["converged"],
                        "maximum_line_loading_percent",
                    ].max()
                ),
            }
        )

    return pd.DataFrame(
        summary_records
    )


def print_threshold_value(
    value: float,
) -> str:
    """Format one threshold for terminal output."""
    if np.isnan(value):
        return "not reached"

    return f"{value:.2f} MW"


def print_results(
    *,
    results: pd.DataFrame,
    threshold_summary: pd.DataFrame,
) -> None:
    """Print the sweep results and physical thresholds."""
    print("=== Three-bus spatial attack threshold sweep ===")
    print()
    print(
        "The same attack magnitude is added either to upstream "
        "Region A or downstream Region B."
    )
    print()

    display_columns = [
        "attack_location",
        "attack_increment_mw",
        "minimum_voltage_pu",
        "line_0_loading_percent",
        "line_1_loading_percent",
        "total_line_losses_mw",
        "voltage_violation",
        "line_overload",
        "converged",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("Threshold summary")

    for _, row in threshold_summary.iterrows():
        print()
        print(f"  {row['attack_location']}")
        print(
            "    First sampled voltage violation: "
            + print_threshold_value(
                float(
                    row[
                        "first_voltage_violation_mw"
                    ]
                )
            )
        )
        print(
            "    First sampled line overload: "
            + print_threshold_value(
                float(
                    row[
                        "first_line_overload_mw"
                    ]
                )
            )
        )
        print(
            "    First sampled non-convergence: "
            + print_threshold_value(
                float(
                    row[
                        "first_nonconvergence_mw"
                    ]
                )
            )
        )
        print(
            "    Lowest converged voltage: "
            f"{float(row['lowest_converged_voltage_pu']):.4f} p.u."
        )
        print(
            "    Highest converged line loading: "
            f"{float(row['highest_converged_line_loading_percent']):.2f}%"
        )


def save_figures(
    *,
    results: pd.DataFrame,
    output_directory: Path,
    minimum_voltage_limit_pu: float,
) -> None:
    """Save the threshold-sweep figures."""
    converged_results = results[
        results["converged"]
    ]

    # Minimum voltage against attack magnitude.
    plt.figure(figsize=(10, 5))

    for attack_location in [
        "Region A",
        "Region B",
    ]:
        location_results = converged_results[
            converged_results[
                "attack_location"
            ]
            == attack_location
        ]

        plt.plot(
            location_results[
                "attack_increment_mw"
            ],
            location_results[
                "minimum_voltage_pu"
            ],
            marker="o",
            label=f"Attack at {attack_location}",
        )

    plt.axhline(
        minimum_voltage_limit_pu,
        linestyle="--",
        label="Lower voltage limit",
    )

    plt.xlabel("Additional attack demand (MW)")
    plt.ylabel("Minimum bus voltage (p.u.)")
    plt.title(
        "Voltage Sensitivity to Attack Magnitude and Location"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_16_minimum_voltage_thresholds.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Line 0 loading.
    plt.figure(figsize=(10, 5))

    for attack_location in [
        "Region A",
        "Region B",
    ]:
        location_results = converged_results[
            converged_results[
                "attack_location"
            ]
            == attack_location
        ]

        plt.plot(
            location_results[
                "attack_increment_mw"
            ],
            location_results[
                "line_0_loading_percent"
            ],
            marker="o",
            label=f"Attack at {attack_location}",
        )

    plt.axhline(
        100.0,
        linestyle="--",
        label="Thermal limit",
    )

    plt.xlabel("Additional attack demand (MW)")
    plt.ylabel("Line 0 loading (%)")
    plt.title(
        "Upstream Line Loading versus Attack Magnitude"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_16_line_0_thresholds.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Line 1 loading.
    plt.figure(figsize=(10, 5))

    for attack_location in [
        "Region A",
        "Region B",
    ]:
        location_results = converged_results[
            converged_results[
                "attack_location"
            ]
            == attack_location
        ]

        plt.plot(
            location_results[
                "attack_increment_mw"
            ],
            location_results[
                "line_1_loading_percent"
            ],
            marker="o",
            label=f"Attack at {attack_location}",
        )

    plt.axhline(
        100.0,
        linestyle="--",
        label="Thermal limit",
    )

    plt.xlabel("Additional attack demand (MW)")
    plt.ylabel("Line 1 loading (%)")
    plt.title(
        "Downstream Line Loading versus Attack Magnitude"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_16_line_1_thresholds.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Maximum line loading.
    plt.figure(figsize=(10, 5))

    for attack_location in [
        "Region A",
        "Region B",
    ]:
        location_results = converged_results[
            converged_results[
                "attack_location"
            ]
            == attack_location
        ]

        plt.plot(
            location_results[
                "attack_increment_mw"
            ],
            location_results[
                "maximum_line_loading_percent"
            ],
            marker="o",
            label=f"Attack at {attack_location}",
        )

    plt.axhline(
        100.0,
        linestyle="--",
        label="Thermal limit",
    )

    plt.xlabel("Additional attack demand (MW)")
    plt.ylabel("Maximum line loading (%)")
    plt.title(
        "Maximum Network Loading versus Attack Magnitude"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_16_maximum_line_loading.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Active line losses.
    plt.figure(figsize=(10, 5))

    for attack_location in [
        "Region A",
        "Region B",
    ]:
        location_results = converged_results[
            converged_results[
                "attack_location"
            ]
            == attack_location
        ]

        plt.plot(
            location_results[
                "attack_increment_mw"
            ],
            location_results[
                "total_line_losses_mw"
            ],
            marker="o",
            label=f"Attack at {attack_location}",
        )

    plt.xlabel("Additional attack demand (MW)")
    plt.ylabel("Total active line losses (MW)")
    plt.title(
        "Network Losses versus Attack Magnitude"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_16_line_losses.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Region-B voltage specifically.
    plt.figure(figsize=(10, 5))

    for attack_location in [
        "Region A",
        "Region B",
    ]:
        location_results = converged_results[
            converged_results[
                "attack_location"
            ]
            == attack_location
        ]

        plt.plot(
            location_results[
                "attack_increment_mw"
            ],
            location_results[
                "bus_2_voltage_pu"
            ],
            marker="o",
            label=f"Attack at {attack_location}",
        )

    plt.axhline(
        minimum_voltage_limit_pu,
        linestyle="--",
        label="Lower voltage limit",
    )

    plt.xlabel("Additional attack demand (MW)")
    plt.ylabel("Region B voltage (p.u.)")
    plt.title(
        "Downstream Voltage versus Attack Magnitude"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_16_region_b_voltage.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    # ============================================================
    # 1. Sweep configuration
    # ============================================================

    baseline_region_demand_mw = 15.0
    load_power_factor = 0.95

    minimum_voltage_limit_pu = 0.95
    maximum_voltage_limit_pu = 1.05

    attack_increments_mw = np.arange(
        0.0,
        10.0 + 0.5,
        0.5,
    )

    output_directory = Path("results")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # 2. Run both spatial attack families
    # ============================================================

    result_records: list[dict[str, object]] = []

    for attack_location in [
        "Region A",
        "Region B",
    ]:
        for attack_increment_mw in (
            attack_increments_mw
        ):
            if attack_location == "Region A":
                region_a_p_mw = (
                    baseline_region_demand_mw
                    + float(attack_increment_mw)
                )
                region_b_p_mw = (
                    baseline_region_demand_mw
                )
            else:
                region_a_p_mw = (
                    baseline_region_demand_mw
                )
                region_b_p_mw = (
                    baseline_region_demand_mw
                    + float(attack_increment_mw)
                )

            network = build_three_bus_network(
                region_a_p_mw=region_a_p_mw,
                region_b_p_mw=region_b_p_mw,
                load_power_factor=load_power_factor,
            )

            converged = run_power_flow(
                network
            )

            if not converged:
                result_records.append(
                    empty_result_record(
                        attack_location=attack_location,
                        attack_increment_mw=float(
                            attack_increment_mw
                        ),
                        region_a_p_mw=region_a_p_mw,
                        region_b_p_mw=region_b_p_mw,
                    )
                )
                continue

            result_records.append(
                extract_result_record(
                    network=network,
                    attack_location=attack_location,
                    attack_increment_mw=float(
                        attack_increment_mw
                    ),
                    region_a_p_mw=region_a_p_mw,
                    region_b_p_mw=region_b_p_mw,
                    minimum_voltage_limit_pu=(
                        minimum_voltage_limit_pu
                    ),
                    maximum_voltage_limit_pu=(
                        maximum_voltage_limit_pu
                    ),
                )
            )

    results = pd.DataFrame(
        result_records
    )

    threshold_summary = (
        build_threshold_summary(
            results
        )
    )

    # ============================================================
    # 3. Save data and figures
    # ============================================================

    results.to_csv(
        output_directory
        / "step_16_spatial_threshold_sweep_results.csv",
        index=False,
    )

    threshold_summary.to_csv(
        output_directory
        / "step_16_spatial_threshold_summary.csv",
        index=False,
    )

    configuration = {
        "baseline_region_demand_mw": (
            baseline_region_demand_mw
        ),
        "load_power_factor": load_power_factor,
        "minimum_voltage_limit_pu": (
            minimum_voltage_limit_pu
        ),
        "maximum_voltage_limit_pu": (
            maximum_voltage_limit_pu
        ),
        "attack_increment_start_mw": float(
            attack_increments_mw[0]
        ),
        "attack_increment_end_mw": float(
            attack_increments_mw[-1]
        ),
        "attack_increment_step_mw": 0.5,
        "attack_locations": [
            "Region A",
            "Region B",
        ],
    }

    (
        output_directory
        / "step_16_spatial_threshold_config.json"
    ).write_text(
        json.dumps(
            configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_figures(
        results=results,
        output_directory=output_directory,
        minimum_voltage_limit_pu=(
            minimum_voltage_limit_pu
        ),
    )

    # ============================================================
    # 4. Print results
    # ============================================================

    print_results(
        results=results,
        threshold_summary=threshold_summary,
    )

    print()
    print("Files saved:")
    print(
        "  results/"
        "step_16_spatial_threshold_sweep_results.csv"
    )
    print(
        "  results/"
        "step_16_spatial_threshold_summary.csv"
    )
    print(
        "  results/"
        "step_16_spatial_threshold_config.json"
    )
    print(
        "  results/"
        "step_16_minimum_voltage_thresholds.png"
    )
    print(
        "  results/"
        "step_16_line_0_thresholds.png"
    )
    print(
        "  results/"
        "step_16_line_1_thresholds.png"
    )
    print(
        "  results/"
        "step_16_maximum_line_loading.png"
    )
    print(
        "  results/"
        "step_16_line_losses.png"
    )
    print(
        "  results/"
        "step_16_region_b_voltage.png"
    )


if __name__ == "__main__":
    main()
