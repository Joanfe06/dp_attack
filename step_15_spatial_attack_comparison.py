from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import pandapower as pp
except ImportError as exc:
    raise SystemExit(
        "pandapower is not installed in the active virtual environment.\n"
        "Install it with:\n"
        "    pip install pandapower\n"
    ) from exc


@dataclass(frozen=True)
class Scenario:
    """One spatial load-attack scenario."""

    name: str
    region_a_p_mw: float
    region_b_p_mw: float


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
    Build the same radial 20 kV network used in Step 14.

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
        name="Three-bus spatial attack comparison",
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

    # Line 0 carries the demand of both regions.
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

    # Line 1 carries only the demand located at region B.
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
) -> None:
    """Run a balanced AC Newton-Raphson power flow."""
    pp.runpp(
        network,
        algorithm="nr",
        calculate_voltage_angles=False,
        init="flat",
        max_iteration=30,
        tolerance_mva=1e-8,
        numba=False,
    )

    if not network.converged:
        raise RuntimeError(
            "The AC power flow did not converge."
        )


def extract_scenario_results(
    *,
    scenario: Scenario,
    network: pp.pandapowerNet,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Extract summary, bus, and line results for one scenario."""
    bus_results = network.res_bus.copy()
    bus_results.insert(
        0,
        "scenario",
        scenario.name,
    )
    bus_results.insert(
        1,
        "bus_name",
        network.bus["name"].to_numpy(),
    )
    bus_results.insert(
        2,
        "bus_index",
        network.bus.index.to_numpy(),
    )

    line_results = network.res_line.copy()
    line_results.insert(
        0,
        "scenario",
        scenario.name,
    )
    line_results.insert(
        1,
        "line_name",
        network.line["name"].to_numpy(),
    )
    line_results.insert(
        2,
        "line_index",
        network.line.index.to_numpy(),
    )

    minimum_voltage_pu = float(
        network.res_bus["vm_pu"].min()
    )

    minimum_voltage_bus = int(
        network.res_bus["vm_pu"].idxmin()
    )

    line_0_loading_percent = float(
        network.res_line.at[
            0,
            "loading_percent",
        ]
    )

    line_1_loading_percent = float(
        network.res_line.at[
            1,
            "loading_percent",
        ]
    )

    total_line_losses_mw = float(
        network.res_line["pl_mw"].sum()
    )

    external_grid_active_power_mw = float(
        network.res_ext_grid.at[
            0,
            "p_mw",
        ]
    )

    external_grid_reactive_power_mvar = float(
        network.res_ext_grid.at[
            0,
            "q_mvar",
        ]
    )

    summary = {
        "scenario": scenario.name,
        "region_a_demand_mw": scenario.region_a_p_mw,
        "region_b_demand_mw": scenario.region_b_p_mw,
        "total_load_mw": (
            scenario.region_a_p_mw
            + scenario.region_b_p_mw
        ),
        "minimum_voltage_pu": minimum_voltage_pu,
        "minimum_voltage_bus": minimum_voltage_bus,
        "line_0_loading_percent": line_0_loading_percent,
        "line_1_loading_percent": line_1_loading_percent,
        "maximum_line_loading_percent": float(
            network.res_line[
                "loading_percent"
            ].max()
        ),
        "total_line_losses_mw": total_line_losses_mw,
        "external_grid_active_power_mw": (
            external_grid_active_power_mw
        ),
        "external_grid_reactive_power_mvar": (
            external_grid_reactive_power_mvar
        ),
        "voltage_violation": bool(
            np.any(
                (
                    network.res_bus["vm_pu"]
                    < 0.95
                )
                | (
                    network.res_bus["vm_pu"]
                    > 1.05
                )
            )
        ),
        "line_overload": bool(
            np.any(
                network.res_line[
                    "loading_percent"
                ]
                > 100.0
            )
        ),
        "converged": bool(network.converged),
    }

    return summary, bus_results, line_results


def add_changes_from_baseline(
    summary_results: pd.DataFrame,
) -> pd.DataFrame:
    """Add scenario changes relative to the baseline."""
    baseline_row = (
        summary_results[
            summary_results["scenario"]
            == "Baseline"
        ]
        .iloc[0]
    )

    result = summary_results.copy()

    result[
        "minimum_voltage_change_pu"
    ] = (
        result["minimum_voltage_pu"]
        - float(
            baseline_row[
                "minimum_voltage_pu"
            ]
        )
    )

    result[
        "line_0_loading_change_percentage_points"
    ] = (
        result["line_0_loading_percent"]
        - float(
            baseline_row[
                "line_0_loading_percent"
            ]
        )
    )

    result[
        "line_1_loading_change_percentage_points"
    ] = (
        result["line_1_loading_percent"]
        - float(
            baseline_row[
                "line_1_loading_percent"
            ]
        )
    )

    result[
        "loss_change_mw"
    ] = (
        result["total_line_losses_mw"]
        - float(
            baseline_row[
                "total_line_losses_mw"
            ]
        )
    )

    return result


def print_comparison(
    summary_results: pd.DataFrame,
) -> None:
    """Print an interpretable comparison table."""
    print(
        "=== Spatial false-price attack comparison ==="
    )
    print()
    print(
        "The same additional active demand is placed either "
        "at Region A or at downstream Region B."
    )
    print()

    display_columns = [
        "scenario",
        "region_a_demand_mw",
        "region_b_demand_mw",
        "minimum_voltage_pu",
        "line_0_loading_percent",
        "line_1_loading_percent",
        "total_line_losses_mw",
        "voltage_violation",
        "line_overload",
    ]

    print(
        summary_results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("Changes relative to baseline")

    change_columns = [
        "scenario",
        "minimum_voltage_change_pu",
        "line_0_loading_change_percentage_points",
        "line_1_loading_change_percentage_points",
        "loss_change_mw",
    ]

    print(
        summary_results[
            change_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    attack_a_row = (
        summary_results[
            summary_results["scenario"]
            == "Attack at Region A"
        ]
        .iloc[0]
    )

    attack_b_row = (
        summary_results[
            summary_results["scenario"]
            == "Attack at Region B"
        ]
        .iloc[0]
    )

    print()
    print("Direct spatial comparison")
    print(
        "  Minimum voltage under attack at A: "
        f"{attack_a_row['minimum_voltage_pu']:.4f} p.u."
    )
    print(
        "  Minimum voltage under attack at B: "
        f"{attack_b_row['minimum_voltage_pu']:.4f} p.u."
    )
    print(
        "  Difference caused by placing the same attack downstream: "
        f"{attack_b_row['minimum_voltage_pu'] - attack_a_row['minimum_voltage_pu']:.4f} p.u."
    )
    print(
        "  Line 1 loading under attack at A: "
        f"{attack_a_row['line_1_loading_percent']:.2f}%"
    )
    print(
        "  Line 1 loading under attack at B: "
        f"{attack_b_row['line_1_loading_percent']:.2f}%"
    )


def save_results(
    *,
    output_directory: Path,
    summary_results: pd.DataFrame,
    all_bus_results: pd.DataFrame,
    all_line_results: pd.DataFrame,
    networks: dict[str, pp.pandapowerNet],
) -> None:
    """Save tables, networks, and comparison figures."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_results.to_csv(
        output_directory
        / "step_15_spatial_attack_summary.csv",
        index=False,
    )

    all_bus_results.to_csv(
        output_directory
        / "step_15_spatial_attack_bus_results.csv",
        index=False,
    )

    all_line_results.to_csv(
        output_directory
        / "step_15_spatial_attack_line_results.csv",
        index=False,
    )

    for scenario_name, network in networks.items():
        safe_name = (
            scenario_name.lower()
            .replace(" ", "_")
        )

        pp.to_json(
            network,
            output_directory
            / f"step_15_{safe_name}.json",
        )

    scenario_names = (
        summary_results["scenario"]
        .tolist()
    )

    # Figure 1: minimum voltage.
    plt.figure(figsize=(9, 5))

    plt.bar(
        scenario_names,
        summary_results[
            "minimum_voltage_pu"
        ],
    )

    plt.axhline(
        0.95,
        linestyle="--",
        label="Lower preliminary limit",
    )

    plt.xlabel("Scenario")
    plt.ylabel("Minimum voltage (p.u.)")
    plt.title(
        "Minimum Bus Voltage by Attack Location"
    )
    plt.xticks(rotation=10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_15_minimum_voltage.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Figure 2: line loading grouped by scenario.
    x_positions = np.arange(
        len(scenario_names)
    )
    bar_width = 0.36

    plt.figure(figsize=(10, 5))

    plt.bar(
        x_positions - bar_width / 2.0,
        summary_results[
            "line_0_loading_percent"
        ],
        width=bar_width,
        label="Line 0: Grid to Region A",
    )

    plt.bar(
        x_positions + bar_width / 2.0,
        summary_results[
            "line_1_loading_percent"
        ],
        width=bar_width,
        label="Line 1: Region A to Region B",
    )

    plt.axhline(
        100.0,
        linestyle="--",
        label="Thermal limit",
    )

    plt.xticks(
        x_positions,
        scenario_names,
        rotation=10,
    )

    plt.xlabel("Scenario")
    plt.ylabel("Line loading (%)")
    plt.title(
        "Line Loading by Attack Location"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_15_line_loading_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Figure 3: voltage at each bus in each scenario.
    bus_names = (
        all_bus_results[
            all_bus_results["scenario"]
            == "Baseline"
        ]["bus_name"]
        .tolist()
    )

    plt.figure(figsize=(10, 5))

    for scenario_name in scenario_names:
        scenario_bus_results = (
            all_bus_results[
                all_bus_results["scenario"]
                == scenario_name
            ]
            .sort_values("bus_index")
        )

        plt.plot(
            bus_names,
            scenario_bus_results[
                "vm_pu"
            ],
            marker="o",
            label=scenario_name,
        )

    plt.axhline(
        0.95,
        linestyle="--",
        label="Lower preliminary limit",
    )

    plt.axhline(
        1.05,
        linestyle="--",
        label="Upper preliminary limit",
    )

    plt.xlabel("Bus")
    plt.ylabel("Voltage magnitude (p.u.)")
    plt.title(
        "Voltage Profile by Attack Location"
    )
    plt.xticks(rotation=15)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_15_voltage_profiles.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Figure 4: active line losses.
    plt.figure(figsize=(9, 5))

    plt.bar(
        scenario_names,
        summary_results[
            "total_line_losses_mw"
        ],
    )

    plt.xlabel("Scenario")
    plt.ylabel("Active line losses (MW)")
    plt.title(
        "Network Losses by Attack Location"
    )
    plt.xticks(rotation=10)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_directory
        / "step_15_line_losses.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    baseline_region_demand_mw = 15.0

    # This value comes from the earlier parameter sweep:
    # approximately 50% attacked EVs with a strong false-price signal.
    attack_increment_mw = 3.42

    load_power_factor = 0.95

    scenarios = [
        Scenario(
            name="Baseline",
            region_a_p_mw=baseline_region_demand_mw,
            region_b_p_mw=baseline_region_demand_mw,
        ),
        Scenario(
            name="Attack at Region A",
            region_a_p_mw=(
                baseline_region_demand_mw
                + attack_increment_mw
            ),
            region_b_p_mw=baseline_region_demand_mw,
        ),
        Scenario(
            name="Attack at Region B",
            region_a_p_mw=baseline_region_demand_mw,
            region_b_p_mw=(
                baseline_region_demand_mw
                + attack_increment_mw
            ),
        ),
    ]

    summary_records: list[
        dict[str, object]
    ] = []

    bus_result_frames: list[
        pd.DataFrame
    ] = []

    line_result_frames: list[
        pd.DataFrame
    ] = []

    networks: dict[
        str,
        pp.pandapowerNet,
    ] = {}

    for scenario in scenarios:
        network = build_three_bus_network(
            region_a_p_mw=(
                scenario.region_a_p_mw
            ),
            region_b_p_mw=(
                scenario.region_b_p_mw
            ),
            load_power_factor=(
                load_power_factor
            ),
        )

        run_power_flow(network)

        (
            summary,
            bus_results,
            line_results,
        ) = extract_scenario_results(
            scenario=scenario,
            network=network,
        )

        summary_records.append(summary)
        bus_result_frames.append(
            bus_results
        )
        line_result_frames.append(
            line_results
        )
        networks[scenario.name] = network

    summary_results = pd.DataFrame(
        summary_records
    )

    summary_results = add_changes_from_baseline(
        summary_results
    )

    all_bus_results = pd.concat(
        bus_result_frames,
        ignore_index=True,
    )

    all_line_results = pd.concat(
        line_result_frames,
        ignore_index=True,
    )

    print(
        f"Attack increment: "
        f"{attack_increment_mw:.2f} MW"
    )
    print(
        f"Load power factor: "
        f"{load_power_factor:.2f}"
    )
    print()

    print_comparison(
        summary_results
    )

    output_directory = Path("results")

    save_results(
        output_directory=output_directory,
        summary_results=summary_results,
        all_bus_results=all_bus_results,
        all_line_results=all_line_results,
        networks=networks,
    )

    print()
    print("Files saved:")
    print(
        "  results/"
        "step_15_spatial_attack_summary.csv"
    )
    print(
        "  results/"
        "step_15_spatial_attack_bus_results.csv"
    )
    print(
        "  results/"
        "step_15_spatial_attack_line_results.csv"
    )
    print(
        "  results/"
        "step_15_baseline.json"
    )
    print(
        "  results/"
        "step_15_attack_at_region_a.json"
    )
    print(
        "  results/"
        "step_15_attack_at_region_b.json"
    )
    print(
        "  results/"
        "step_15_minimum_voltage.png"
    )
    print(
        "  results/"
        "step_15_line_loading_comparison.png"
    )
    print(
        "  results/"
        "step_15_voltage_profiles.png"
    )
    print(
        "  results/"
        "step_15_line_losses.png"
    )


if __name__ == "__main__":
    main()
