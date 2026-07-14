from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandapower as pp
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
    """Calculate inductive reactive power from active power and power factor."""
    if active_power_mw < 0:
        raise ValueError("Active load power cannot be negative.")

    if not 0 < power_factor <= 1:
        raise ValueError(
            "Power factor must be greater than 0 and no greater than 1."
        )

    angle_rad = np.arccos(power_factor)
    return float(active_power_mw * np.tan(angle_rad))


def build_three_bus_network() -> pp.pandapowerNet:
    """
    Build a small radial 20 kV network.

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
        name="Three-bus residential network",
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

    region_a_active_power_mw = 15.0
    region_b_active_power_mw = 15.0
    load_power_factor = 0.95

    pp.create_load(
        network,
        bus=bus_region_a,
        p_mw=region_a_active_power_mw,
        q_mvar=reactive_power_from_power_factor(
            active_power_mw=region_a_active_power_mw,
            power_factor=load_power_factor,
        ),
        name="Region A fixed demand at 18:00",
    )

    pp.create_load(
        network,
        bus=bus_region_b,
        p_mw=region_b_active_power_mw,
        q_mvar=reactive_power_from_power_factor(
            active_power_mw=region_b_active_power_mw,
            power_factor=load_power_factor,
        ),
        name="Region B fixed demand at 18:00",
    )

    return network


def run_power_flow(
    network: pp.pandapowerNet,
) -> None:
    """Run a balanced AC power flow."""
    pp.runpp(
        network,
        algorithm="nr",
        calculate_voltage_angles=False,
        init="flat",
        max_iteration=30,
        tolerance_mva=1e-8,
    )

    if not network.converged:
        raise RuntimeError(
            "The AC power flow did not converge."
        )


def print_input_data(
    network: pp.pandapowerNet,
) -> None:
    """Print the network data supplied to the solver."""
    print("=== Three-bus AC power-flow simulation ===")
    print()
    print("Network topology")
    print("  External grid -> Region A -> Region B")
    print()

    print("Buses")
    print(
        network.bus[
            [
                "name",
                "vn_kv",
                "in_service",
            ]
        ].to_string()
    )
    print()

    print("Lines")
    print(
        network.line[
            [
                "name",
                "from_bus",
                "to_bus",
                "length_km",
                "r_ohm_per_km",
                "x_ohm_per_km",
                "max_i_ka",
            ]
        ].to_string()
    )
    print()

    print("Loads")
    print(
        network.load[
            [
                "name",
                "bus",
                "p_mw",
                "q_mvar",
            ]
        ].to_string()
    )


def print_power_flow_results(
    network: pp.pandapowerNet,
) -> None:
    """Print the most important power-flow outputs."""
    print()
    print("=== Power-flow results ===")
    print(f"Converged: {network.converged}")
    print()

    bus_results = network.res_bus.copy()
    bus_results.insert(
        0,
        "name",
        network.bus["name"],
    )

    print("Bus results")
    print(
        bus_results[
            [
                "name",
                "vm_pu",
                "va_degree",
                "p_mw",
                "q_mvar",
            ]
        ].to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )
    print()

    line_results = network.res_line.copy()
    line_results.insert(
        0,
        "name",
        network.line["name"],
    )

    print("Line results")
    print(
        line_results[
            [
                "name",
                "p_from_mw",
                "q_from_mvar",
                "pl_mw",
                "i_ka",
                "loading_percent",
            ]
        ].to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )
    print()

    ext_grid_results = network.res_ext_grid.copy()
    ext_grid_results.insert(
        0,
        "name",
        network.ext_grid["name"],
    )

    print("External-grid supply")
    print(
        ext_grid_results[
            [
                "name",
                "p_mw",
                "q_mvar",
            ]
        ].to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )
    print()

    minimum_voltage_pu = float(
        network.res_bus["vm_pu"].min()
    )

    maximum_line_loading_percent = float(
        network.res_line[
            "loading_percent"
        ].max()
    )

    total_active_load_mw = float(
        network.load["p_mw"].sum()
    )

    total_reactive_load_mvar = float(
        network.load["q_mvar"].sum()
    )

    total_active_line_losses_mw = float(
        network.res_line["pl_mw"].sum()
    )

    print("Summary")
    print(
        f"  Total active load: "
        f"{total_active_load_mw:.2f} MW"
    )
    print(
        f"  Total reactive load: "
        f"{total_reactive_load_mvar:.2f} MVAr"
    )
    print(
        f"  Total active line losses: "
        f"{total_active_line_losses_mw:.4f} MW"
    )
    print(
        f"  Minimum bus voltage: "
        f"{minimum_voltage_pu:.4f} p.u."
    )
    print(
        f"  Maximum line loading: "
        f"{maximum_line_loading_percent:.2f}%"
    )

    voltage_violation = bool(
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
    )

    line_overload = bool(
        np.any(
            network.res_line[
                "loading_percent"
            ]
            > 100.0
        )
    )

    print(
        f"  Voltage outside [0.95, 1.05] p.u.: "
        f"{voltage_violation}"
    )
    print(
        f"  Line loading above 100%: "
        f"{line_overload}"
    )


def save_results(
    network: pp.pandapowerNet,
    output_directory: Path,
) -> None:
    """Save result tables and figures."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    bus_results = network.res_bus.copy()
    bus_results.insert(
        0,
        "name",
        network.bus["name"],
    )

    line_results = network.res_line.copy()
    line_results.insert(
        0,
        "name",
        network.line["name"],
    )

    ext_grid_results = network.res_ext_grid.copy()
    ext_grid_results.insert(
        0,
        "name",
        network.ext_grid["name"],
    )

    bus_results.to_csv(
        output_directory
        / "step_14_bus_results.csv",
        index_label="bus",
    )

    line_results.to_csv(
        output_directory
        / "step_14_line_results.csv",
        index_label="line",
    )

    ext_grid_results.to_csv(
        output_directory
        / "step_14_ext_grid_results.csv",
        index_label="external_grid",
    )

    pp.to_json(
        network,
        output_directory
        / "step_14_three_bus_network.json",
    )

    plt.figure(figsize=(8, 4))
    plt.bar(
        network.bus["name"],
        network.res_bus["vm_pu"],
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
    plt.title("Bus Voltages")
    plt.xticks(rotation=15)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory
        / "step_14_bus_voltages.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(
        network.line["name"],
        network.res_line[
            "loading_percent"
        ],
    )
    plt.axhline(
        100.0,
        linestyle="--",
        label="Thermal limit",
    )
    plt.xlabel("Line")
    plt.ylabel("Loading (%)")
    plt.title("Line Loading")
    plt.xticks(rotation=15)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory
        / "step_14_line_loadings.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    bus_x = np.array(
        [0.0, 1.0, 2.0]
    )
    bus_y = np.array(
        [0.0, 0.0, 0.0]
    )

    plt.figure(figsize=(9, 3))
    plt.plot(
        bus_x,
        bus_y,
        marker="o",
    )

    for bus_index in network.bus.index:
        plt.text(
            bus_x[bus_index],
            bus_y[bus_index] + 0.08,
            network.bus.at[
                bus_index,
                "name",
            ],
            ha="center",
        )

    for line_index in network.line.index:
        from_bus = int(
            network.line.at[
                line_index,
                "from_bus",
            ]
        )
        to_bus = int(
            network.line.at[
                line_index,
                "to_bus",
            ]
        )

        midpoint_x = (
            bus_x[from_bus]
            + bus_x[to_bus]
        ) / 2.0

        plt.text(
            midpoint_x,
            -0.08,
            (
                f"{network.res_line.at[line_index, 'loading_percent']:.1f}%"
            ),
            ha="center",
        )

    plt.xlabel("Radial feeder position")
    plt.title(
        "Three-Bus Network "
        "(labels below lines show loading)"
    )
    plt.yticks([])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        output_directory
        / "step_14_network_topology.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    network = build_three_bus_network()

    print_input_data(network)

    run_power_flow(network)

    print_power_flow_results(network)

    output_directory = Path("results")

    save_results(
        network=network,
        output_directory=output_directory,
    )

    print()
    print("Files saved:")
    print(
        "  results/step_14_bus_results.csv"
    )
    print(
        "  results/step_14_line_results.csv"
    )
    print(
        "  results/step_14_ext_grid_results.csv"
    )
    print(
        "  results/step_14_three_bus_network.json"
    )
    print(
        "  results/step_14_bus_voltages.png"
    )
    print(
        "  results/step_14_line_loadings.png"
    )
    print(
        "  results/step_14_network_topology.png"
    )


if __name__ == "__main__":
    main()
