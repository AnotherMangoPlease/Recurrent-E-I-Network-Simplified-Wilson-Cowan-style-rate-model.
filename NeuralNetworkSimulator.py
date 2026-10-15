"""Interactive recurrent excitatory-inhibitory neural rate network.

This is a compact Wilson-Cowan-style firing-rate model, not a biologically
detailed brain simulation. Each cell has a continuous activity in [0, 1].
For neuron i the update is

    tau * dr_i/dt = -r_i + sigmoid(beta * (I_i - threshold))
    I_i = g_E * sum_j(W_E[i,j] * r_j)
        - g_I * sum_j(W_I[i,j] * r_j) + I_external + pulse_i + noise_i

The nonnegative E and I matrices contain directed sparse random connections,
normalized by the expected number of incoming connections from each
population. Independent Gaussian input noise is sampled at every integration
step. Euler updates are bounded by the sigmoid and use dt <= tau.

Run with ``python neural_network_simulator.py``. Run the built-in model checks
with ``python neural_network_simulator.py --check``.
"""

from __future__ import annotations

import argparse
import csv
import secrets
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelParameters:
    """Parameters that can be changed while the network is running."""

    excitatory_strength: float = 3.0
    inhibitory_strength: float = 4.2
    external_input: float = 1.15
    noise_sd: float = 0.16
    threshold: float = 1.0
    tau: float = 0.18
    stimulus_amplitude: float = 3.5
    stimulus_duration: float = 0.18


class RecurrentRateNetwork:
    """Numerical model only; it has no plotting or GUI dependencies."""

    def __init__(
        self,
        n_neurons: int = 300,
        seed: int = 7,
        parameters: ModelParameters | None = None,
        inhibitory_fraction: float = 0.20,
        connection_probability: float = 0.12,
        dt: float = 0.01,
    ) -> None:
        if not 200 <= n_neurons <= 500:
            raise ValueError("n_neurons must be between 200 and 500")
        if not 0.0 < inhibitory_fraction < 1.0:
            raise ValueError("inhibitory_fraction must be between 0 and 1")
        if not 0.0 < connection_probability <= 1.0:
            raise ValueError("connection_probability must be in (0, 1]")
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.n_neurons = n_neurons
        self.seed = int(seed)
        self.dt = float(dt)
        self.connection_probability = float(connection_probability)
        self.parameters = parameters or ModelParameters()
        self.rng = np.random.default_rng(self.seed)

        self.is_inhibitory = self.rng.random(n_neurons) < inhibitory_fraction
        # Guarantee both populations even with a small valid network.
        self.is_inhibitory[0] = False
        self.is_inhibitory[-1] = True
        self.excitatory_ids = np.flatnonzero(~self.is_inhibitory)
        self.inhibitory_ids = np.flatnonzero(self.is_inhibitory)
        self.n_exc = len(self.excitatory_ids)
        self.n_inh = len(self.inhibitory_ids)

        # W[target, source] makes each row the incoming connections to a cell.
        self.w_exc_unit = self._make_unit_weights(self.excitatory_ids)
        self.w_inh_unit = self._make_unit_weights(self.inhibitory_ids)

        beta = 1.35
        initial = self._sigmoid(beta * (self.parameters.external_input - self.parameters.threshold))
        self.rates = np.clip(initial + self.rng.normal(0.0, 0.015, n_neurons), 0.0, 1.0)
        self.time = 0.0
        self.pulse_ids = np.empty(0, dtype=np.int32)
        self.pulse_end = -1.0
        self.pulse_amplitude = 0.0

    def _make_unit_weights(self, source_ids: np.ndarray) -> np.ndarray:
        n = self.n_neurons
        mask = np.zeros((n, n), dtype=bool)
        mask[:, source_ids] = self.rng.random((n, len(source_ids))) < self.connection_probability
        # The visualized network has no autapses.
        mask[np.arange(n), np.arange(n)] = False
        # Keep expected total synaptic gain nearly independent of network size.
        return mask.astype(np.float32) / (self.connection_probability * len(source_ids))

    @staticmethod
    def _sigmoid(values: float | np.ndarray) -> float | np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))

    def step(self, count: int = 1) -> np.ndarray:
        """Advance count Euler steps and return the latest activity vector."""
        p = self.parameters
        if p.tau < self.dt:
            raise ValueError(f"tau must be at least dt ({self.dt:g} s)")
        leak_step = self.dt / p.tau
        beta = 1.35

        for _ in range(count):
            drive = (
                p.excitatory_strength * (self.w_exc_unit @ self.rates)
                - p.inhibitory_strength * (self.w_inh_unit @ self.rates)
                + p.external_input
                - p.threshold
                + self.rng.normal(0.0, p.noise_sd, self.n_neurons)
            )
            if self.time < self.pulse_end and self.pulse_ids.size:
                drive[self.pulse_ids] += self.pulse_amplitude

            target = self._sigmoid(beta * drive)
            # With dt <= tau this is a convex move toward a value in [0, 1].
            self.rates += leak_step * (target - self.rates)
            np.clip(self.rates, 0.0, 1.0, out=self.rates)
            self.time += self.dt

        return self.rates

    def stimulate(self, neuron_ids: np.ndarray, amplitude: float, duration: float) -> None:
        """Apply a rectangular external-current pulse to the chosen cells."""
        ids = np.asarray(neuron_ids, dtype=np.int32)
        if ids.size == 0:
            raise ValueError("select at least one neuron before stimulating")
        if np.any(ids < 0) or np.any(ids >= self.n_neurons):
            raise ValueError("stimulus neuron index is outside the network")
        self.pulse_ids = np.unique(ids)
        self.pulse_amplitude = max(0.0, float(amplitude))
        self.pulse_end = self.time + max(self.dt, float(duration))


def run_model_checks() -> None:
    """Quick deterministic checks for reproducibility and numerical behavior."""
    params_a = ModelParameters()
    params_b = ModelParameters()
    first = RecurrentRateNetwork(seed=123, parameters=params_a)
    second = RecurrentRateNetwork(seed=123, parameters=params_b)
    assert np.array_equal(first.w_exc_unit, second.w_exc_unit)
    assert np.array_equal(first.w_inh_unit, second.w_inh_unit)
    assert np.array_equal(first.rates, second.rates)
    assert first.n_exc + first.n_inh == 300
    assert first.n_inh > 0 and first.n_exc > 0

    first.step(500)
    second.step(500)
    assert np.array_equal(first.rates, second.rates)
    assert np.all(np.isfinite(first.rates))
    assert np.all((first.rates >= 0.0) & (first.rates <= 1.0))

    quiet = RecurrentRateNetwork(seed=44)
    stimulated = RecurrentRateNetwork(seed=44)
    group = np.arange(20)
    stimulated.stimulate(group, amplitude=5.0, duration=0.20)
    quiet.step(8)
    stimulated.step(8)
    assert stimulated.rates[group].mean() > quiet.rates[group].mean()
    print("Model checks passed: seeded repeatability, bounded rates, and stimulus response.")


class NeuralNetworkApp:
    """Matplotlib widgets and four views for a RecurrentRateNetwork."""

    WINDOW_SECONDS = 12.0
    HEATMAP_SECONDS = 20.0
    GROUP_SIZE = 24
    FRAME_DT = 0.04
    STEPS_PER_FRAME = 4
    N_DISPLAY_EDGES = 1100

    def __init__(self, seed: int = 7) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.collections import LineCollection
        from matplotlib.widgets import Button, Slider, TextBox

        self.plt = plt
        self.FuncAnimation = FuncAnimation
        self.LineCollection = LineCollection
        self.Button = Button
        self.Slider = Slider
        self.TextBox = TextBox

        self.parameters = ModelParameters()
        self.seed = int(seed)
        self.network = RecurrentRateNetwork(seed=self.seed, parameters=self.parameters)
        self.positions = self._make_positions(self.network.n_neurons, self.seed)
        self.selected_center = int(self.network.excitatory_ids[0])
        self.selected_ids = np.empty(0, dtype=np.int32)
        self._select_group(self.selected_center)
        self.paused = False
        self.frame_index = 0
        self.pulse_was_active = False
        self.times: list[float] = [0.0]
        self.rate_samples: list[np.ndarray] = [self.network.rates.copy()]
        view_capacity = int(self.HEATMAP_SECONDS / self.FRAME_DT) + 2
        self.view_times: deque[float] = deque([0.0], maxlen=view_capacity)
        self.view_rates: deque[np.ndarray] = deque([self.network.rates.copy()], maxlen=view_capacity)
        self.figure = plt.figure(figsize=(15.5, 11), facecolor="#f3f6fb")
        self.figure.canvas.manager.set_window_title("Recurrent E/I Neural Network")

        self._build_axes()
        self._build_network_plot()
        self._build_time_plots()
        self._build_controls()
        self._update_metrics()

        self.animation = FuncAnimation(
            self.figure,
            self._update,
            interval=int(self.FRAME_DT * 1000),
            blit=False,
            cache_frame_data=False,
        )
        self.figure.canvas.mpl_connect("button_press_event", self._on_network_click)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

    @staticmethod
    def _make_positions(n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 991)
        angles = rng.uniform(0.0, 2.0 * np.pi, n)
        radii = np.sqrt(rng.uniform(0.05, 1.0, n))
        return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

    def _build_axes(self) -> None:
        fig = self.figure
        fig.text(0.045, 0.962, "RECURRENT E/I NETWORK", color="#17243a", fontsize=18,
                 weight="bold", ha="left", va="top")
        fig.text(0.045, 0.932,
                 "Simplified Wilson–Cowan-style rate model (not a detailed brain simulation)  ·  click a cell to select a group",
                 color="#56657c", fontsize=9.5, ha="left", va="top")
        self.metric_text = fig.text(0.045, 0.895, "", color="#29384e", fontsize=10,
                                    family="monospace", ha="left", va="center")

        self.ax_network = fig.add_axes([0.035, 0.385, 0.46, 0.47], facecolor="#ffffff")
        self.ax_neuron = fig.add_axes([0.535, 0.695, 0.425, 0.16], facecolor="#ffffff")
        self.ax_population = fig.add_axes([0.535, 0.515, 0.425, 0.11], facecolor="#ffffff")
        self.ax_heatmap = fig.add_axes([0.535, 0.355, 0.425, 0.09], facecolor="#ffffff")

        self._style_axis(self.ax_neuron, "Individual neuron activity", "Time (s)", "Rate (0–1)")
        self._style_axis(self.ax_population, "Excitatory and inhibitory activity", "Time (s)", "Mean rate")
        self._style_axis(self.ax_heatmap, "Activity history heatmap", "Time (s)", "Population / cell index")
        self.ax_network.set_title("Animated network  ·  edge brightness follows source activity",
                                  loc="left", color="#17243a", fontsize=10, pad=9)
        self.ax_network.set_xlim(-1.08, 1.08)
        self.ax_network.set_ylim(-1.08, 1.08)
        self.ax_network.set_aspect("equal")
        self.ax_network.set_xticks([])
        self.ax_network.set_yticks([])
        for spine in self.ax_network.spines.values():
            spine.set_visible(False)

    @staticmethod
    def _style_axis(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_title(title, loc="left", color="#17243a", fontsize=10, pad=8)
        ax.set_xlabel(xlabel, color="#53627a", fontsize=8)
        ax.set_ylabel(ylabel, color="#53627a", fontsize=8)
        ax.tick_params(colors="#68778e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#d5deea")
        ax.grid(True, color="#e7edf5", alpha=0.95, linewidth=0.65)

    def _build_network_plot(self) -> None:
        net = self.network
        rng = np.random.default_rng(self.seed + 77)
        target, source = np.nonzero((net.w_exc_unit + net.w_inh_unit) > 0)
        if source.size > self.N_DISPLAY_EDGES:
            chosen = rng.choice(source.size, self.N_DISPLAY_EDGES, replace=False)
            source, target = source[chosen], target[chosen]
        self.edge_source = source.astype(np.int32)
        self.edge_target = target.astype(np.int32)
        self.edge_is_inh = net.is_inhibitory[self.edge_source]
        segments = np.stack((self.positions[self.edge_source], self.positions[self.edge_target]), axis=1)
        self.edge_collection = self.LineCollection(segments, linewidths=0.6, zorder=1)
        self.ax_network.add_collection(self.edge_collection)

        self.exc_points = self.ax_network.scatter(
            self.positions[net.excitatory_ids, 0], self.positions[net.excitatory_ids, 1],
            c=net.rates[net.excitatory_ids], cmap="viridis", vmin=0, vmax=1,
            s=np.full(net.n_exc, 23.0), marker="o", edgecolors="#147f8e", linewidths=0.45,
            zorder=3, label="Excitatory",
        )
        self.inh_points = self.ax_network.scatter(
            self.positions[net.inhibitory_ids, 0], self.positions[net.inhibitory_ids, 1],
            c=net.rates[net.inhibitory_ids], cmap="viridis", vmin=0, vmax=1,
            s=np.full(net.n_inh, 27.0), marker="^", edgecolors="#c8426b", linewidths=0.5,
            zorder=3, label="Inhibitory",
        )
        self.selection_ring = self.ax_network.scatter([], [], s=[], facecolors="none",
                                                       edgecolors="#27364c", linewidths=1.2, zorder=4)
        if hasattr(self, "network_legend"):
            self.network_legend.remove()
        if hasattr(self, "network_note"):
            self.network_note.remove()
        self.network_legend = self.ax_network.legend(loc="upper right", frameon=True, fontsize=8,
                                                     facecolor="#ffffff", edgecolor="#cbd5e2", framealpha=0.95)
        for text in self.network_legend.get_texts():
            text.set_color("#34445b")
        self.network_note = self.ax_network.text(
            0.02, 0.02, "E  circle  ·  I  triangle  ·  cyan edges: excitatory  ·  pink edges: inhibitory",
            color="#56657c", fontsize=7.3, transform=self.ax_network.transAxes,
            ha="left", va="bottom")
        self._refresh_network_artists()

    def _build_time_plots(self) -> None:
        net = self.network
        # Fixed example cells make the raster legend stable across resets.
        self.tracked_ids = np.concatenate((net.excitatory_ids[:3], net.inhibitory_ids[:2]))
        colors = ["#00899b", "#3474cf", "#7855b5", "#c47a24", "#cc4d73"]
        self.neuron_lines = []
        for idx, color in zip(self.tracked_ids, colors):
            kind = "I" if net.is_inhibitory[idx] else "E"
            line, = self.ax_neuron.plot([], [], color=color, lw=1.35, label=f"{kind} #{idx}")
            self.neuron_lines.append(line)
        self.selected_line, = self.ax_neuron.plot([], [], color="#26364e", lw=1.8, ls="--",
                                                   label="Selected cell", visible=False)
        self.ax_neuron.set_ylim(0, 1)
        self.neuron_legend = self.ax_neuron.legend(
            loc="upper right", ncols=3, fontsize=7, facecolor="#ffffff",
            edgecolor="#d4dce8", labelcolor="#34445b", framealpha=0.95)

        self.exc_mean_line, = self.ax_population.plot([], [], color="#00899b", lw=1.8, label="Excitatory")
        self.inh_mean_line, = self.ax_population.plot([], [], color="#cc4d73", lw=1.8, label="Inhibitory")
        self.ax_population.set_ylim(0, 1)
        self.ax_population.legend(loc="upper right", ncols=2, fontsize=7.5, facecolor="#ffffff",
                                  edgecolor="#d4dce8", labelcolor="#34445b", framealpha=0.95)

        self.heat_order = np.concatenate((net.excitatory_ids, net.inhibitory_ids))
        heat = net.rates[self.heat_order, None]
        self.heat_image = self.ax_heatmap.imshow(
            heat, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis",
            vmin=0, vmax=1, extent=(0, 0.04, 0, net.n_neurons),
        )
        self.ax_heatmap.set_yticks([net.n_exc / 2, net.n_exc + net.n_inh / 2])
        self.ax_heatmap.set_yticklabels(["E cells", "I cells"])
        self.ax_heatmap.tick_params(axis="y", labelsize=7.5)
        self.ax_heatmap.set_xlim(0, 1)
        self.ax_heatmap.set_ylim(0, net.n_neurons)

    def _build_controls(self) -> None:
        fig = self.figure
        fig.text(0.045, 0.345, "EXPERIMENT CONTROLS", color="#53627a", fontsize=8.5,
                 weight="bold", ha="left", va="center")
        slider_specs = [
            ("gE recurrent", "excitatory_strength", 0.0, 6.0, 3.0, "%.2f"),
            ("gI inhibition", "inhibitory_strength", 0.0, 8.0, 4.2, "%.2f"),
            ("External input", "external_input", 0.0, 2.5, 1.15, "%.2f"),
            ("Noise SD", "noise_sd", 0.0, 0.8, 0.16, "%.2f"),
            ("Threshold", "threshold", 0.0, 2.5, 1.0, "%.2f"),
            ("Tau (s)", "tau", 0.05, 0.60, 0.18, "%.2f"),
            ("Pulse level", "stimulus_amplitude", 0.0, 8.0, 3.5, "%.2f"),
            ("Pulse duration", "stimulus_duration", 0.05, 0.80, 0.18, "%.2f"),
        ]
        cols_x = [0.045, 0.285, 0.525, 0.765]
        rows_y = [0.235, 0.160]
        self.sliders: dict[str, Any] = {}
        for i, (label, key, low, high, value, fmt) in enumerate(slider_specs):
            col, row = i % 4, i // 4
            ax = fig.add_axes([cols_x[col], rows_y[row], 0.19, 0.022], facecolor="#e9eef5")
            widget = self.Slider(ax, label, low, high, valinit=value, valfmt=fmt,
                                 color="#168fa0", track_color="#d9e2ee")
            widget.label.set_color("#364760")
            widget.label.set_fontsize(8)
            widget.label.set_position((0.0, 1.55))
            widget.label.set_horizontalalignment("left")
            widget.label.set_verticalalignment("bottom")
            widget.valtext.set_color("#203147")
            widget.valtext.set_fontsize(8)
            widget.valtext.set_position((1.0, 1.55))
            widget.valtext.set_horizontalalignment("right")
            widget.valtext.set_verticalalignment("bottom")
            widget.on_changed(lambda _value, param=key: self._read_parameters())
            self.sliders[key] = widget

        buttons = [
            ("Pause", 0.045, 0.075, 0.095, self._toggle_pause),
            ("Reset", 0.150, 0.075, 0.095, self._reset),
            ("Stim pulse", 0.255, 0.075, 0.12, self._stimulate),
            ("Export CSV", 0.385, 0.075, 0.12, self._export_csv),
            ("New seed", 0.695, 0.075, 0.11, self._new_seed),
        ]
        self.pause_button = None
        self.buttons: dict[str, Any] = {}
        for label, x, y, w, callback in buttons:
            ax = fig.add_axes([x, y, w, 0.045])
            button = self.Button(ax, label, color="#e8eef6", hovercolor="#d5e4f5")
            button.label.set_color("#18314f")
            button.label.set_fontsize(8.5)
            button.on_clicked(callback)
            self.buttons[label] = button
            if label == "Pause":
                self.pause_button = button

        fig.text(0.565, 0.095, "Seed", color="#53627a", fontsize=8, ha="right", va="center")
        seed_ax = fig.add_axes([0.57, 0.075, 0.095, 0.045], facecolor="#ffffff")
        self.seed_box = self.TextBox(seed_ax, "", initial=str(self.seed), color="#ffffff",
                                     hovercolor="#eef5fc")
        self.seed_box.label.set_color("#53627a")
        self.seed_box.text_disp.set_color("#203147")
        self.seed_box.on_submit(self._seed_submitted)
        fig.text(0.045, 0.045,
                 "Click the network to select 24 nearby cells. Parameter changes take effect on the next frame.",
                 color="#5d6b80", fontsize=7.7, ha="left", va="center")

    def _read_parameters(self) -> None:
        for key, widget in self.sliders.items():
            setattr(self.parameters, key, float(widget.val))

    def _select_group(self, center: int) -> None:
        self.selected_center = int(center)
        distances = np.sum((self.positions - self.positions[self.selected_center]) ** 2, axis=1)
        self.selected_ids = np.argpartition(distances, self.GROUP_SIZE)[:self.GROUP_SIZE]
        if hasattr(self, "selection_ring"):
            self._refresh_selection_ring()

    def _refresh_selection_ring(self) -> None:
        ids = self.selected_ids
        rates = self.network.rates[ids]
        self.selection_ring.set_offsets(self.positions[ids])
        self.selection_ring.set_sizes(30.0 + 20.0 * rates)

    def _refresh_network_artists(self) -> None:
        rates = self.network.rates
        self.exc_points.set_array(rates[self.network.excitatory_ids])
        self.inh_points.set_array(rates[self.network.inhibitory_ids])
        self.exc_points.set_sizes(18.0 + 26.0 * rates[self.network.excitatory_ids])
        self.inh_points.set_sizes(20.0 + 27.0 * rates[self.network.inhibitory_ids])
        source_rates = rates[self.edge_source]
        base_rgb = np.where(self.edge_is_inh[:, None],
                            np.array([0.78, 0.17, 0.42]), np.array([0.06, 0.48, 0.58]))
        strength = 0.28 + 0.72 * source_rates
        rgba = np.column_stack((base_rgb * strength[:, None], 0.06 + 0.52 * source_rates))
        self.edge_collection.set_color(rgba)
        self.edge_collection.set_linewidths(0.35 + 0.8 * source_rates)
        self._refresh_selection_ring()

    def _on_network_click(self, event: Any) -> None:
        if event.inaxes is not self.ax_network or event.xdata is None or event.ydata is None:
            return
        point = np.array([event.xdata, event.ydata])
        center = int(np.argmin(np.sum((self.positions - point) ** 2, axis=1)))
        self._select_group(center)
        self._set_message(f"Selected {len(self.selected_ids)} cells near neuron #{center}.")
        self.figure.canvas.draw_idle()

    def _update(self, _frame: int) -> tuple[Any, ...]:
        self._read_parameters()
        self.network.step(self.STEPS_PER_FRAME)
        self.frame_index += 1
        self.times.append(self.network.time)
        self.rate_samples.append(self.network.rates.copy())
        self.view_times.append(self.network.time)
        self.view_rates.append(self.network.rates.copy())

        time_values = np.asarray(self.view_times)
        samples = np.asarray(self.view_rates)
        window_start = max(0.0, self.network.time - self.WINDOW_SECONDS)
        visible = time_values >= window_start
        x = time_values[visible]
        visible_rates = samples[visible]
        for line, neuron_id in zip(self.neuron_lines, self.tracked_ids):
            line.set_data(x, visible_rates[:, neuron_id])
        self.selected_line.set_data(x, visible_rates[:, self.selected_center])
        self.selected_line.set_visible(self.selected_center not in self.tracked_ids)
        time_right = max(1.0, self.network.time)
        time_left = max(0.0, time_right - self.WINDOW_SECONDS)
        self.ax_neuron.set_xlim(time_left, time_right)

        self.exc_mean_line.set_data(x, visible_rates[:, self.network.excitatory_ids].mean(axis=1))
        self.inh_mean_line.set_data(x, visible_rates[:, self.network.inhibitory_ids].mean(axis=1))
        self.ax_population.set_xlim(time_left, time_right)

        heat_start = max(0.0, self.network.time - self.HEATMAP_SECONDS)
        heat_visible = time_values >= heat_start
        heat_data = samples[heat_visible][:, self.heat_order].T
        self.heat_image.set_data(heat_data)
        heat_times = time_values[heat_visible]
        heat_left = float(heat_times[0])
        heat_right = max(heat_left + self.FRAME_DT, float(heat_times[-1]))
        self.heat_image.set_extent((heat_left, heat_right, 0, self.network.n_neurons))
        self.ax_heatmap.set_xlim(max(0.0, self.network.time - self.HEATMAP_SECONDS),
                                 max(1.0, self.network.time))

        self._refresh_network_artists()
        self._update_metrics()
        if self.network.time < self.network.pulse_end:
            self._set_message(f"Stimulus active on {len(self.selected_ids)} selected cells.")
        elif getattr(self, "pulse_was_active", False):
            self._set_message("Stimulus pulse ended.")
            self.pulse_was_active = False
        return (self.edge_collection, self.exc_points, self.inh_points, self.selection_ring,
                *self.neuron_lines, self.selected_line, self.exc_mean_line, self.inh_mean_line,
                self.heat_image)

    def _update_metrics(self) -> None:
        rates = self.network.rates
        high = np.mean(rates > 0.70) * 100.0
        variability = np.std(rates)
        self.metric_text.set_text(
            f"t {self.network.time:6.2f} s     mean {rates.mean():.3f}     σ(activity) {variability:.3f}"
            f"     highly active (>0.70) {high:4.1f}%     E {self.network.n_exc} / I {self.network.n_inh}"
        )

    def _set_message(self, message: str) -> None:
        if hasattr(self, "status_text"):
            self.status_text.set_text(message)
        else:
            self.status_text = self.figure.text(0.045, 0.022, message, color="#087d8b",
                                                fontsize=7.6, ha="left", va="center")

    def _toggle_pause(self, _event: Any) -> None:
        if self.paused:
            self.paused = False
            self.pause_button.label.set_text("Pause")
            self.animation.event_source.start()
            self._set_message("Simulation resumed.")
        else:
            self.paused = True
            self.animation.event_source.stop()
            self.pause_button.label.set_text("Resume")
            self._set_message("Simulation paused.")
        self.figure.canvas.draw_idle()

    def _seed_value(self) -> int:
        value = int(self.seed_box.text.strip())
        if not 0 <= value <= 4_294_967_295:
            raise ValueError("Seed must be an integer from 0 to 4294967295.")
        return value

    def _seed_submitted(self, _text: str) -> None:
        try:
            self.seed = self._seed_value()
            self._set_message(f"Seed set to {self.seed}. Press Reset to rebuild the same network.")
        except ValueError as exc:
            self._set_message(str(exc))

    def _reset(self, _event: Any = None) -> None:
        try:
            self.seed = self._seed_value()
        except ValueError as exc:
            self._set_message(str(exc))
            return
        self._read_parameters()
        self.network = RecurrentRateNetwork(seed=self.seed, parameters=self.parameters)
        self.positions = self._make_positions(self.network.n_neurons, self.seed)
        self.times = [0.0]
        self.rate_samples = [self.network.rates.copy()]
        view_capacity = int(self.HEATMAP_SECONDS / self.FRAME_DT) + 2
        self.view_times = deque([0.0], maxlen=view_capacity)
        self.view_rates = deque([self.network.rates.copy()], maxlen=view_capacity)
        self.frame_index = 0
        self.pulse_was_active = False
        self.selected_center = int(self.network.excitatory_ids[0])
        self._select_group(self.selected_center)
        self.tracked_ids = np.concatenate((self.network.excitatory_ids[:3], self.network.inhibitory_ids[:2]))
        # Rebuild collections after the seeded topology and coordinates change.
        self.edge_collection.remove()
        self.exc_points.remove()
        self.inh_points.remove()
        self.selection_ring.remove()
        self._build_network_plot()
        self.heat_order = np.concatenate((self.network.excitatory_ids, self.network.inhibitory_ids))
        self.heat_image.set_data(self.network.rates[self.heat_order, None])
        self.heat_image.set_extent((0, self.FRAME_DT, 0, self.network.n_neurons))
        self.ax_heatmap.set_ylim(0, self.network.n_neurons)
        self.ax_heatmap.set_yticks([self.network.n_exc / 2,
                                    self.network.n_exc + self.network.n_inh / 2])
        self.ax_heatmap.set_yticklabels(["E cells", "I cells"])
        for line, neuron_id in zip(self.neuron_lines, self.tracked_ids):
            line.set_data([], [])
            line.set_label(f"{'I' if self.network.is_inhibitory[neuron_id] else 'E'} #{neuron_id}")
        self.neuron_legend.remove()
        self.neuron_legend = self.ax_neuron.legend(
            loc="upper right", ncols=3, fontsize=7, facecolor="#ffffff",
            edgecolor="#d4dce8", labelcolor="#34445b", framealpha=0.95)
        for line in (self.exc_mean_line, self.inh_mean_line, self.selected_line):
            line.set_data([], [])
        self.selected_line.set_visible(False)
        self.ax_neuron.set_xlim(0, 1)
        self.ax_population.set_xlim(0, 1)
        self.ax_heatmap.set_xlim(0, 1)
        if self.paused:
            self.paused = False
            self.pause_button.label.set_text("Pause")
            self.animation.event_source.start()
        self._update_metrics()
        self._set_message(f"Reset complete. Seed {self.seed}; the same seed reproduces the same network.")
        self.figure.canvas.draw_idle()

    def _new_seed(self, _event: Any) -> None:
        self.seed = secrets.randbelow(4_294_967_296)
        self.seed_box.set_val(str(self.seed))
        self._reset()

    def _stimulate(self, _event: Any) -> None:
        self._read_parameters()
        self.network.stimulate(self.selected_ids, self.parameters.stimulus_amplitude,
                               self.parameters.stimulus_duration)
        self.pulse_was_active = True
        self._set_message(f"Pulse {self.parameters.stimulus_amplitude:.2f} for "
                          f"{self.parameters.stimulus_duration:.2f} s on {len(self.selected_ids)} cells.")
        self.figure.canvas.draw_idle()

    def _export_csv(self, _event: Any) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            path = filedialog.asksaveasfilename(
                parent=root,
                title="Export recorded neural activity",
                defaultextension=".csv",
                initialfile=f"neural_activity_seed_{self.seed}.csv",
                filetypes=[("CSV files", "*.csv")],
            )
            if not path:
                return
            self._write_csv(path)
            self._set_message(f"Exported {len(self.times)} samples × {self.network.n_neurons} neurons to {path}.")
            messagebox.showinfo("CSV export complete", f"Saved simulation data to:\n{path}", parent=root)
        except OSError as exc:
            self._set_message(f"CSV export failed: {exc}")
            messagebox.showerror("CSV export failed", str(exc), parent=root)
        finally:
            root.destroy()

    def _write_csv(self, path: str | Path) -> None:
        with Path(path).open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            header = ["time_s"] + [f"neuron_{idx}_{'I' if self.network.is_inhibitory[idx] else 'E'}"
                                   for idx in range(self.network.n_neurons)]
            header += ["mean_excitatory", "mean_inhibitory"]
            writer.writerow(header)
            for t, rates in zip(self.times, self.rate_samples):
                writer.writerow([f"{t:.4f}", *[f"{value:.6f}" for value in rates],
                                 f"{rates[self.network.excitatory_ids].mean():.6f}",
                                 f"{rates[self.network.inhibitory_ids].mean():.6f}"])

    def _on_close(self, _event: Any) -> None:
        animation = getattr(self, "animation", None)
        event_source = getattr(animation, "event_source", None)
        if event_source is not None:
            event_source.stop()

    def show(self) -> None:
        self.plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive E/I recurrent rate network")
    parser.add_argument("--seed", type=int, default=7, help="initial random seed (default: 7)")
    parser.add_argument("--check", action="store_true", help="run quick numerical checks and exit")
    args = parser.parse_args(argv)
    if args.check:
        run_model_checks()
        return 0

    try:
        app = NeuralNetworkApp(seed=args.seed)
        app.show()
    except ImportError as exc:
        print("Could not start the graphical interface. Install the requirements with:", file=sys.stderr)
        print("  python -m pip install -r requirements.txt", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
