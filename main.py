from pathlib import Path

import numpy as np
import plotly.graph_objects as go


OUTPUT_DIRECTORY = Path("docs")
X_LIMITS = (-100, 100)
Y_LIMITS = (-2, 2)
INITIAL_X_RANGE = (-1, 4)
SAMPLES_PER_UNIT = 200
ADAPTIVE_X_TICK_SCRIPT = """
const xAxis = document.getElementById('{plot_id}');
const tickIntervalForRange = (range) => {
    const span = range[1] - range[0];
    if (span <= 10) return 0.5;
    if (span <= 20) return 1;
    if (span <= 50) return 2;
    if (span <= 100) return 5;
    return 10;
};
const updateXTickInterval = (range) => {
    const dtick = tickIntervalForRange(range);
    if (xAxis.layout.xaxis.dtick !== dtick) {
        Plotly.relayout(xAxis, {'xaxis.dtick': dtick});
    }
};
updateXTickInterval(xAxis.layout.xaxis.range);
xAxis.on('plotly_relayout', (event) => {
    const left = event['xaxis.range[0]'];
    const right = event['xaxis.range[1]'];
    if (left !== undefined && right !== undefined) {
        updateXTickInterval([left, right]);
    }
});
"""


def periodic_remainder(values: np.ndarray, period: float) -> np.ndarray:
    """Return a non-negative remainder for periodic signal definitions."""
    return np.mod(values, period)


def pulse_train(values: np.ndarray, shift: float = 0) -> np.ndarray:
    return np.where(periodic_remainder(values + shift, 2) < 1, 1, 0)


def square_wave(values: np.ndarray, shift: float = 0) -> np.ndarray:
    return np.where(periodic_remainder(values + shift, 2) < 1, 1, -1)


def sawtooth(values: np.ndarray, shift: float = 0) -> np.ndarray:
    return 2 * (periodic_remainder(values + shift, 1) - 0.5)


def triangle(values: np.ndarray, period: float, shift: float = 0) -> np.ndarray:
    return 2 * np.abs(periodic_remainder(values + shift, period) - period / 2) - 1


def sawtooth_with_gaps(values: np.ndarray) -> np.ndarray:
    phase = periodic_remainder(values, 2)
    return np.where(phase < 1, phase, np.nan)


def trapezoid(values: np.ndarray) -> np.ndarray:
    phase = periodic_remainder(values, 8)
    return np.select(
        [phase < 1, phase < 3, phase < 5],
        [phase, 1, 1 - (phase - 3) / 2],
        default=0,
    )


def absolute_cosine(values: np.ndarray) -> np.ndarray:
    return np.abs(np.cos(np.pi * 2 * values))


def negative_absolute_sine(values: np.ndarray) -> np.ndarray:
    return -np.abs(np.sin(np.pi * 2 * values))


PLOTS = [
    ("pulse-train", "Pulse Train Signal", pulse_train),
    ("shifted-pulse-train", "Shifted Pulse Train Signal", lambda values: pulse_train(values, 0.5)),
    ("square-signal", "Square Signal", square_wave),
    ("shifted-square-signal", "Shifted Square Signal", lambda values: square_wave(values, 0.5)),
    ("sawtooth-signal", "Sawtooth Signal", sawtooth),
    ("shifted-sawtooth-signal", "Shifted Sawtooth Signal", lambda values: sawtooth(values, 0.5)),
    ("triangle-signal", "Triangle Signal", lambda values: 2 * np.abs(periodic_remainder(values, 1) - 1) - 1),
    ("wide-triangle-signal", "Wide Triangle Signal", lambda values: triangle(values, 2)),
    ("shifted-triangle-signal", "Shifted Triangle Signal", lambda values: triangle(values, 2, 1.5)),
    ("sawtooth-with-gaps", "Sawtooth Signal With Gaps", sawtooth_with_gaps),
    ("trapezoid-like-signal", "Trapezoid Like Signal", trapezoid),
    ("cosine-like-signal", "Cosine Like Signal", absolute_cosine),
    ("sine-like-signal", "Sine Like Signal", negative_absolute_sine),
    ("mixed-signal", "Mixed Signal", lambda values: np.zeros_like(values)),
]


def axis_tick_labels(values: np.ndarray, unit: str) -> list[str]:
    labels = []
    for value in values:
        if value == 0:
            labels.append("0")
        elif value.is_integer():
            labels.append(f"{int(value)}{unit}")
        else:
            labels.append(f"{value:g}{unit}")
    return labels


def build_figure(signal) -> go.Figure:
    x_values = np.linspace(*X_LIMITS, (X_LIMITS[1] - X_LIMITS[0]) * SAMPLES_PER_UNIT + 1)
    y_values = signal(x_values)
    y_ticks = np.arange(Y_LIMITS[0], Y_LIMITS[1] + 0.5, 0.5)

    figure = go.Figure(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            line={"color": "#7dd3fc", "width": 2},
            hovertemplate="x: %{x:g}T<br>y: %{y:g}h<extra></extra>",
            name="x(t)",
        )
    )
    figure.update_xaxes(
        range=INITIAL_X_RANGE,
        minallowed=X_LIMITS[0],
        maxallowed=X_LIMITS[1],
        tickmode="linear",
        tick0=0,
        dtick=0.5,
        ticksuffix="T",
        showgrid=True,
        gridcolor="rgba(203, 213, 225, 0.35)",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="#e2e8f0",
        linecolor="#e2e8f0",
        tickfont={"color": "#e2e8f0"},
        title_font={"color": "#e2e8f0"},
        title="T",
    )
    figure.update_yaxes(
        range=Y_LIMITS,
        minallowed=Y_LIMITS[0],
        maxallowed=Y_LIMITS[1],
        tickmode="array",
        tickvals=y_ticks,
        ticktext=axis_tick_labels(y_ticks, "h"),
        showgrid=True,
        gridcolor="rgba(203, 213, 225, 0.35)",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="#e2e8f0",
        linecolor="#e2e8f0",
        tickfont={"color": "#e2e8f0"},
        title_font={"color": "#e2e8f0"},
        title="h",
    )
    figure.update_layout(
        plot_bgcolor="#151d2b",
        paper_bgcolor="#111827",
        font={"color": "#e2e8f0"},
        margin={"l": 70, "r": 30, "b": 60, "t": 60},
        hovermode="x",
    )
    return figure


def write_index(plot_pages: list[tuple[str, str]]) -> None:
    links = "\n".join(f'        <li><a href="{filename}">{title}</a></li>' for filename, title in plot_pages)
    (OUTPUT_DIRECTORY / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Signals and Systems Plots</title>
    <style>
        body {{
            background: #111827;
            color: #e2e8f0;
            font-family: system-ui, sans-serif;
            margin: 2rem;
        }}
        a {{ color: #7dd3fc; }}
    </style>
</head>
<body>
    <main>
        <h1>Signals and Systems Plots</h1>
        <ul>
{links}
        </ul>
    </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    plot_pages = []
    for slug, title, signal in PLOTS:
        filename = f"{slug}.html"
        figure = build_figure(signal)
        figure.write_html(
            OUTPUT_DIRECTORY / filename,
            include_plotlyjs="cdn",
            full_html=True,
            config={"displaylogo": False, "responsive": True},
            post_script=ADAPTIVE_X_TICK_SCRIPT,
        )
        plot_pages.append((filename, title))
    write_index(plot_pages)


if __name__ == "__main__":
    main()
