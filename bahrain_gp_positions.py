#!/usr/bin/env python3
"""Animazione della variazione delle posizioni giro per giro in un GP di F1.

Default: Bahrain Grand Prix 2026, disputato a Sepang (Kuala Lumpur, Malesia),
round 16 del calendario 2026. I dati vengono scaricati tramite FastF1.

Uso:
    pip install fastf1 matplotlib
    python bahrain_gp_positions.py                       # mostra a schermo
    python bahrain_gp_positions.py -o bahrain_2026.mp4   # salva video (serve ffmpeg)
    python bahrain_gp_positions.py -o bahrain_2026.gif   # salva GIF
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


def load_positions(year, event, cache_dir):
    """Restituisce (posizioni, colori, titolo).

    posizioni: DataFrame indicizzato per giro (0 = griglia), una colonna per
    pilota (sigla), valori = posizione; NaN dopo il ritiro.
    """
    import fastf1
    import fastf1.plotting

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(year, event, "R")
    session.load(telemetry=False, weather=False, messages=False)

    laps = session.laps.dropna(subset=["Position"])
    if laps.empty:
        raise SystemExit("Nessun dato di posizione disponibile per questa sessione.")

    pos = laps.pivot_table(index="LapNumber", columns="Driver", values="Position")
    pos.index = pos.index.astype(int)

    # Giro 0 = griglia di partenza (0 in GridPosition = partenza dalla pit lane)
    results = session.results.set_index("Abbreviation")
    grid = results["GridPosition"].reindex(pos.columns).astype(float)
    n_drivers = len(pos.columns)
    pit_start = grid.isna() | (grid <= 0)
    grid[pit_start] = np.arange(n_drivers - pit_start.sum() + 1, n_drivers + 1)
    pos.loc[0] = grid
    pos = pos.sort_index()

    colors = {}
    for drv in pos.columns:
        try:
            colors[drv] = fastf1.plotting.get_driver_color(drv, session)
        except Exception:
            team_color = results.loc[drv, "TeamColor"] if drv in results.index else ""
            colors[drv] = f"#{team_color}" if isinstance(team_color, str) and team_color else "#888888"

    ev = session.event
    title = f"{ev['EventName']} {year} – {ev['Location']}, {ev['Country']}"
    return pos, colors, title


def build_animation(pos, colors, title, steps_per_lap=6, interval=40):
    """Grafico a 'bump chart' che si disegna giro per giro."""
    max_lap = int(pos.index.max())
    n_drivers = len(pos.columns)
    x = np.linspace(0, max_lap, max_lap * steps_per_lap + 1)

    # Interpolazione lineare delle posizioni tra un giro e il successivo.
    series = {}
    for drv in pos.columns:
        s = pos[drv].dropna()
        last_lap = s.index.max()
        y = np.interp(x, s.index.values, s.values)
        y[x > last_lap] = np.nan
        series[drv] = (y, last_lap)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, max_lap + max_lap * 0.08)
    ax.set_ylim(n_drivers + 0.7, 0.3)
    ax.set_yticks(range(1, n_drivers + 1))
    ax.set_xlabel("Giro")
    ax.set_ylabel("Posizione")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.15)

    lines, labels = {}, {}
    for drv in pos.columns:
        (lines[drv],) = ax.plot([], [], color=colors[drv], lw=2.2)
        labels[drv] = ax.text(0, 0, drv, color=colors[drv], fontsize=9,
                              fontweight="bold", va="center", visible=False)
    lap_text = ax.text(0.01, 1.01, "", transform=ax.transAxes, fontsize=12)

    def update(frame):
        cur_x = x[: frame + 1]
        for drv, (y, last_lap) in series.items():
            cur_y = y[: frame + 1]
            lines[drv].set_data(cur_x, cur_y)
            valid = ~np.isnan(cur_y)
            if valid.any():
                i = np.flatnonzero(valid)[-1]
                retired = x[frame] > last_lap and last_lap < max_lap
                labels[drv].set_text(f"{drv} ✕" if retired else drv)
                labels[drv].set_position((cur_x[i] + max_lap * 0.01, cur_y[i]))
                labels[drv].set_visible(True)
        lap = int(np.floor(x[frame]))
        lap_text.set_text("Griglia" if lap == 0 else f"Giro {lap}/{max_lap}")
        return [*lines.values(), *labels.values(), lap_text]

    # Pausa finale: ripete l'ultimo frame per ~2 secondi.
    hold = int(2000 / interval)
    frames = list(range(len(x))) + [len(x) - 1] * hold
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    return fig, anim


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--event", default="Bahrain Grand Prix",
                   help="nome evento o numero di round (default: Bahrain Grand Prix)")
    p.add_argument("-o", "--output", help="file di output .mp4 o .gif; se omesso mostra a schermo")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--steps-per-lap", type=int, default=6,
                   help="fotogrammi per giro (più alto = più fluido e più lento)")
    p.add_argument("--cache", default="fastf1_cache")
    args = p.parse_args()

    event = int(args.event) if args.event.isdigit() else args.event
    pos, colors, title = load_positions(args.year, event, args.cache)
    fig, anim = build_animation(pos, colors, title, args.steps_per_lap, 1000 // args.fps)

    if args.output:
        if args.output.lower().endswith(".gif"):
            writer = PillowWriter(fps=args.fps)
        else:
            writer = FFMpegWriter(fps=args.fps, bitrate=3000)
        anim.save(args.output, writer=writer, dpi=100)
        print(f"Salvato: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
