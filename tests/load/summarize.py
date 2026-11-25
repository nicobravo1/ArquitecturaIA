"""Resume un run de Locust (``--csv PREFIJO``) en Markdown + gráficos PNG para el informe.

    python tests/load/summarize.py tests/load/reports/steady

Genera ``PREFIJO_summary.md``, ``PREFIJO_latency.png`` y ``PREFIJO_throughput.png``
(dos gráficos separados: latencia y throughput tienen escalas distintas).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def summarize(prefix: str) -> str:
    stats = pd.read_csv(f"{prefix}_stats.csv")
    rows = []
    for _, r in stats.iterrows():
        n = int(r["Request Count"])
        rows.append({
            "Endpoint": "**Total**" if r["Name"] == "Aggregated" else f"{r['Type']} {r['Name']}".replace("POST POST", "POST").replace("GET GET", "GET"),
            "Requests": n,
            "Fallas": int(r["Failure Count"]),
            "Tasa error": f"{r['Failure Count'] / n:.2%}" if n else "–",
            "RPS": f"{r['Requests/s']:.1f}",
            "p50 ms": int(r["50%"]), "p95 ms": int(r["95%"]), "p99 ms": int(r["99%"]),
            "máx ms": int(r["Max Response Time"]),
        })
    md = [f"# Prueba de carga: `{Path(prefix).name}`", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    hist_path = Path(f"{prefix}_stats_history.csv")
    if hist_path.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h = pd.read_csv(hist_path)
        h = h[h["Name"] == "Aggregated"].copy()
        h["t"] = h["Timestamp"] - h["Timestamp"].min()
        for col in ("50%", "95%", "99%"):
            h[col] = pd.to_numeric(h[col], errors="coerce")
        blue, ink, grid = "#257de0", "#666666", "#e7e7e7"

        def style(ax, ylabel):
            ax.set_xlabel("Tiempo (s)", color=ink); ax.set_ylabel(ylabel, color=ink)
            ax.grid(color=grid, linewidth=0.8); ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=ink)

        fig, ax = plt.subplots(figsize=(8, 3.2))
        for col, color in (("50%", "#84b9f3"), ("95%", "#257de0"), ("99%", "#0e3d7e")):   # rampa secuencial
            ax.plot(h["t"], h[col], color=color, linewidth=2, label=f"p{col[:-1]}")
        ax.legend(frameon=False, ncols=3, loc="upper left")
        style(ax, "Latencia (ms)"); fig.tight_layout()
        fig.savefig(f"{prefix}_latency.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(h["t"], h["Requests/s"], color=blue, linewidth=2)
        ax.fill_between(h["t"], h["Requests/s"], color=blue, alpha=0.12)
        style(ax, "Requests por segundo"); ax.set_ylim(bottom=0); fig.tight_layout()
        fig.savefig(f"{prefix}_throughput.png", dpi=150); plt.close(fig)
        md += [f"![Latencia]({Path(prefix).name}_latency.png)", "",
               f"![Throughput]({Path(prefix).name}_throughput.png)", ""]
    out = "\n".join(md)
    Path(f"{prefix}_summary.md").write_text(out)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(summarize(sys.argv[1]))
