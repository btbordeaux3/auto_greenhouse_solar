"""
size_optimizer.py — Sizing optimization for 6×4 ft greenhouse.

Tests combinations of real purchasable battery, PV, and inverter sizes,
runs each through the MPC controller, and recommends the optimal combo.

Usage:
    python size_optimizer.py                              # default sweep
    python size_optimizer.py --quick                       # even smaller
    python size_optimizer.py --matlab                      # use Simulink plant
"""

import argparse
import sys
import time
import itertools
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from real_parts import (
    SOLAR_PANELS, BATTERIES as REAL_BATTERIES, INVERTERS as REAL_INVERTERS,
    WaterPump, GrowLight, Sensor, DcDcConverter,
)
from run_twin import run, compare_controllers
from optimizer.stochastic_mpc import (
    P_LIGHTS, P_PUMP, P_CONTROLLER, P_SENSORS, P_COMMS, ALWAYS_ON_W,
    BATTERY_EFF, inverter_efficiency, dc_dc_efficiency,
)

RESULTS_DIR = Path(__file__).parent.parent / 'results'

# ── Component catalog (real purchasable sizes) ──────────────────────────────
BATTERIES = [
    ("DL+12V50Ah",  12.8,  50,   640,   180),
    ("DL+12V100Ah", 12.8, 100,  1280,   300),
    ("DL+12V200Ah", 12.8, 200,  2560,   530),
    ("DL+24V100Ah", 25.6, 100,  2560,   600),
    ("Ren12V100Ah", 12.8, 100,  1280,   280),
    ("Ren12V200Ah", 12.8, 200,  2560,   500),
    ("Ren48V50Ah",  51.2,  50,  2560,   580),
]

PV_PANELS = [
    ("Ren100W",   100,   90),
    ("Ren200W",   200,  170),
    ("Ren300W",   300,  230),
    ("HQST100W",  100,   75),
]

INVERTERS = [
    ("Vic300W",   300,  120),
    ("Vic500W",   500,  170),
    ("Ren1000W", 1000,  300),
]

ALWAYS_ON_W = P_CONTROLLER + P_SENSORS + P_COMMS  # ~9.6 W


# ── Metrics collection ──────────────────────────────────────────────────────
def collect_metrics(df: pd.DataFrame, sim_days: int, cfg: dict) -> dict:
    dt_s = float(df['dt_s'].iloc[0]) if 'dt_s' in df.columns else 300.0
    dt_h = dt_s / 3600.0

    soc          = df['soc_true'].values
    temp_f       = df['temp'].values * 9.0 / 5.0 + 32.0
    p_pv         = df['p_pv'].values
    p_load       = df['p_load'].values
    p_vent       = df['p_vent'].values

    min_soc      = float(np.min(soc))
    mean_soc     = float(np.mean(soc))
    temp_in_band = float(((temp_f >= 80 - 5.4) & (temp_f <= 80 + 5.4)).mean() * 100)

    peak_load_w  = float(np.max(p_load))
    inverter_w   = cfg['inverter_w']
    peak_met     = peak_load_w <= inverter_w

    night_mask = df['is_dark'] == True
    if night_mask.any():
        night_min_soc = float(np.min(soc[night_mask.values]))
    else:
        night_min_soc = min_soc
    overnight_survived = night_min_soc > 0.05

    pump_h = df['u_pump'].sum() * dt_h
    lights_h = df['u_lights'].sum() * dt_h
    pump_ok = abs(pump_h / sim_days - 0.5) < 0.15
    lights_ok = abs(lights_h / sim_days - 4.0) < 0.5

    never_shut_down = min_soc >= 0.05   # hard floor enforced by MPC

    total_pv_kwh = p_pv.sum() * dt_h / 1000.0
    total_load_kwh = p_load.sum() * dt_h / 1000.0

    # MPC cost: sum over the whole simulation
    mpc_cost_total = float(df['mpc_cost'].sum())
    mpc_cost_avg   = float(df['mpc_cost'].mean())

    p_dcdc_loss_w = float(df['p_dcdc_loss'].mean()) if 'p_dcdc_loss' in df.columns else 0.0

    return {
        'min_soc':             min_soc * 100,
        'mean_soc':            mean_soc * 100,
        'night_min_soc':       night_min_soc * 100,
        'temp_in_band_pct':    temp_in_band,
        'peak_load_w':         peak_load_w,
        'peak_met':            peak_met,
        'overnight_survived':  overnight_survived,
        'pump_hours_day':      pump_h / sim_days,
        'lights_hours_day':    lights_h / sim_days,
        'pump_compliant':      pump_ok,
        'lights_compliant':    lights_ok,
        'never_shut_down':     never_shut_down,
        'total_pv_kwh':        total_pv_kwh,
        'total_load_kwh':      total_load_kwh,
        'mpc_cost_total':      mpc_cost_total,
        'mpc_cost_avg':        mpc_cost_avg,
        'always_on_w':         ALWAYS_ON_W,
        'p_dcdc_loss_w':       p_dcdc_loss_w,
    }


def estimate_system_cost(cfg: dict) -> float:
    return cfg['battery_cost'] + cfg['pv_cost'] + cfg['inverter_cost']


# ── Sizing sweep ────────────────────────────────────────────────────────────
def run_sweep(sim_days: int = 2, n_scenarios: int = 1,
              quick: bool = False, verbose: bool = False,
              horizon: int = 12, dt_minutes: int = 15,
              use_matlab: bool = False,
              weather_method: str = 'historical'):
    t_start = time.time()

    batteries = BATTERIES
    pvs       = PV_PANELS
    inverters = INVERTERS

    if quick:
        batteries = [b for b in batteries if b[3] in (1200, 2400)]
        pvs       = [p for p in pvs if p[1] in (200, 400)]
        inverters = [i for i in inverters if i[1] == 300]

    total_combos = len(batteries) * len(pvs) * len(inverters)
    print(f"\n{'='*80}")
    print(f"  Greenhouse Sizing Optimization")
    print(f"  {len(batteries)} batteries x {len(pvs)} PV panels x {len(inverters)} inverters")
    print(f"  = {total_combos} total combinations")
    print(f"  Simulation: {sim_days} day(s), dt={dt_minutes}min, horizon={horizon} steps")
    print(f"  Plant: {'Simulink' if use_matlab else 'Python'}")
    print(f"{'='*80}\n")

    rows = []
    combo_idx = 0
    failed = 0

    for bat_label, bat_v, bat_ah, bat_wh, bat_cost in batteries:
        for pv_label, pv_w, pv_cost in pvs:
            for inv_label, inv_w, inv_cost in inverters:
                combo_idx += 1
                cfg = {
                    'battery_label':  bat_label,
                    'battery_v':      bat_v,
                    'battery_ah':     bat_ah,
                    'battery_wh':     bat_wh,
                    'battery_cost':   bat_cost,
                    'pv_label':       pv_label,
                    'pv_w':           pv_w,
                    'pv_cost':        pv_cost,
                    'inverter_label': inv_label,
                    'inverter_w':     inv_w,
                    'inverter_cost':  inv_cost,
                }
                total_cost = bat_cost + pv_cost + inv_cost

                print(f"  [{combo_idx}/{total_combos}] "
                      f"Bat={bat_label} ({bat_wh}Wh)  "
                      f"PV={pv_label}  "
                      f"Inv={inv_label}  "
                      f"Cost=${total_cost}", end="", flush=True)

                try:
                    hw_label = f'{int(bat_wh)}Wh_{int(pv_w)}Wp_{int(inv_w)}W'
                    hw_dir = RESULTS_DIR / hw_label
                    hw_dir.mkdir(parents=True, exist_ok=True)
                    df = run(
                        sim_days       = sim_days,
                        n_scenarios    = n_scenarios,
                        horizon_steps  = horizon,
                        use_matlab     = use_matlab,
                        weather_method = weather_method,
                        dt_minutes     = dt_minutes,
                        battery_wh     = float(bat_wh),
                        pv_wpeak       = float(pv_w),
                        inverter_w     = float(inv_w),
                        controller     = 'mpc',
                        results_dir    = hw_dir,
                    )
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    failed += 1
                    continue

                metrics = collect_metrics(df, sim_days, cfg)
                metrics['total_cost'] = total_cost
                metrics.update(cfg)
                rows.append(metrics)

                status = "OK" if (metrics['never_shut_down']
                                  and metrics['pump_compliant']
                                  and metrics['lights_compliant']) else "FAIL"
                print(f"  {status} SOC={metrics['min_soc']:.0f}% "
                      f"Tband={metrics['temp_in_band_pct']:.0f}% "
                      f"Pump={metrics['pump_hours_day']:.2f}h "
                      f"Lights={metrics['lights_hours_day']:.1f}h "
                      f"Cost={metrics['mpc_cost_avg']:.0f}")

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"  Completed {combo_idx} combos in {elapsed/60:.1f} min ({failed} failed)")
    print(f"{'='*80}\n")

    if not rows:
        print("  No successful simulations!")
        return

    result_df = pd.DataFrame(rows)
    sweep_dir = RESULTS_DIR / 'sizing_sweep'
    sweep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sweep_dir / 'sizing_results.csv'
    result_df.to_csv(csv_path, index=False)
    print(f"  Results saved => {csv_path}\n")

    generate_report(result_df)
    plot_sizing_results(result_df, save_dir=sweep_dir)


# ── Report generation ───────────────────────────────────────────────────────
def generate_report(df: pd.DataFrame):
    print(f"{'='*100}")
    print(f"  SIZING OPTIMIZATION REPORT")
    print(f"{'='*100}")

    viable = df[
        (df['never_shut_down'] == True)
        & (df['pump_compliant'] == True)
        & (df['lights_compliant'] == True)
    ].copy()

    if len(viable) == 0:
        print("\n  No fully compliant system found. Using partial matches.")
        viable = df.copy()
        viable['score'] = (
            viable['never_shut_down'] * 100
            + viable['temp_in_band_pct'] * 0.5
            - viable['total_cost'] * 0.02
            - viable['mpc_cost_avg'] * 1e-7
        )
    else:
        # Normalize MPC cost and system cost to [0,1] and combine
        mpc_max = viable['mpc_cost_avg'].max()
        mpc_min = viable['mpc_cost_avg'].min()
        cost_max = viable['total_cost'].max()
        cost_min = viable['total_cost'].min()

        def norm(x, lo, hi):
            return (x - lo) / (hi - lo) if hi > lo else 0.5

        viable['mpc_norm'] = viable['mpc_cost_avg'].apply(lambda x: norm(x, mpc_min, mpc_max))
        viable['cost_norm'] = viable['total_cost'].apply(lambda x: norm(x, cost_min, cost_max))

        # Score: lower is better for both MPC cost and system cost
        # Weight: 60% system cost, 40% MPC cost (user wants small+cheap + good control)
        viable['score'] = -(
            0.60 * viable['cost_norm']
            + 0.40 * viable['mpc_norm']
            - 0.05 * viable['temp_in_band_pct'] / 100.0  # small bonus for temp compliance
        )

    viable = viable.sort_values('score', ascending=False)

    print(f"\n  {'Rank':>4}  {'Battery':>12}  {'PV':>7}  {'Inverter':>9}  "
          f"{'Cost':>6}  {'MinSOC':>6}  {'Tband%':>7}  {'Pump':>5}  {'Lghts':>6}  "
          f"{'MPCcost':>9}  {'Score':>6}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*7}  {'-'*9}  "
          f"{'-'*6}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*6}  "
          f"{'-'*9}  {'-'*6}")
    for i, (_, row) in enumerate(viable.head(15).iterrows()):
        mpc_str = f"{row['mpc_cost_avg']:.1e}" if isinstance(row['mpc_cost_avg'], float) else f"{row['mpc_cost_avg']}"
        print(f"  {i+1:>4}  {row['battery_label']:>12}  {row['pv_label']:>7}  "
              f"{row['inverter_label']:>9}  "
              f"${row['total_cost']:>4.0f}  "
              f"{row['min_soc']:>5.1f}%  "
              f"{row['temp_in_band_pct']:>6.1f}%  "
              f"{row['pump_hours_day']:>4.2f}  "
              f"{row['lights_hours_day']:>5.1f}  "
              f"{row['mpc_cost_avg']:>9.1e}  "
              f"{row['score']:>6.2f}")

    best = viable.iloc[0]
    print(f"\n  {'-'*80}")
    print(f"  RECOMMENDED: {best['battery_label']} battery + "
          f"{best['pv_label']} PV + {best['inverter_label']} inverter")
    print(f"    Total cost:   ${best['total_cost']:.0f}")
    print(f"    Battery:      {best['battery_wh']:.0f} Wh ({best['battery_label']})")
    print(f"    PV:           {best['pv_w']:.0f} W ({best['pv_label']})")
    print(f"    Inverter:     {best['inverter_w']:.0f} W ({best['inverter_label']})")
    print(f"    Min SOC:      {best['min_soc']:.1f}%")
    print(f"    Night min:    {best['night_min_soc']:.1f}%")
    print(f"    Temp in band: {best['temp_in_band_pct']:.1f}%")
    print(f"    Pump/day:     {best['pump_hours_day']:.2f}h")
    print(f"    Lights/day:   {best['lights_hours_day']:.1f}h")
    print(f"    MPC cost avg: {best['mpc_cost_avg']:.1e}")
    print(f"  {'-'*80}")

    budget = viable[viable['total_cost'] <= 500].sort_values('score', ascending=False)
    if len(budget) > 0:
        bbest = budget.iloc[0]
        print(f"\n  BEST BUDGET (<$500): {bbest['battery_label']} + {bbest['pv_label']}")
        print(f"    Cost: ${bbest['total_cost']:.0f} | "
              f"Min SOC: {bbest['min_soc']:.1f}% | "
              f"Temp in band: {bbest['temp_in_band_pct']:.1f}% | "
              f"MPC cost: {bbest['mpc_cost_avg']:.1e}")


def plot_sizing_results(df: pd.DataFrame, save_dir: Optional[Path] = None):
    viable = df[df['never_shut_down'] == True].copy()
    if len(viable) == 0:
        viable = df.copy()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Greenhouse Sizing Optimization — All Combos', fontsize=15, fontweight='bold')

    # 1. Cost vs Min SOC scatter (every combo labelled)
    ax = axes[0, 0]
    all_ok = (df['never_shut_down'] == True) & (df['pump_compliant'] == True) & (df['lights_compliant'] == True)
    fail = df[~all_ok]
    ok   = df[all_ok]
    if len(fail):
        ax.scatter(fail['total_cost'], fail['min_soc'], marker='x', c='red',  s=60,  label='Fails constraints')
    if len(ok):
        sc = ax.scatter(ok['total_cost'], ok['min_soc'],
                        c=ok['temp_in_band_pct'], cmap='RdYlGn', s=100, alpha=0.8,
                        edgecolors='k', linewidths=0.5, label='Meets constraints')
        plt.colorbar(sc, ax=ax, label='Temp in band (%)')
    for _, r in df.iterrows():
        ax.annotate(f"{int(r['battery_wh'])}Wh\n{int(r['pv_w'])}W",
                     (r['total_cost'], r['min_soc']), fontsize=5, ha='center', va='bottom')
    ax.set_xlabel('System cost ($)')
    ax.set_ylabel('Minimum SOC (%)')
    ax.set_title('Cost vs Survivability (all combos)')
    ax.axhline(5, color='red', ls='--', lw=1, alpha=0.5, label='Hard floor (5%)')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    # 2. Heatmap: battery x PV (min SOC)
    ax = axes[0, 1]
    pivot_soc = df.pivot_table(values='min_soc', index='pv_w', columns='battery_wh', aggfunc='first')
    if not pivot_soc.empty:
        im = ax.imshow(pivot_soc.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, label='Min SOC (%)')
        for i in range(len(pivot_soc.index)):
            for j in range(len(pivot_soc.columns)):
                val = pivot_soc.values[i, j]
                okf = 'green' if val >= 5 else 'red'
                ax.text(j, i, f"{val:.0f}%", ha='center', va='center', fontsize=9, fontweight='bold', color=okf)
        ax.set_xticks(range(len(pivot_soc.columns)))
        ax.set_xticklabels([f'{c:.0f} Wh' for c in pivot_soc.columns], fontsize=9)
        ax.set_yticks(range(len(pivot_soc.index)))
        ax.set_yticklabels([f'{r:.0f} W' for r in pivot_soc.index], fontsize=9)
        ax.set_xlabel('Battery')
        ax.set_ylabel('PV')
        ax.set_title('Min SOC — Battery x PV')

    # 3. Heatmap: battery x PV (feasibility + temp band)
    ax = axes[0, 2]
    df['feasible'] = (df['never_shut_down'] == True) & (df['pump_compliant'] == True) & (df['lights_compliant'] == True)
    pivot_feas = df.pivot_table(values='temp_in_band_pct', index='pv_w', columns='battery_wh', aggfunc='first')
    if not pivot_feas.empty:
        cmap = plt.cm.RdYlGn
        im = ax.imshow(pivot_feas.values, cmap=cmap, aspect='auto', vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, label='Temp in band (%)')
        for i in range(len(pivot_feas.index)):
            for j in range(len(pivot_feas.columns)):
                val = pivot_feas.values[i, j]
                feas = df[(df['pv_w'] == pivot_feas.index[i]) & (df['battery_wh'] == pivot_feas.columns[j])]['feasible'].values
                mark = '✓' if len(feas) > 0 and feas[0] else '✗'
                ax.text(j, i, f"{val:.0f}%\n{mark}", ha='center', va='center', fontsize=8, fontweight='bold')
        ax.set_xticks(range(len(pivot_feas.columns)))
        ax.set_xticklabels([f'{c:.0f} Wh' for c in pivot_feas.columns], fontsize=9)
        ax.set_yticks(range(len(pivot_feas.index)))
        ax.set_yticklabels([f'{r:.0f} W' for r in pivot_feas.index], fontsize=9)
        ax.set_xlabel('Battery')
        ax.set_ylabel('PV')
        ax.set_title('Temp compliance + Feasibility (✓/✗)')

    # 4. Bar: Top 10 by score
    ax = axes[1, 0]
    if 'mpc_norm' not in viable.columns:
        viable['score'] = (
            viable['never_shut_down'] * 100
            + viable['temp_in_band_pct'] * 0.5
            - viable['total_cost'] * 0.02
            - viable['mpc_cost_avg'] * 1e-7
        )
    top = viable.sort_values('score', ascending=False).head(10)
    labels = [f"{r['battery_wh']:.0f}Wh\n{r['pv_w']:.0f}W\n${r['total_cost']:.0f}" for _, r in top.iterrows()]
    colors = ['green' if r['never_shut_down'] else 'red' for _, r in top.iterrows()]
    ax.bar(range(len(top)), top['score'], color=colors, alpha=0.7)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Score')
    ax.set_title('Top 10 Configurations (by score)')
    ax.grid(True, alpha=0.3)

    # 5. Cost breakdown for top 5
    ax = axes[1, 1]
    top_config = viable.sort_values('score', ascending=False).head(5)
    x = np.arange(len(top_config))
    width = 0.25
    ax.bar(x - width, top_config['battery_cost'], width, label='Battery', alpha=0.8)
    ax.bar(x, top_config['pv_cost'], width, label='PV', alpha=0.8)
    ax.bar(x + width, top_config['inverter_cost'], width, label='Inverter', alpha=0.8)
    ax.set_xticks(x)
    labels = [f"{r['battery_label']}\n{r['pv_label']}" for _, r in top_config.iterrows()]
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Cost ($)')
    ax.set_title('Cost Breakdown — Top 5')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Bar: cheapest feasible
    ax = axes[1, 2]
    feasible = df[(df['never_shut_down'] == True) & (df['pump_compliant'] == True) & (df['lights_compliant'] == True)].copy()
    if len(feasible) > 0:
        cheapest = feasible.sort_values('total_cost').head(8)
        labels = [f"{r['battery_wh']:.0f}Wh\n{r['pv_w']:.0f}W\n{r['inverter_label']}" for _, r in cheapest.iterrows()]
        ax.bar(range(len(cheapest)), cheapest['total_cost'], color='royalblue', alpha=0.7)
        ax.set_xticks(range(len(cheapest)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('System cost ($)')
        ax.set_title('Cheapest Feasible Systems')
        ax.grid(True, alpha=0.3)
        for i, (_, r) in enumerate(cheapest.iterrows()):
            ax.text(i, r['total_cost'] + 10, f"${r['total_cost']:.0f}", ha='center', fontsize=8, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No feasible systems found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Cheapest Feasible Systems')
    plt.tight_layout()
    out_dir = Path(save_dir) if save_dir else RESULTS_DIR
    out = out_dir / 'sizing_optimization.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Plot saved => {out}")
    plt.close()


# ── Comparison sweep ────────────────────────────────────────────────────────
def run_comparison_sweep(sim_days: int = 2, dt_minutes: int = 30,
                         n_scenarios: int = 3, quick: bool = False,
                         weather_method: str = 'historical',
                         use_matlab: bool = False):
    """Run controller comparison across a subset of hardware combos."""
    batteries = BATTERIES if not quick else [b for b in BATTERIES if b[3] in (1200, 2400)]
    pvs       = PV_PANELS if not quick else [p for p in PV_PANELS if p[1] in (200, 400)]
    inverters = INVERTERS if not quick else [i for i in INVERTERS if i[1] == 300]

    print(f"\n{'='*90}")
    print(f"  SIZING + CONTROLLER COMPARISON")
    print(f"  {len(batteries)} batteries x {len(pvs)} PV x {len(inverters)} inverters")
    print(f"  x 3 controllers = {len(batteries)*len(pvs)*len(inverters)*3} total runs")
    print(f"  Simulation: {sim_days} day(s), dt={dt_minutes}min")
    print(f"{'='*90}\n")

    all_results = []

    for bat in batteries:
        for pv in pvs:
            for inv in inverters:
                bat_label, bat_v, bat_ah, bat_wh, bat_cost = bat
                pv_label, pv_w, pv_cost = pv
                inv_label, inv_w, inv_cost = inv
                total_cost = bat_cost + pv_cost + inv_cost

                print(f"\n  ── {bat_label} ({bat_wh}Wh) + {pv_label} + {inv_label} (${total_cost}) ──")
                hw_label = f'{int(bat_wh)}Wh_{int(pv_w)}Wp_{int(inv_w)}W'
                hw_dir = RESULTS_DIR / hw_label
                hw_dir.mkdir(parents=True, exist_ok=True)
                try:
                    comp = compare_controllers(
                        sim_days=sim_days,
                        dt_minutes=dt_minutes,
                        battery_wh=float(bat_wh),
                        pv_wpeak=float(pv_w),
                        inverter_w=float(inv_w),
                        n_scenarios=n_scenarios,
                        weather_method=weather_method,
                        use_matlab=use_matlab,
                        results_dir=hw_dir,
                    )
                    for ctrl, metrics in comp.items():
                        all_results.append({
                            'battery': bat_label,
                            'battery_wh': bat_wh,
                            'pv': pv_label,
                            'pv_w': pv_w,
                            'inverter': inv_label,
                            'inverter_w': inv_w,
                            'total_cost': total_cost,
                            'controller': ctrl,
                            'mean_soc': metrics['mean_soc'],
                            'min_soc': metrics['min_soc'],
                            'temp_in_band': metrics['temp_in_band'],
                            'pump_h': metrics['pump_h'],
                            'lights_h': metrics['lights_h'],
                            'total_pv_kwh': metrics['total_pv_kwh'],
                            'max_temp_f': metrics['max_temp_f'],
                            'min_temp_f': metrics['min_temp_f'],
                        })
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    continue

    # ── Final best-combo summary ─────────────────────────────────────────
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df['feasible'] = (
            (summary_df['min_soc'] >= 5)
            & (summary_df['pump_h'].between(0.35, 0.65))
            & (summary_df['lights_h'].between(3.5, 4.5))
        )
        feasible = summary_df[summary_df['feasible']]
        pool = feasible if len(feasible) > 0 else summary_df
        pool = pool.copy()
        cost_min, cost_max = pool['total_cost'].min(), pool['total_cost'].max()
        temp_min, temp_max = pool['temp_in_band'].min(), pool['temp_in_band'].max()
        def norm(x, lo, hi): return (x - lo) / (hi - lo) if hi > lo else 0.5
        pool['score'] = (
            0.5 * (1 - pool['total_cost'].apply(lambda x: norm(x, cost_min, cost_max)))
            + 0.3 * pool['temp_in_band'].apply(lambda x: norm(x, temp_min, temp_max)) / 100
            + 0.2 * pool['mean_soc'] / 100
        )
        pool = pool.sort_values('score', ascending=False)

        print(f"\n{'='*100}")
        print(f"  FINAL RANKING — Best combos across all controllers & hardware")
        print(f"{'='*100}")
        print(f"  {'Rank':>4}  {'Controller':>14}  {'Battery':>14}  {'PV':>7}  {'Inv':>6}  "
              f"{'Cost':>5}  {'MinSOC':>6}  {'Tband%':>7}  {'Score':>6}")
        print(f"  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*7}  {'-'*6}  "
              f"{'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}")
        for i, (_, row) in enumerate(pool.head(10).iterrows()):
            print(f"  {i+1:>4}  {row['controller']:>14}  {row['battery']:>14}  "
                  f"{row['pv']:>7}  {row['inverter_w']:>4.0f}W  "
                  f"${row['total_cost']:>3.0f}  "
                  f"{row['min_soc']:>5.1f}%  "
                  f"{row['temp_in_band']:>6.1f}%  "
                  f"{row['score']:>6.3f}")

        best = pool.iloc[0]
        print(f"\n  {'★' if len(feasible) > 0 else ' '}  BEST OVERALL: "
              f"{best['battery']} + {best['pv']} + {best['inverter']} "
              f"(${best['total_cost']:.0f})  —  "
              f"Controller: {best['controller']}")
        print(f"     Min SOC: {best['min_soc']:.1f}%  |  "
              f"Temp in band: {best['temp_in_band']:.1f}%  |  "
              f"Score: {best['score']:.3f}")
        if len(feasible) == 0 and len(summary_df) > 0:
            print(f"     Note: No combo met all constraints — showing best partial match")
        print(f"{'='*100}\n")


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Greenhouse Sizing Optimization')
    parser.add_argument('--sim-days',  type=int, default=2,     help='Simulation days per combo')
    parser.add_argument('--scenarios', type=int, default=1,     help='Weather scenarios')
    parser.add_argument('--horizon',   type=int, default=12,    help='MPC horizon steps')
    parser.add_argument('--dt',        type=int, default=30,    help='Time step minutes (default 30 for speed; use 15 for precision)')
    parser.add_argument('--quick',     action='store_true',      help='Reduced sweep (3x2x1=6 combos)')
    parser.add_argument('--verbose',   action='store_true',      help='Verbose MPC output')
    parser.add_argument('--matlab',    action='store_true',      help='Use MATLAB/Simulink plant')
    parser.add_argument('--compare',   action='store_true',      help='Run controller comparison across sizings')
    parser.add_argument('--weather',   default='historical',     help="Weather method: 'gp' or 'historical'")
    args = parser.parse_args()

    if args.compare:
        run_comparison_sweep(
            sim_days=args.sim_days,
            dt_minutes=args.dt,
            n_scenarios=args.scenarios,
            quick=args.quick,
            weather_method=args.weather,
            use_matlab=args.matlab,
        )
    else:
        run_sweep(
            sim_days     = args.sim_days,
            n_scenarios  = args.scenarios,
            quick        = args.quick,
            verbose      = args.verbose,
            horizon      = args.horizon,
            dt_minutes   = args.dt,
            use_matlab   = args.matlab,
            weather_method = args.weather,
        )
