# =============================================================================
# Black-Litterman Portfolio Optimisation
# Evolution from MPT (Sharpe) → PMPT (Sortino) → BL (Views-Blended Returns)
#
# Tickers  : TCS, RELIANCE, HDFCBANK, HINDUNILVR, CIPLA, POWERGRID, TATASTEEL, MARUTI
# Benchmark: ^NSEI (Nifty 50)
# Views    : RELIANCE  +5% above implied equilibrium
#            HDFCBANK  +6% above implied equilibrium
#            POWERGRID +6% above implied equilibrium
#            TATASTEEL +6% above implied equilibrium
#
# Outputs (all saved to BL_Outputs/):
#   bl_1_prior_vs_posterior.png       Prior vs Posterior expected returns
#   bl_2_weight_comparison.png        Market-cap vs BL weights (bar)
#   bl_3_allocation_donut.png         BL allocation donut + bar
#   bl_4_efficient_frontier.html      BL frontier (Plotly dark)
#   bl_5_cumulative_performance.png   BL portfolio vs constituents (10yr)
#   bl_6_drawdown.html                Underwater drawdown (Plotly dark)
#   bl_7_metrics_table.csv            Full metrics table
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from scipy.optimize import minimize

try:
    import yfinance as yf
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)
    logging.getLogger('peewee').setLevel(logging.CRITICAL)
except ImportError:
    raise ImportError("pip install yfinance  then re-run.")

np.random.seed(42)

# =============================================================================
# 1. CONSTANTS
# =============================================================================
TICKERS   = [
    'TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS',
    'CIPLA.NS', 'POWERGRID.NS', 'TATASTEEL.NS', 'MARUTI.NS'
]
BENCHMARK = '^NSEI'
N         = len(TICKERS)
T         = 252          # trading days
RF        = 0.06         # risk-free / MAR
TAU       = 0.05         # BL scaling scalar (uncertainty in prior)
DELTA     = 2.5          # market risk-aversion coefficient
WEIGHT_THRESHOLD = 0.001
N_SIM     = 5000

SHORT = {
    'TCS.NS':'TCS', 'RELIANCE.NS':'RELIANCE', 'HDFCBANK.NS':'HDFCBANK',
    'HINDUNILVR.NS':'HINDUNILVR', 'CIPLA.NS':'CIPLA',
    'POWERGRID.NS':'POWERGRID', 'TATASTEEL.NS':'TATASTEEL', 'MARUTI.NS':'MARUTI'
}
COLORS = {
    'TCS.NS':'#B5D4F4', 'RELIANCE.NS':'#FAC775', 'HDFCBANK.NS':'#9FE1CB',
    'HINDUNILVR.NS':'#F4C0D1', 'CIPLA.NS':'#C8B4E8',
    'POWERGRID.NS':'#1D9E75', 'TATASTEEL.NS':'#F4A261', 'MARUTI.NS':'#378ADD'
}
BG   = '#F9F8F5'
DARK = '#1C1C1E'
RED  = '#E24B4A'
TEAL = '#1D9E75'
BLUE = '#378ADD'

OUTPUT_DIR = 'BL_Outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Outputs → {os.path.abspath(OUTPUT_DIR)}\n")

def op(fname): return os.path.join(OUTPUT_DIR, fname)

# =============================================================================
# 2. DATA
# =============================================================================
print("Downloading 10 years of data...")
raw = yf.download(TICKERS + [BENCHMARK], period='10y',
                  auto_adjust=True, progress=False)
prices_all = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
prices_all = prices_all.ffill().dropna()

prices    = prices_all[TICKERS]
ret_all   = prices_all.pct_change().dropna()
returns   = ret_all[TICKERS]          # daily returns (N_DAYS × 8)
bench_ret = ret_all[BENCHMARK]

print(f"  {len(returns)} trading days  "
      f"({returns.index[0].date()} → {returns.index[-1].date()})")

# Annualised sample covariance
Sigma = returns.cov().values * T      # (8×8)

# =============================================================================
# 3. MARKET-CAP EQUILIBRIUM WEIGHTS (approximate free-float NSE weights)
#    These proxy as the "market portfolio" for implied returns.
#    Adjust if you have more precise data.
# =============================================================================
mcap_weights_raw = np.array([
    0.18,   # TCS
    0.22,   # RELIANCE
    0.16,   # HDFCBANK
    0.08,   # HINDUNILVR
    0.06,   # CIPLA
    0.09,   # POWERGRID
    0.11,   # TATASTEEL
    0.10,   # MARUTI
])
w_mkt = mcap_weights_raw / mcap_weights_raw.sum()   # normalise to 1

# =============================================================================
# 4. BLACK-LITTERMAN MATHEMATICS
# =============================================================================

# ── Step 1: Implied equilibrium excess returns  Π = δ · Σ · w_mkt ────────────
Pi = DELTA * Sigma @ w_mkt          # (8,) — prior expected excess returns

# ── Step 2: Investor Views ────────────────────────────────────────────────────
#   View 1: RELIANCE  outperforms its equilibrium by +5%  → absolute view
#   View 2: HDFCBANK  outperforms its equilibrium by +6%  → absolute view
#   View 3: POWERGRID outperforms its equilibrium by +6%  → absolute view
#   View 4: TATASTEEL outperforms its equilibrium by +6%  → absolute view
#
#   Each row of P is a portfolio that expresses one view.
#   Absolute views: one 1 in the relevant column, 0 elsewhere.
#
VIEW_TICKERS  = ['RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'TATASTEEL.NS']
VIEW_EXCESS   = np.array([0.15, 0.25, 0.05, 0.05])   # view alphas (excess over Π_i)

K = len(VIEW_TICKERS)               # number of views
P = np.zeros((K, N))                # picking matrix (K × N)
Q = np.zeros(K)                     # view returns vector

for k, vt in enumerate(VIEW_TICKERS):
    col = TICKERS.index(vt)
    P[k, col] = 1.0                 # absolute view on asset 'col'
    Q[k] = Pi[col] + VIEW_EXCESS[k] # view = equilibrium + investor conviction

# ── Step 3: Uncertainty of views  Ω = diag(P · τΣ · Pᵀ) ────────────────────
#   Standard He-Litterman proportional specification:
Omega = np.diag(np.diag(P @ (TAU * Sigma) @ P.T))    # (K × K)

# ── Step 4: BL posterior expected returns ─────────────────────────────────────
#   μ_BL = [ (τΣ)⁻¹ + Pᵀ Ω⁻¹ P ]⁻¹ · [ (τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q ]
tauSigma     = TAU * Sigma
tauSigma_inv = np.linalg.inv(tauSigma)
Omega_inv    = np.linalg.inv(Omega)

M_inv = tauSigma_inv + P.T @ Omega_inv @ P     # (N × N)
M     = np.linalg.inv(M_inv)
mu_BL = M @ (tauSigma_inv @ Pi + P.T @ Omega_inv @ Q)   # (N,) posterior

# ── Step 5: Posterior covariance ──────────────────────────────────────────────
#   Σ_BL = Σ + M   (He-Litterman full posterior)
Sigma_BL = Sigma + M                                     # (N × N)

# Add RF to convert excess → total expected returns
mu_BL_total = mu_BL + RF
Pi_total    = Pi    + RF

print("\n=== BLACK-LITTERMAN EXPECTED RETURNS ===")
print(f"{'Asset':<16} {'Prior Π+RF':>12} {'BL Posterior':>13} {'Δ':>9}")
print("-" * 55)
for i, t in enumerate(TICKERS):
    delta_r = (mu_BL_total[i] - Pi_total[i]) * 100
    print(f"{SHORT[t]:<16} {Pi_total[i]*100:>11.2f}%  "
          f"{mu_BL_total[i]*100:>12.2f}%  {delta_r:>+8.2f}%")

# =============================================================================
# 5. OPTIMIZER  (Mean-Variance on BL posterior)
# =============================================================================
class BlackLittermanOptimizer:
    """
    Mean-Variance optimiser using BL posterior returns and covariance.
    Maximises the Sharpe Ratio (using RF as threshold, consistent with BL framework).
    Long-only, fully-invested.
    """
    def __init__(self, mu, cov, rf=0.06):
        self.mu  = mu    # (N,) total expected returns
        self.cov = cov   # (N×N) covariance
        self.rf  = rf
        self.n   = len(mu)

    def port_return(self, w):
        return float(w @ self.mu)

    def port_vol(self, w):
        return float(np.sqrt(w @ self.cov @ w))

    def sharpe(self, w):
        r, v = self.port_return(w), self.port_vol(w)
        return (r - self.rf) / v if v > 1e-9 else -np.inf

    def _neg_sharpe(self, w):
        return -self.sharpe(w)

    def optimise(self):
        w0  = np.full(self.n, 1.0 / self.n)
        res = minimize(
            self._neg_sharpe, w0, method='SLSQP',
            bounds=[(0.0, 1.0)] * self.n,
            constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
            options={'ftol': 1e-12, 'maxiter': 2000}
        )
        w = np.where(res.x < WEIGHT_THRESHOLD, 0.0, res.x)
        self.optimal_weights = w / w.sum()
        return self.optimal_weights

    def simulate(self, n=5000):
        records = []
        for _ in range(n):
            w = np.random.dirichlet(np.ones(self.n))
            records.append({
                'Return'  : self.port_return(w) * 100,
                'Volatility': self.port_vol(w)  * 100,
                'Sharpe'  : self.sharpe(w),
            })
        return pd.DataFrame(records)

print("\nRunning BL optimisation (Max Sharpe on posterior)...")
bl_opt = BlackLittermanOptimizer(mu_BL_total, Sigma_BL, rf=RF)
W_bl   = bl_opt.optimise()
sim_df = bl_opt.simulate(N_SIM)

bl_ret = bl_opt.port_return(W_bl)
bl_vol = bl_opt.port_vol(W_bl)
bl_sr  = bl_opt.sharpe(W_bl)

print(f"\n  Expected Return : {bl_ret*100:.2f}%")
print(f"  Volatility      : {bl_vol*100:.2f}%")
print(f"  Sharpe Ratio    : {bl_sr:.4f}")
print(f"\n  Optimal weights:")
for t, w in zip(TICKERS, W_bl):
    if w >= WEIGHT_THRESHOLD:
        print(f"    {SHORT[t]:<14}  {w*100:.2f}%")

# =============================================================================
# 6. METRICS TABLE  → bl_7_metrics_table.csv
# =============================================================================
rows = []
for i, t in enumerate(TICKERS):
    rows.append({
        'Asset'          : t,
        'Prior Return'   : round(Pi_total[i], 4),
        'BL Return'      : round(mu_BL_total[i], 4),
        'Return Delta'   : round(mu_BL_total[i] - Pi_total[i], 4),
        'Volatility'     : round(np.sqrt(Sigma_BL[i, i]), 4),
        'Sharpe (BL)'    : round((mu_BL_total[i] - RF) / np.sqrt(Sigma_BL[i, i]), 4),
        'BL Weight'      : round(W_bl[i], 4),
        'Market Weight'  : round(w_mkt[i], 4),
    })
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(op('bl_7_metrics_table.csv'), index=False)
print(f"\nMetrics saved → bl_7_metrics_table.csv")

# =============================================================================
# HELPER: axis styling
# =============================================================================
def style_ax(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color='#DDDBD4', lw=0.6, ls='--', alpha=0.8)
    for sp in ['top','right']:  ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']:ax.spines[sp].set_color('#DDDBD4')
    ax.tick_params(colors='#666', labelsize=9)

# =============================================================================
# 7. PLOT 1 — Prior vs Posterior Expected Returns
# =============================================================================
print("\n[Plot 1] Prior vs Posterior returns...")
short_labels = [SHORT[t] for t in TICKERS]
x = np.arange(N)
w_bar = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
style_ax(ax)

b1 = ax.bar(x - w_bar/2, Pi_total  * 100, w_bar,
            color=BLUE, alpha=0.80, label='Prior (Equilibrium Π + RF)', edgecolor=BG, linewidth=1.2)
b2 = ax.bar(x + w_bar/2, mu_BL_total * 100, w_bar,
            color=TEAL, alpha=0.90, label='BL Posterior', edgecolor=BG, linewidth=1.2)

# value labels on top of bars
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=7.5, color=BLUE, fontweight='bold')
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=7.5, color=TEAL, fontweight='bold')

# annotate the view assets with Δ arrows
for vt, ve in zip(VIEW_TICKERS, VIEW_EXCESS):
    i = TICKERS.index(vt)
    ax.annotate(f'+{ve*100:.0f}% view',
                xy=(i + w_bar/2, mu_BL_total[i]*100),
                xytext=(i + w_bar/2, mu_BL_total[i]*100 + 2.5),
                fontsize=8, color=RED, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=9.5)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.set_ylabel('Expected Annual Return (%)', fontsize=10, color='#555')
ax.set_title('Black-Litterman: Prior Equilibrium vs. Posterior Expected Returns\n'
             'Red annotations show investor views blended into the posterior',
             fontsize=13, fontweight='bold', color=DARK, pad=12)
ax.legend(fontsize=9.5, framealpha=0.9)
ax.axhline(RF*100, color='#AAAAAA', lw=1, ls=':', zorder=1)
ax.text(N-0.5, RF*100+0.3, f'RF = {RF*100:.0f}%', fontsize=8, color='#999')

plt.tight_layout()
plt.savefig(op('bl_1_prior_vs_posterior.png'), dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved → bl_1_prior_vs_posterior.png")

# =============================================================================
# 8. PLOT 2 — Market-cap weights vs BL weights
# =============================================================================
print("[Plot 2] Weight comparison...")
fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor(BG)
style_ax(ax)

b3 = ax.bar(x - w_bar/2, w_mkt  * 100, w_bar,
            color='#B4B2A9', alpha=0.85, label='Market-Cap Weights', edgecolor=BG, lw=1.2)
b4 = ax.bar(x + w_bar/2, W_bl   * 100, w_bar,
            color=[COLORS[t] for t in TICKERS],
            alpha=0.92, label='BL Optimal Weights', edgecolor=BG, lw=1.2)

for bar in b3:
    if bar.get_height() > 0.3:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=7.5, color='#666')
for bar in b4:
    if bar.get_height() > 0.3:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=7.5, color=DARK, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=9.5)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.set_ylabel('Portfolio Weight (%)', fontsize=10, color='#555')
ax.set_title('Market-Cap Weights vs. BL Optimal Weights\n'
             'Shift shows how investor views tilt allocation away from market equilibrium',
             fontsize=13, fontweight='bold', color=DARK, pad=12)
ax.legend(fontsize=9.5, framealpha=0.9)

plt.tight_layout()
plt.savefig(op('bl_2_weight_comparison.png'), dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved → bl_2_weight_comparison.png")

# =============================================================================
# 9. PLOT 3 — Allocation donut + horizontal bar
# =============================================================================
print("[Plot 3] Allocation donut + bar...")
active   = [(t, W_bl[i]) for i, t in enumerate(TICKERS) if W_bl[i] >= WEIGHT_THRESHOLD]
inactive = [(t, W_bl[i]) for i, t in enumerate(TICKERS) if W_bl[i] <  WEIGHT_THRESHOLD]

fig2 = plt.figure(figsize=(14, 7))
fig2.patch.set_facecolor(BG)
gs   = gridspec.GridSpec(1, 2, figure=fig2,
                         left=0.05, right=0.95, top=0.88, bottom=0.12, wspace=0.35)
ax_d = fig2.add_subplot(gs[0, 0])
ax_b = fig2.add_subplot(gs[0, 1])

# ── Donut ─────────────────────────────────────────────────────────────────────
ax_d.set_facecolor(BG)
a_col = [COLORS[t] for t, _ in active]
a_wts = [w for _, w in active]
a_lbl = [SHORT[t] for t, _ in active]

_, _, ats = ax_d.pie(
    a_wts, colors=a_col, autopct='%1.1f%%', pctdistance=0.72,
    startangle=90, wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=3),
    explode=[0.04]*len(active)
)
for at in ats:
    at.set_fontsize(11); at.set_fontweight('bold'); at.set_color('white')

ax_d.add_patch(plt.Circle((0,0), 0.48, color=BG))
ax_d.text(0,  0.10, 'Sharpe', ha='center', fontsize=10, color='#888')
ax_d.text(0, -0.12, f'{bl_sr:.4f}', ha='center',
          fontsize=20, fontweight='bold', color=DARK)

leg_h = [mpatches.Patch(color=a_col[i], label=f'{a_lbl[i]}  {a_wts[i]*100:.2f}%')
         for i in range(len(active))]
ax_d.legend(handles=leg_h, loc='lower center', bbox_to_anchor=(0.5, -0.14),
            ncol=2, fontsize=9.5, framealpha=0, handlelength=1.2)
ax_d.set_title('BL Optimal Allocation\n(Active assets only)',
               fontsize=11, fontweight='bold', color=DARK, pad=10)

# ── Horizontal bar (all assets) ───────────────────────────────────────────────
style_ax(ax_b)
all_lbl = [SHORT[t] for t in TICKERS]
all_wts = [W_bl[i]*100 for i in range(N)]
bar_col = [COLORS[t] if W_bl[i] >= WEIGHT_THRESHOLD else '#D3D1C7'
           for i, t in enumerate(TICKERS)]
edge_c  = [RED if W_bl[i] >= WEIGHT_THRESHOLD else '#BBBBBB'
           for i in range(N)]

bars = ax_b.barh(all_lbl[::-1], all_wts[::-1],
                 color=bar_col[::-1], edgecolor=edge_c[::-1],
                 linewidth=1.2, height=0.60)
for bar, val in zip(bars, all_wts[::-1]):
    lbl = f'{val:.2f}%' if val >= 0.1 else '0.00%  (excluded)'
    ax_b.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
              lbl, va='center', ha='left', fontsize=8.5,
              color=DARK if val >= 0.1 else '#AAAAAA',
              fontweight='bold' if val >= 0.1 else 'normal')

ax_b.set_xlabel('Weight (%)', fontsize=10, color='#555', labelpad=6)
ax_b.set_xlim(0, max(all_wts)*1.35)
ax_b.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax_b.set_title('All Assets — BL Assigned Weights\n(Grey = zeroed by optimiser)',
               fontsize=11, fontweight='bold', color=DARK, pad=10)

pill_y = -0.16
for idx, (lbl, val, col) in enumerate([
    ('Expected Return', f'{bl_ret*100:.2f}%', TEAL),
    ('Volatility',      f'{bl_vol*100:.2f}%', BLUE),
    ('Sharpe Ratio',    f'{bl_sr:.4f}',        RED),
]):
    xc = 0.15 + idx*0.35
    ax_b.text(xc, pill_y,      lbl, ha='center', fontsize=8,
              color='#888', transform=ax_b.transAxes)
    ax_b.text(xc, pill_y-0.07, val, ha='center', fontsize=12,
              fontweight='bold', color=col, transform=ax_b.transAxes)

fig2.suptitle(
    f'Black-Litterman Optimal Portfolio — Asset Allocation\n'
    f'Sharpe = {bl_sr:.4f}   |   Views: RELIANCE +5%, HDFCBANK +6%, POWERGRID +6%, TATASTEEL +6%',
    fontsize=12, fontweight='bold', color=DARK, y=0.98)

plt.savefig(op('bl_3_allocation_donut.png'), dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved → bl_3_allocation_donut.png")

# =============================================================================
# 10. PLOT 4 — Efficient Frontier (Plotly dark, matching PMPT style)
# =============================================================================
print("[Plot 4] Efficient frontier (Plotly)...")
fig_ef = go.Figure()

fig_ef.add_trace(go.Scatter(
    x=sim_df['Volatility'], y=sim_df['Return'],
    mode='markers',
    marker=dict(color=sim_df['Sharpe'], colorscale='Viridis',
                size=4, opacity=0.6,
                colorbar=dict(title='Sharpe Ratio', thickness=14)),
    name='Random Portfolios',
    hovertemplate='Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.3f}<extra></extra>',
))
fig_ef.add_trace(go.Scatter(
    x=[bl_vol*100], y=[bl_ret*100],
    mode='markers',
    marker=dict(symbol='star', size=22, color='red',
                line=dict(width=1.5, color='white')),
    name='BL Max Sharpe',
    hovertemplate=(f'<b>BL Optimal Portfolio</b><br>'
                   f'Vol: {bl_vol*100:.2f}%<br>'
                   f'Return: {bl_ret*100:.2f}%<br>'
                   f'Sharpe: {bl_sr:.4f}<extra></extra>'),
))
fig_ef.add_annotation(
    x=bl_vol*100, y=bl_ret*100,
    text=f'<b>⭐ Max Sharpe: {bl_sr:.2f}</b>',
    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
    arrowcolor='white', ax=70, ay=-55,
    font=dict(size=12, color='white'),
    bgcolor='rgba(200,0,0,0.85)',
    bordercolor='white', borderwidth=1, borderpad=5
)
fig_ef.update_layout(
    title='<b>Black-Litterman Efficient Frontier</b><br>'
          '<sup>5,000 random portfolios · BL posterior returns & covariance</sup>',
    xaxis_title='Annual Volatility (%)',
    yaxis_title='Expected Return (%)',
    template='plotly_dark', width=950, height=620,
    showlegend=False,
)
fig_ef.write_html(op('bl_4_efficient_frontier.html'))
print("  Saved → bl_4_efficient_frontier.html")

# =============================================================================
# 11. PLOT 5 — Cumulative Performance (10-year, real returns)
# =============================================================================
print("[Plot 5] Cumulative performance...")

port_dr  = returns.dot(W_bl)
port_cum = (1 + port_dr).cumprod()
asset_cum= (1 + returns).cumprod()
final_v  = port_cum.iloc[-1]
total_r  = (final_v - 1) * 100
n_days   = len(port_cum)
mar_line = (1 + RF / T) ** np.arange(n_days)

fig3, ax3 = plt.subplots(figsize=(15, 7.5))
fig3.patch.set_facecolor(BG)
ax3.set_facecolor(BG)
fig3.subplots_adjust(left=0.07, right=0.80, top=0.88, bottom=0.10)
style_ax(ax3)

# inactive (dashed, faded)
for t, _ in inactive:
    ax3.plot(asset_cum.index, asset_cum[t],
             color=COLORS[t], lw=0.9, alpha=0.30, ls='--', zorder=2)
# active (solid)
for t, _ in active:
    ax3.plot(asset_cum.index, asset_cum[t],
             color=COLORS[t], lw=1.8, alpha=0.90, ls='-', zorder=4)
# portfolio
ax3.plot(port_cum.index, port_cum,
         color=RED, lw=3.8, zorder=8, solid_capstyle='round')
# baselines
ax3.axhline(1.0, color='#AAAAAA', lw=0.9, ls=':', zorder=1)
ax3.plot(port_cum.index, mar_line,
         color='#3B6D11', lw=1.4, ls='-.', alpha=0.75, zorder=3)
ax3.text(port_cum.index[8], 1.04, '$1 baseline', fontsize=7.5, color='#AAAAAA')

# ── right-side labels with nudge algorithm ────────────────────────────────────
label_data = [{'y': final_v,      'text': f'BL Portfolio  ${final_v:.2f}',
               'color': RED,      'bold': True,  'size': 9.5},
              {'y': mar_line[-1], 'text': f'RF/MAR 6%  ${mar_line[-1]:.2f}',
               'color': '#3B6D11','bold': False, 'size': 8.0}]
for t, _ in active:
    yv = asset_cum[t].iloc[-1]
    label_data.append({'y': yv, 'text': f'{SHORT[t]}  ${yv:.2f}',
                       'color': COLORS[t], 'bold': True, 'size': 8.0})
for t, _ in inactive:
    yv = asset_cum[t].iloc[-1]
    label_data.append({'y': yv, 'text': f'{SHORT[t]}  ${yv:.2f}',
                       'color': '#BBBBBB', 'bold': False, 'size': 7.5})

label_data.sort(key=lambda d: d['y'], reverse=True)

# dynamic min-gap
ax3.set_xlim(returns.index[0], returns.index[-1])
ymin3, ymax3 = ax3.get_ylim()
MIN_GAP = max((ymax3 - ymin3) * 0.042, 0.12)

placed = []
def nudge(y):
    for py in placed:
        if abs(y - py) < MIN_GAP:
            y = py - MIN_GAP
    placed.append(y)
    return y

last_date = returns.index[-1]
for ld in label_data:
    y_orig   = ld['y']
    y_placed = nudge(y_orig)
    ax3.plot(last_date, y_orig, 'o', color=ld['color'],
             markersize=4, zorder=9, markeredgewidth=0, clip_on=False)
    if abs(y_placed - y_orig) > 0.01:
        ax3.annotate('',
            xy=(last_date, y_orig), xytext=(last_date, y_placed),
            arrowprops=dict(arrowstyle='-', color=ld['color'], lw=0.6, alpha=0.5),
            clip_on=False)
    ax3.annotate(ld['text'],
        xy=(last_date, y_placed),
        xytext=(10, 0), textcoords='offset points',
        fontsize=ld['size'], fontweight='bold' if ld['bold'] else 'normal',
        color=ld['color'], va='center', ha='left', clip_on=False)

# legend
handles5 = [
    mpatches.Patch(color=RED,       label=f'BL Portfolio (SR={bl_sr:.2f})'),
    mpatches.Patch(color='#3B6D11', label='RF/MAR 6% p.a.'),
    mpatches.Patch(color='#555',    label='Active assets — solid'),
    mpatches.Patch(color='#BBBBBB', label='Excluded — dashed'),
]
ax3.legend(handles=handles5, fontsize=8.5, framealpha=0.92,
           loc='upper left', handlelength=1.4, borderpad=0.8)

ax3.set_xlabel('Date', fontsize=10, color='#555', labelpad=6)
ax3.set_ylabel('Value per $1 invested', fontsize=10, color='#555', labelpad=6)
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax3.get_xticklabels(), rotation=0, ha='center')

fig3.suptitle(
    'Black-Litterman Portfolio — Cumulative Performance vs. All Constituents\n'
    f'$1 Invested · 10-Year Backtest · '
    f'{returns.index[0].strftime("%b %Y")} – {returns.index[-1].strftime("%b %Y")}',
    fontsize=13, fontweight='bold', color=DARK, y=0.97)

metrics_str = (
    f"BL Sharpe: {bl_sr:.4f}   |   E[Return]: {bl_ret*100:.2f}%   |   "
    f"Volatility: {bl_vol*100:.2f}%   |   Final Value: ${final_v:.2f}   |   "
    f"Total Return: +{total_r:.1f}%   |   RF: {RF*100:.0f}%"
)
fig3.text(0.44, 0.005, metrics_str,
          ha='center', va='bottom', fontsize=8, color='#666', style='italic')

plt.savefig(op('bl_5_cumulative_performance.png'), dpi=180,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("  Saved → bl_5_cumulative_performance.png")

# =============================================================================
# 12. PLOT 6 — Drawdown (Plotly dark)
# =============================================================================
print("[Plot 6] Drawdown (Plotly)...")
cum_r    = (1 + port_dr).cumprod()
run_max  = cum_r.cummax()
drawdown = (cum_r - run_max) / run_max

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown.index, y=drawdown * 100,
    fill='tozeroy', fillcolor='rgba(255,0,0,0.3)',
    line=dict(color='red', width=1.5), name='Drawdown',
    hovertemplate='%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>',
))
fig_dd.update_layout(
    title='<b>Underwater Drawdown Curve</b><br>'
          '<sup>Black-Litterman Max Sharpe Portfolio</sup>',
    xaxis_title='Date', yaxis_title='Drawdown (%)',
    template='plotly_dark', width=1050, height=520,
)
fig_dd.write_html(op('bl_6_drawdown.html'))
print("  Saved → bl_6_drawdown.html")

# =============================================================================
# 13. SUMMARY
# =============================================================================
print("\n" + "="*60)
print("  BLACK-LITTERMAN — FINAL SUMMARY")
print("="*60)
print(f"  Expected Annual Return  : {bl_ret*100:.2f}%")
print(f"  Annual Volatility       : {bl_vol*100:.2f}%")
print(f"  Sharpe Ratio            : {bl_sr:.4f}")
print(f"  Final $1 Value (10yr)   : ${final_v:.2f}")
print(f"  Total Return            : +{total_r:.1f}%")
print("-"*60)
print("  Investor Views Applied:")
for vt, ve in zip(VIEW_TICKERS, VIEW_EXCESS):
    print(f"    {SHORT[vt]:<14}  +{ve*100:.0f}% above equilibrium")
print("-"*60)
print("  Optimal Weights:")
for t, w in zip(TICKERS, W_bl):
    bar  = '█' * int(w * 40)
    flag = '●' if w >= WEIGHT_THRESHOLD else ' '
    print(f"  {flag} {SHORT[t]:<14}  {w*100:>6.2f}%  {bar}")
print("="*60)
print(f"\n✓ All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")