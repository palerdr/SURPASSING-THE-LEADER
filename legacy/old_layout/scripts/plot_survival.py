#!/usr/bin/env python3
"""
Survival Probability Landscape Visualizer

Plots each factor of P(survival) = base_curve * cardiac * referee * physicality
individually, then shows their composition across realistic game scenarios.

This helps you see where the hyperparameter knobs bite hardest before training.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Constants (mirrored from Constants.py) ──
CYLINDER_MAX = 300
BASE_CURVE_K = 3
CARDIAC_DECAY = 0.85
REFEREE_DECAY = 0.88
REFEREE_FLOOR = 0.4
PHYSICALITY_HAL = 1.0
PHYSICALITY_BAKU = 0.85  # tuned up from 0.7 for playability


# ── Individual factors ──

def base_curve(t):
    return np.maximum(0, 1 - (t / CYLINDER_MAX) ** BASE_CURVE_K)

def cardiac_modifier(ttd):
    return CARDIAC_DECAY ** (ttd / 60.0)

def referee_modifier(n):
    return np.maximum(REFEREE_FLOOR, REFEREE_DECAY ** n)

def survival_prob(death_dur, ttd, cprs, physicality):
    if death_dur >= CYLINDER_MAX:
        return 0.0
    return base_curve(death_dur) * cardiac_modifier(ttd) * referee_modifier(cprs) * physicality


# ── Figure ──

fig = plt.figure(figsize=(16, 14))
fig.suptitle("Drop The Handkerchief — Survival Probability Landscape", fontsize=15, fontweight='bold')
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

# ── Panel 1: Base Curve (oxygen deprivation) ──
ax1 = fig.add_subplot(gs[0, 0])
t = np.linspace(0, 300, 300)
ax1.plot(t, base_curve(t), 'k-', linewidth=2)
ax1.axvline(x=60, color='green', linestyle='--', alpha=0.7, label='60s (1-min death)')
ax1.axvline(x=120, color='orange', linestyle='--', alpha=0.7, label='120s (2-min death)')
ax1.axvline(x=200, color='red', linestyle='--', alpha=0.7, label='200s')
ax1.set_xlabel('Death Duration (seconds)')
ax1.set_ylabel('Base Survival Factor')
ax1.set_title(f'Base Curve: max(0, 1 - (t/300)^{BASE_CURVE_K})')
ax1.legend(fontsize=8)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)
# Annotate key values
ax1.annotate(f'{base_curve(60):.3f}', xy=(60, base_curve(60)), xytext=(90, base_curve(60)+0.05),
             arrowprops=dict(arrowstyle='->', color='green'), fontsize=8, color='green')
ax1.annotate(f'{base_curve(120):.3f}', xy=(120, base_curve(120)), xytext=(150, base_curve(120)+0.08),
             arrowprops=dict(arrowstyle='->', color='orange'), fontsize=8, color='orange')

# ── Panel 2: Cardiac Modifier ──
ax2 = fig.add_subplot(gs[0, 1])
ttd = np.linspace(0, 600, 300)  # up to 10 minutes cumulative
ax2.plot(ttd, cardiac_modifier(ttd), 'b-', linewidth=2)
# Mark common TTD milestones
for mark, label in [(60, '1 death (60s)'), (120, '2 deaths'), (180, '3 deaths'), (300, '5 min TTD')]:
    val = cardiac_modifier(mark)
    ax2.plot(mark, val, 'bo', markersize=5)
    ax2.annotate(f'{label}\n{val:.2f}', xy=(mark, val), xytext=(mark+20, val+0.03),
                 fontsize=7, ha='left')
ax2.set_xlabel('Prior Total Time Dead (seconds)')
ax2.set_ylabel('Cardiac Multiplier')
ax2.set_title(f'Cardiac Degradation: {CARDIAC_DECAY}^(ttd/60)')
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, alpha=0.3)

# ── Panel 3: Referee Modifier ──
ax3 = fig.add_subplot(gs[1, 0])
cprs = np.arange(0, 16)
ref_vals = referee_modifier(cprs)
ax3.bar(cprs, ref_vals, color=['steelblue' if v > REFEREE_FLOOR else 'firebrick' for v in ref_vals],
        edgecolor='black', linewidth=0.5)
ax3.axhline(y=REFEREE_FLOOR, color='red', linestyle='--', alpha=0.7,
            label=f'Floor = {REFEREE_FLOOR}')
# Find where floor kicks in
floor_cpr = next(n for n in cprs if REFEREE_DECAY ** n <= REFEREE_FLOOR)
ax3.axvline(x=floor_cpr - 0.5, color='red', linestyle=':', alpha=0.5)
ax3.annotate(f'Floor kicks in at CPR #{floor_cpr}', xy=(floor_cpr, REFEREE_FLOOR),
             xytext=(floor_cpr + 2, REFEREE_FLOOR + 0.15),
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=8, color='red')
ax3.set_xlabel('CPRs Performed (both players combined)')
ax3.set_ylabel('Referee Multiplier')
ax3.set_title(f'Referee Fatigue: max({REFEREE_FLOOR}, {REFEREE_DECAY}^n)')
ax3.legend(fontsize=8)
ax3.set_ylim(0, 1.1)
ax3.grid(True, alpha=0.3, axis='y')

# ── Panel 4: Composed — Hal vs Baku across death durations (fresh state) ──
ax4 = fig.add_subplot(gs[1, 1])
t = np.linspace(1, 299, 298)
for phys, name, color in [(PHYSICALITY_HAL, 'Hal (1.0)', 'darkblue'),
                           (PHYSICALITY_BAKU, f'Baku ({PHYSICALITY_BAKU})', 'crimson'),
                           (0.7, 'Baku (0.7, original)', 'crimson')]:
    ls = '-' if phys != 0.7 else ':'
    p_vals = base_curve(t) * phys  # fresh: cardiac=1, referee=1
    ax4.plot(t, p_vals, color=color, linestyle=ls, linewidth=2 if phys != 0.7 else 1,
             label=f'{name}', alpha=0.9 if phys != 0.7 else 0.5)
ax4.set_xlabel('Death Duration (seconds)')
ax4.set_ylabel('P(survival)')
ax4.set_title('First Death: Hal vs Baku (no prior damage, fresh referee)')
ax4.legend(fontsize=8)
ax4.set_ylim(-0.05, 1.05)
ax4.grid(True, alpha=0.3)

# ── Panel 5: Composed — Progressive game deterioration for Baku ──
ax5 = fig.add_subplot(gs[2, 0])
t = np.linspace(1, 299, 298)
scenarios = [
    (0, 0, 'Fresh (0 TTD, 0 CPR)', 'green'),
    (60, 1, 'After 1 death (60s TTD, 1 CPR)', 'gold'),
    (120, 2, 'After 2 deaths (120s TTD, 2 CPR)', 'orange'),
    (180, 4, 'After 3 deaths (180s TTD, 4 CPR)', 'red'),
    (300, 6, 'After 5 min TTD (300s, 6 CPR)', 'darkred'),
]
for ttd_val, cpr_val, label, color in scenarios:
    p_vals = np.array([survival_prob(d, ttd_val, cpr_val, PHYSICALITY_BAKU) for d in t])
    ax5.plot(t, p_vals, color=color, linewidth=1.5, label=label)
ax5.set_xlabel('Death Duration (seconds)')
ax5.set_ylabel('P(survival)')
ax5.set_title(f'Baku ({PHYSICALITY_BAKU}) — Survival Degrades Over Match')
ax5.legend(fontsize=7, loc='upper right')
ax5.set_ylim(-0.05, 1.05)
ax5.grid(True, alpha=0.3)

# ── Panel 6: The strategic question — at what death duration is Baku <50%? ──
ax6 = fig.add_subplot(gs[2, 1])
# For each game state, find where P drops below 50%
ttd_range = np.arange(0, 361, 60)
cpr_range = np.arange(0, 8)
# Heatmap: death duration that gives 50% survival for Baku
threshold = 0.5
heatmap = np.zeros((len(cpr_range), len(ttd_range)))
for i, cprs_val in enumerate(cpr_range):
    for j, ttd_val in enumerate(ttd_range):
        # Binary search for the death duration where P crosses threshold
        for d in range(1, 300):
            p = survival_prob(d, ttd_val, cprs_val, PHYSICALITY_BAKU)
            if p < threshold:
                heatmap[i, j] = d
                break
        else:
            heatmap[i, j] = 300  # never drops below threshold before 300

im = ax6.imshow(heatmap, aspect='auto', cmap='RdYlGn',
                extent=[-0.5, len(ttd_range)-0.5, len(cpr_range)-0.5, -0.5],
                vmin=60, vmax=280)
ax6.set_xticks(range(len(ttd_range)))
ax6.set_xticklabels([f'{t}s' for t in ttd_range], fontsize=7)
ax6.set_yticks(range(len(cpr_range)))
ax6.set_yticklabels([str(n) for n in cpr_range])
ax6.set_xlabel('Prior TTD (seconds)')
ax6.set_ylabel('CPRs Performed')
ax6.set_title(f'Baku: Death Duration for <50% Survival\n(lower = more dangerous)')
# Add text annotations
for i in range(len(cpr_range)):
    for j in range(len(ttd_range)):
        val = int(heatmap[i, j])
        color = 'white' if val < 160 else 'black'
        ax6.text(j, i, f'{val}s', ha='center', va='center', fontsize=7, color=color)
cb = plt.colorbar(im, ax=ax6, shrink=0.8)
cb.set_label('Death Duration (s)', fontsize=8)

plt.savefig('/home/jcena/STL/STL/scripts/survival_landscape.png', dpi=150, bbox_inches='tight')
print("Saved to scripts/survival_landscape.png")
plt.close()
