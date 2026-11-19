"""Regenerate Figures 5 and 6 from the same code that produces the numbers in
Section 6.  Writes figures/fig5_bound_mirror.pdf and figures/fig6_closedform_solver.pdf."""
import os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # noqa: F401
from common import *
from experiments import place, solve, Ep, vp3, both, disag, HALF, FOVS

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
SIGMAS = [0.0, 0.5, 1.0, 2.0]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.2})
COL = {40: "#1b6ca8", 60: "#c8521b", 90: "#2e7d32"}


def boot_median_ci(a, n=400, rng=None):
    a = np.asarray(a); rng = rng or np.random.default_rng(0)
    if len(a) < 5: return np.nan, np.nan
    m = [np.median(rng.choice(a, len(a))) for _ in range(n)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


# ---------------------------------------------------------------- Figure 5
def figure5():
    rng = np.random.default_rng(2)
    EDGES = np.array([0, 1/6, 1/3, 1/2, 2/3, 5/6, 1.0])
    mid = 0.5*(EDGES[:-1]+EDGES[1:])
    radial = {fov: [[] for _ in range(len(EDGES)-1)] for fov in FOVS}
    radial_rho = {fov: [[] for _ in range(len(EDGES)-1)] for fov in FOVS}
    for fov, f in FOVS.items():
        n = 0
        while n < 3000:
            o = observe_corner(place(f, rng), rand_frame(rng), f, 0.5, rng)
            if o is None: continue
            c, L = o
            comps = admissible(coeffs(c, L))
            if not comps: continue
            n += 1
            rad = np.hypot(c[0], c[1])/HALF
            k = np.searchsorted(EDGES, rad, side="right")-1
            if 0 <= k < len(EDGES)-1:
                radial[fov][k].append(np.sqrt(comps[0][0])/f)
                radial_rho[fov][k].append(np.sqrt(comps[0][0])/np.hypot(c[0], c[1]))

    rng = np.random.default_rng(3); fv = 866.0
    EB = [(0, 5), (5, 15), (15, 45), (45, 180)]
    stat = {b: [0, 0] for b in EB}
    n = 0
    while n < 6000:
        Fr = rand_frame(rng); Dg = np.diag(rng.choice([-1., 1.], 3))
        P1, P2 = place(fv, rng), place(fv, rng)
        o1 = observe_corner(P1, Fr, fv, 0.5, rng)
        o2 = observe_corner(P2, Fr@Dg, fv, 0.5, rng)
        if o1 is None or o2 is None: continue
        S1, S2 = both(*o1, fv), both(*o2, fv)
        if S1 is None or S2 is None: continue
        n += 1
        sep = np.degrees(np.arccos(np.clip(
            (P1/np.linalg.norm(P1))@(P2/np.linalg.norm(P2)), -1, 1)))
        best = min(((disag(S1[i], S2[j]), (i, j)) for i in range(2)
                    for j in range(2)))[1]
        t = (int(np.argmin([disag(S1[k], Fr) for k in range(2)])),
             int(np.argmin([disag(S2[k], Fr@Dg) for k in range(2)])))
        for b in EB:
            if b[0] <= sep < b[1]:
                stat[b][0] += 1; stat[b][1] += (best == t)

    fig, ax2 = plt.subplots(1, 2, figsize=(5.4, 2.4))
    figm, axm = plt.subplots(figsize=(3.3, 2.4))
    ax = [ax2[0], ax2[1], axm]          # panel (c), mirror resolution, is its own figure in the Online Resource
    for fov in FOVS:
        med = [np.median(v) if len(v) > 4 else np.nan for v in radial[fov]]
        lo = [boot_median_ci(v)[0] if len(v) > 4 else np.nan for v in radial[fov]]
        hi = [boot_median_ci(v)[1] if len(v) > 4 else np.nan for v in radial[fov]]
        ax[0].plot(mid, med, "o-", color=COL[fov], ms=3, label=f"{fov}° FOV")
        ax[0].fill_between(mid, lo, hi, color=COL[fov], alpha=0.18, lw=0)
    ax[0].axvspan(0, 1/3, color="0.9", zorder=0)
    ax[0].axvspan(2/3, 1.0, color="0.9", zorder=0)
    ax[0].set_xlabel(r"$\rho$ / image half-width")
    ax[0].set_ylabel(r"median $f_{\min}/f$")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, None)
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].set_title("(a)", fontsize=9, loc="left")
    for fov in FOVS:
        med = [np.median(v) if len(v) > 4 else np.nan for v in radial_rho[fov]]
        lo = [boot_median_ci(v)[0] if len(v) > 4 else np.nan for v in radial_rho[fov]]
        hi = [boot_median_ci(v)[1] if len(v) > 4 else np.nan for v in radial_rho[fov]]
        ax[1].plot(mid, med, "o-", color=COL[fov], ms=3)
        ax[1].fill_between(mid, lo, hi, color=COL[fov], alpha=0.18, lw=0)
    ax[1].set_xlabel(r"$\rho$ / image half-width")
    ax[1].set_ylabel(r"median $f_{\min}/\rho$")
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, None)
    ax[1].set_title("(b)", fontsize=9, loc="left")

    xs = np.arange(len(EB))
    frac = [100*stat[b][1]/stat[b][0] for b in EB]
    err = [100*1.96*np.sqrt(max(stat[b][1]/stat[b][0]*(1-stat[b][1]/stat[b][0]), 1e-9)
                            / stat[b][0]) for b in EB]
    ax[2].errorbar(xs, frac, yerr=err, fmt="o", color="#c8521b", ms=5, capsize=3)
    for i, b in enumerate(EB):
        ax[2].text(i, 101.5, f"n={stat[b][0]}", ha="center", fontsize=6)
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels([f"{b[0]}–{b[1]}" for b in EB])
    ax[2].set_xlabel("ray separation (degrees)")
    ax[2].set_ylabel("mirror sign resolved correctly (%)")
    ax[2].set_ylim(85, 104); ax[2].axhline(100, color="0.6", lw=0.6, ls=":")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_bound_mirror.pdf")
    print(f"wrote {OUT}/fig5_bound_mirror.pdf")
    figm.tight_layout(); figm.savefig(f"{OUT}/fig5c_mirror.pdf")
    print(f"wrote {OUT}/fig5c_mirror.pdf")


# ---------------------------------------------------------------- Figure 6
def figure6():
    rng = np.random.default_rng(1)
    e1 = {fov: {s: [] for s in SIGMAS} for fov in FOVS}
    for fov, f in FOVS.items():
        for s in SIGMAS:
            n = 0
            while n < 1000:
                D = rand_frame(rng)
                o = observe_corner(place(f, rng), D, f, s, rng)
                if o is None: continue
                n += 1
                Dh = closed_form(*o, f)
                if Dh is None: continue
                e1[fov][s].append(angle_err_deg(Dh, D, o[0], f))

    rng = np.random.default_rng(4)
    sol = {s: [] for s in SIGMAS}; vp = {s: [] for s in SIGMAS}
    fails = {s: 0 for s in SIGMAS}
    for s in SIGMAS:
        n = 0
        while n < 3000:
            f = rng.uniform(400, 1500); Fr = rand_frame(rng)
            d1, d2, d3 = Fr[:, 0], Fr[:, 1], Fr[:, 2]
            psi = rng.uniform(0, 2*np.pi)
            d1p = np.cos(psi)*d1+np.sin(psi)*d2
            d2p = -np.sin(psi)*d1+np.cos(psi)*d2
            o1 = observe_corner(place(f, rng), np.column_stack([d1, d2, d3]), f, s, rng)
            o2 = observe_corner(place(f, rng), np.column_stack([d1p, d2p, d3]), f, s, rng)
            if o1 is None or o2 is None: continue
            (ca, L1), (cb, L2) = o1, o2
            l1, l2, l3 = L1; l1p, l2p, l3p = L2
            rts = solve(l1, l2, l3, l3p)
            n += 1
            if not rts:
                fails[s] += 1
            else:
                best = min(rts, key=lambda x: Ep(x, l1p, l2p, l3, l3p))
                sol[s].append(abs(np.sqrt(best)-f)/f)
            oa = observe_corner(place(f, rng), Fr, f, s, rng)
            ob = observe_corner(place(f, rng), Fr, f, s, rng)
            if oa and ob:
                fb = vp3([(oa[1][i], ob[1][i]) for i in range(3)])
                if fb: vp[s].append(abs(fb-f)/f)

    SP = SIGMAS[1:]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for fov in FOVS:
        med = [np.median(e1[fov][s]) for s in SP]
        lo = [boot_median_ci(e1[fov][s])[0] for s in SP]
        hi = [boot_median_ci(e1[fov][s])[1] for s in SP]
        ax[0].plot(SP, med, "o-", color=COL[fov], ms=3, label=f"{fov}$^\\circ$ median")
        ax[0].fill_between(SP, lo, hi, color=COL[fov], alpha=0.2, lw=0)
    p90 = [np.percentile(np.concatenate([e1[fov][s] for fov in FOVS]), 90) for s in SP]
    ax[0].plot(SP, p90, "s--", color="0.35", ms=3, label="90th pct (all FOV)")
    ax[0].set_xlabel(r"endpoint noise $\sigma$ (px)")
    ax[0].set_ylabel("edge-direction error (degrees)")
    ax[0].set_xticks(SP); ax[0].set_ylim(0, None)
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    ax[0].set_title("(a)", fontsize=9, loc="left")

    for name, dat, col, mk in (("corner-plus-line quadratic", sol, "#1b6ca8", "o"),
                               ("three vanishing points", vp, "#c8521b", "^")):
        med = [100*np.median(dat[s]) for s in SP]
        lo = [100*boot_median_ci(dat[s])[0] for s in SP]
        hi = [100*boot_median_ci(dat[s])[1] for s in SP]
        ax[1].plot(SP, med, mk+"-", color=col, ms=3.5, label=name+", median")
        ax[1].fill_between(SP, lo, hi, color=col, alpha=0.2, lw=0)
        ax[1].plot(SP, [100*np.percentile(dat[s], 90) for s in SP], mk+":",
                   color=col, ms=3, alpha=0.75, label=name+", 90th pct")
    ax[1].set_xlabel(r"endpoint noise $\sigma$ (px)")
    ax[1].set_ylabel("relative error in $f$ (%)")
    ax[1].set_xticks(SP)
    ax[1].legend(frameon=False, fontsize=6, loc="upper left")
    ax[1].set_title("(b)", fontsize=9, loc="left")
    ax[1].set_ylim(-9, None)
    ax[1].axhline(0, color="0.8", lw=0.6)
    for s in SP:
        ax[1].annotate(f"no root\n{100*fails[s]/3000:.0f}%" if s == SP[0] else f"{100*fails[s]/3000:.0f}%", (s, -6.5), fontsize=6,
                       color="0.4", ha="center")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_closedform_solver.pdf")
    print(f"wrote {OUT}/fig6_closedform_solver.pdf")


if __name__ == "__main__":
    figure5(); figure6()
