"""
symbolic.py -- symbolic and numerical verification of the algebraic claims in
"Below the Parallelepiped: What One or Two Right-Angle Corners Tell a
Camera of Unknown Focal Length".

Run:  python3 symbolic.py             (requires sympy, numpy)
      python3 symbolic.py --quick     (skips the randomized searches)

Each check prints PASS/FAIL and the part of the paper it backs.
"""

import sys

import numpy as np
import sympy as sp

QUICK = "--quick" in sys.argv

f, x = sp.symbols('f x', positive=True)
u, v = sp.symbols('u v', real=True)

results = []


# Every check is keyed to a LaTeX label, optionally followed by "|note".  If the
# compiled paper's .aux file is found, the label is shown with its current
# number ("Theorem 2", "Eq. (7)"), so these locations cannot go stale when the
# paper is reorganised.  Pass the .aux path as --aux, or leave main.aux beside
# this script.
import os as _os, re as _re, sys as _sys
def _load_aux():
    """labels of the article, and of its Online Resource if supplement.aux sits beside it"""
    cands = []
    if "--aux" in _sys.argv: cands.append(_sys.argv[_sys.argv.index("--aux") + 1])
    cands += ["main.aux", _os.path.join("..", "main.aux")]
    out = {}
    for p in cands:
        if _os.path.exists(p):
            for q in (p, _os.path.join(_os.path.dirname(p) or ".", "supplement.aux")):
                if not _os.path.exists(q): continue
                txt = open(q, errors="ignore").read()
                for lab, num, kind in _re.findall(
                        r"\\newlabel\{([^}@]*)\}\{\{([^}]*)\}\{[^}]*\}\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\{([a-zA-Z]*)[^}]*\}", txt):
                    out.setdefault(lab, (num, kind))
                for lab, num in _re.findall(r"\\newlabel\{([^}@]*)\}\{\{([^}]*)\}\{[^}]*\}\}", txt):
                    out.setdefault(lab, (num, ""))             # plain two-field labels (no hyperref)
                # cleveref records the true kind of each label
                for lab, kind, num in _re.findall(
                        r"\\newlabel\{([^}@]*)@cref\}\{\{\[([a-zA-Z]*)\]\[[^\]]*\]\[[^\]]*\]([^}]*)\}", txt):
                    out[lab] = (num, kind)
            return out
    return out
_AUX = _load_aux()
_KIND = {"equation": "Eq. ({})", "theorem": "Theorem {}", "proposition": "Proposition {}",
         "lemma": "Lemma {}", "corollary": "Corollary {}", "remark": "Remark {}",
         "section": "Sec. {}", "subsection": "Sec. {}", "subsubsection": "Sec. {}"}
def _where(key):
    lab, _, note = key.partition("|")
    if lab in _AUX:
        num, kind = _AUX[lab]
        return _KIND.get(kind, "{}").format(num) + note + f" [{lab}]"
    return lab + note
def check(name, where, ok):
    results.append((name, where, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {_where(where)}: {name}")


K = sp.diag(f, f, 1)
c = sp.Matrix([u, v, 1])
r = K.inv() * c                     # (u/f, v/f, 1)

a1, b1, e1, a2, b2, e2, a3, b3, e3 = sp.symbols(
    'a1 b1 e1 a2 b2 e2 a3 b3 e3', real=True)


def normal(a, b, e):
    return sp.Matrix([f * a, f * b, e])


def tvec(a, b, e):
    return normal(a, b, e).cross(r)


# ----------------------------------------------------------------- Section 2.1
# t_i points along the observed half-edge for every f: with l=(a,b,e),
# c=(u,v,1), K=diag(f,f,1), the image direction of t_i = n_i x r is
# (f^2+u^2+v^2)(b,-a)/P_z.
#
# The image direction of a 3D direction t at the image point c is the
# derivative of the projection along t, NOT the location of t's vanishing
# point: d and -d share a vanishing point, so the vanishing point cannot
# carry the orientation.  The line passes through c, so e = -(a u + b v).

a_, b_ = sp.symbols('a b', real=True)
Pz = sp.symbols('Pz', positive=True)
lam = sp.symbols('lambda')
t_sec21 = tvec(a_, b_, -(a_ * u + b_ * v))
P_world = sp.Matrix([u * Pz / f, v * Pz / f, Pz])   # projects to c
X = P_world + lam * t_sec21
img = sp.Matrix([f * X[0] / X[2], f * X[1] / X[2]])
t_img = sp.simplify(sp.diff(img, lam).subs(lam, 0))
check("t_i image direction is a positive multiple of (f^2+u^2+v^2)(b,-a)/P_z, "
      "so its orientation does not depend on f", "sec:setup",
      sp.simplify(t_img - (f**2 + u**2 + v**2) / Pz
                  * sp.Matrix([b_, -a_])) == sp.zeros(2, 1))


# ----------------------------------------------------------------- Section 3.2
# Because every interpretation plane contains the ray (n_i . r = 0),
#     g_ij = |r|^2 (n_i . n_j)   and   x g_ij = (x + rho^2)(A_ij x + e_i e_j),
# so the sign condition is a product of three LINEAR functions of x = f^2.

a1, b1, e1, a2, b2, e2, a3, b3, e3 = sp.symbols(
    'a1 b1 e1 a2 b2 e2 a3 b3 e3', real=True)
rho2 = u**2 + v**2


def normal(a, b, e):
    return sp.Matrix([f * a, f * b, e])


def tvec(a, b, e):
    return normal(a, b, e).cross(r)


concurrent = {e1: -(a1 * u + b1 * v), e2: -(a2 * u + b2 * v),
              e3: -(a3 * u + b3 * v)}

n1, n2 = normal(a1, b1, e1), normal(a2, b2, e2)
check("n_i . r = 0: every interpretation plane contains the ray",
      "sec:admissible", sp.simplify((n1.dot(r)).subs(concurrent)) == 0)

g12 = (tvec(a1, b1, e1)).dot(tvec(a2, b2, e2))
check("g_ij = |r|^2 (n_i . n_j), by the Binet-Cauchy identity", "eq:gfactor",
      sp.simplify((g12 - r.dot(r) * n1.dot(n2)).subs(concurrent)) == 0)

A12 = a1 * a2 + b1 * b2
check("x g_ij = (x + rho^2)(A_ij x + e_i e_j)", "eq:lin",
      sp.simplify(sp.expand((x * g12).subs(f, sp.sqrt(x)).subs(concurrent))
                  - sp.expand(((x + rho2) * (A12 * x + e1 * e2)).subs(concurrent)))
      == 0)

# the quadratic of the earlier derivation carries x + rho^2 as a factor
B12 = ((a1 * v - b1 * u) * (a2 * v - b2 * u)
       - u * (a1 * e2 + a2 * e1) - v * (b1 * e2 + b2 * e1))
C12 = e1 * e2 * rho2
check("the quadratic A x^2 + B x + C equals (x + rho^2)(A x + e_i e_j), "
      "i.e. B = A rho^2 + e_i e_j and C = e_i e_j rho^2", "sec:admissible",
      sp.simplify(sp.expand((A12 * x**2 + B12 * x + C12).subs(concurrent))
                  - sp.expand(((x + rho2) * (A12 * x + e1 * e2)).subs(concurrent)))
      == 0)


check("g_ij / (n_i . n_j) = (f^2 + rho^2)/f^2 > 0, so the sign condition is "
      "exactly (n1.n2)(n2.n3)(n3.n1) < 0", "eq:planes",
      sp.simplify(((g12 / n1.dot(n2)).subs(concurrent))
                  - (f**2 + rho2) / f**2) == 0)
check("n_i . n_j = A_ij f^2 + e_i e_j is linear in f^2, which is why the "
      "condition is a product of three linear functions", "sec:admissible",
      sp.simplify(sp.expand(n1.dot(n2)) - (A12 * f**2 + e1 * e2)) == 0)


def lin(ai, bi, ei, aj, bj, ej):
    return (ai * aj + bi * bj) * x + ei * ej


L = sp.expand(lin(a1, b1, e1, a2, b2, e2)
              * lin(a2, b2, e2, a3, b3, e3)
              * lin(a3, b3, e3, a1, b1, e1))

# ------------------------------------------------- Proposition 1(1) and 1(2)
check("L(0) = (e1 e2 e3)^2 >= 0", "prop:bound|(1)",
      sp.simplify(L.subs(x, 0) - (e1 * e2 * e3)**2) == 0)
check("leading coefficient of L is A12 A23 A31", "prop:bound|(2)",
      sp.simplify(sp.Poly(L, x).coeff_monomial(x**3)
                  - (a1 * a2 + b1 * b2) * (a2 * a3 + b2 * b3)
                  * (a3 * a1 + b3 * b1)) == 0)
check("each factor has at most one root, so L has degree 3 in x",
      "prop:bound|(3)", sp.Poly(L, x).degree() == 3)

# ------------------ the radial parameterization and the closed-form theorem
# psi_i is the signed angle from the outward radial direction to the half-edge
# (b_i, -a_i).  Then e_i = rho sin psi_i and A_ij = cos(psi_i - psi_j), so
#     A_ij x + e_i e_j = rho^2 sin psi_i sin psi_j (lambda_ij mu + 1),
# with mu = x/rho^2 and lambda_ij = 1 + cot psi_i cot psi_j.  The prefactors
# multiply to a square, so the sign condition is prod(lambda_ij mu + 1) < 0.
s1, s2, s3 = sp.symbols('psi1 psi2 psi3', real=True)
rho = sp.sqrt(rho2)
# a half-edge at angle psi from the radial direction c/rho
cu, cv = u / rho, v / rho


def line_at(psi):
    """(a, b, e) of the edge whose half-edge leaves c at angle psi."""
    dx = cu * sp.cos(psi) - cv * sp.sin(psi)
    dy = cv * sp.cos(psi) + cu * sp.sin(psi)
    a, b = -dy, dx            # so that the half-edge (b, -a) equals (dx, dy)
    return a, b, -(a * u + b * v)


a_1, b_1, e_1 = line_at(s1)
a_2, b_2, e_2 = line_at(s2)
check("e_i = rho sin(psi_i)", "eq:psi",
      sp.simplify(sp.expand_trig(e_1 - rho * sp.sin(s1))) == 0)
check("A_ij = cos(psi_i - psi_j)", "eq:psi",
      sp.simplify(sp.expand_trig(a_1 * a_2 + b_1 * b_2 - sp.cos(s1 - s2))) == 0)

mu = sp.symbols('mu', positive=True)
lam12 = 1 + sp.cot(s1) * sp.cot(s2)
check("A_ij x + e_i e_j = rho^2 sin.sin (lambda_ij mu + 1)",
      "eq:lambda",
      sp.simplify(sp.expand_trig(
          ((a_1*a_2 + b_1*b_2) * (mu*rho2) + e_1*e_2)
          - rho2*sp.sin(s1)*sp.sin(s2)*(lam12*mu + 1))) == 0)
check("lambda_ij = cos(psi_i - psi_j)/(sin psi_i sin psi_j)", "sec:admissible",
      sp.simplify(sp.expand_trig(lam12 - sp.cos(s1-s2)/(sp.sin(s1)*sp.sin(s2))))
      == 0)
check("the three prefactors multiply to (sin psi_1 sin psi_2 sin psi_3)^2 >= 0, "
      "so they drop out of the sign condition", "eq:mu",
      sp.simplify(sp.expand_trig(
          (sp.sin(s1)*sp.sin(s2))*(sp.sin(s2)*sp.sin(s3))*(sp.sin(s3)*sp.sin(s1))
          - (sp.sin(s1)*sp.sin(s2)*sp.sin(s3))**2)) == 0)

# the threshold in closed form
check("f_ij^2 = -rho^2 sin psi_i sin psi_j / cos(psi_i - psi_j) = rho^2 / -lambda_ij",
      "eq:thresh",
      sp.simplify(sp.expand_trig(
          -rho2*sp.sin(s1)*sp.sin(s2)/sp.cos(s1-s2) - rho2/(-lam12))) == 0)

# at most two of the lambda are negative
t1, t2, t3 = sp.cot(s1), sp.cot(s2), sp.cot(s3)
check("(t1 t2)(t2 t3)(t3 t1) = (t1 t2 t3)^2 >= 0 with t_i = cot psi_i, so the "
      "three products cannot all be below -1: |S| <= 2", "thm:interval",
      sp.simplify((t1*t2)*(t2*t3)*(t3*t1) - (t1*t2*t3)**2) == 0)

# ------------------------------------- Prop. 3, the general calibration matrix
# Nothing in the derivation used the form of K.  n_i . r = l_i . c = 0 holds
# for every K, so the condition is always (l1 w* l2)(l2 w* l3)(l3 w* l1) < 0
# with w* = K K^T the dual image of the absolute conic, each factor linear in
# the six entries of w*.
fx, fy, sk, pxc, pyc = sp.symbols('f_x f_y s p_x p_y', real=True)
Kg = sp.Matrix([[fx, sk, pxc], [0, fy, pyc], [0, 0, 1]])
L1 = sp.Matrix([a1, b1, e1])
L2 = sp.Matrix([a2, b2, e2])
cg = sp.Matrix([u, v, 1])
rg = Kg.inv() * cg
N1, N2 = Kg.T * L1, Kg.T * L2
check("n_i . r = l_i . c = 0 for ANY calibration matrix", "sec:identity",
      sp.simplify((N1.dot(rg)).subs(concurrent)) == 0)
check("g_ij = |r|^2 (n_i . n_j) for ANY calibration matrix", "sec:identity",
      sp.simplify((((N1.cross(rg)).dot(N2.cross(rg)))
                   - rg.dot(rg) * N1.dot(N2)).subs(concurrent)) == 0)
check("n_i . n_j = l_i^T (K K^T) l_j, the DIAC form",
      "eq:diac",
      sp.simplify(sp.expand(N1.dot(N2) - (L1.T * (Kg * Kg.T) * L2)[0, 0])) == 0)
_W = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f'w{min(i,j)}{max(i,j)}'))
check("l_i^T w* l_j is linear in the six entries of w*", "prop:general",
      sp.Poly(sp.expand((L1.T * _W * L2)[0, 0]),
              *sorted(_W.free_symbols, key=str)).total_degree() == 1)
check("the three coefficient products are perfect squares, so the anisotropic "
      "admissible region avoids the origin and both axis directions at infinity",
      "sec:slices",
      all(sp.simplify(sp.expand(p1 * p2 * p3) - sp.expand(q**2)) == 0
          for p1, p2, p3, q in
          [(a1*a2, a2*a3, a3*a1, a1*a2*a3),
           (b1*b2, b2*b3, b3*b1, b1*b2*b3),
           (e1*e2, e2*e3, e3*e1, e1*e2*e3)]))

# ------------------------------------- Prop. 4, the first-order noise model
# d x_ij / d psi_i = -rho^2 sin(2 psi_j) / (2 cos^2(psi_i - psi_j))
_x = -rho2 * sp.sin(s1) * sp.sin(s2) / sp.cos(s1 - s2)
check("d x_ij / d psi_i = -rho^2 sin(2 psi_j) / (2 cos^2(psi_i-psi_j))",
      "eq:sd",
      sp.simplify(sp.expand_trig(sp.diff(_x, s1)
                  + rho2 * sp.sin(2*s2) / (2*sp.cos(s1-s2)**2))) == 0)
check("and symmetrically in j, which gives the stated sd by the delta method",
      "eq:sd",
      sp.simplify(sp.expand_trig(sp.diff(_x, s2)
                  + rho2 * sp.sin(2*s1) / (2*sp.cos(s1-s2)**2))) == 0)

# ---------------------------- the parity identity behind the likelihood (7.2)
# For independent sign flips with probabilities p_j,
#     P(an even number flip) = (1 + prod_j (1 - 2 p_j)) / 2.
_p = sp.symbols('p1 p2 p3', positive=True)
_even = 0
for _bits in [(0,0,0),(0,1,1),(1,0,1),(1,1,0)]:          # even-weight patterns
    _t = 1
    for _k in range(3):
        _t *= _p[_k] if _bits[_k] else (1 - _p[_k])
    _even += _t
check("P(even number of sign flips) = (1 + prod(1 - 2 p_j))/2",
      "eq:pi",
      sp.simplify(sp.expand(_even
                  - (1 + sp.prod([1 - 2*_p[k] for k in range(3)]))/2)) == 0)

# ----------------- the solver for ANY K, and its identity with the classical form
_fx, _fy, _sk, _px, _py = sp.symbols('F_x F_y S_k P_x P_y', real=True)
_K = sp.Matrix([[_fx, _sk, _px], [0, _fy, _py], [0, 0, 1]])
_Ws, _W = _K * _K.T, (_K * _K.T).inv()
def _L(n): return sp.Matrix(sp.symbols(f'{n}a {n}b {n}e', real=True))
_l1, _l2, _l3, _l3p = _L('q1'), _L('q2'), _L('q3'), _L('q3p')
_n = [_K.T * l for l in (_l1, _l2, _l3, _l3p)]
_w = _n[2].cross(_n[3])
_E = sp.expand((_n[0].dot(_n[1])) * (_w.dot(_w)) - (_n[0].dot(_w)) * (_n[1].dot(_w)))
_v3 = _l3.cross(_l3p)
_cf = sp.expand((_l1.T * _Ws * _l2)[0, 0] * (_v3.T * _W * _v3)[0, 0]
                - (_l1.dot(_v3)) * (_l2.dot(_v3)))
check("solver for ANY K: E = det(K)^2 [(l1 w* l2)(v3 w v3) - (l1.v3)(l2.v3)]",
      "eq:cfsolver",
      sp.simplify(sp.cancel(_E / _cf) - sp.det(_K)**2) == 0)
_h = _W * _v3
_cl = sp.expand(((_l1.cross(_h)).T * _W * (_l2.cross(_h)))[0, 0])
check("...and it IS the classical v1^T w v2 = 0, with v_i = l_i x horizon",
      "eq:cfsolver",
      sp.simplify(sp.cancel(_cl / _cf) - 1 / sp.det(_K)**2) == 0)

# ------------------------- why right angles: the condition is a perfect square
_ki, _kj, _R2, _g, _Ti, _Tj, _c = sp.symbols('k_i k_j R_2 g_ij T_i T_j c', real=True)
_cond = sp.expand((_ki*_kj*_R2 + _g)**2 - _c**2*(_ki**2*_R2 + _Ti)*(_kj**2*_R2 + _Tj))
check("at a right angle the pair condition is the perfect square (k_i k_j R2 + g)^2",
      "sec:why",
      sp.simplify(_cond.subs(_c, 0) - (_ki*_kj*_R2 + _g)**2) == 0)
check("at any other angle the normalisation T_i survives (d/dT_i != 0)",
      "sec:why",
      sp.simplify(sp.diff(_cond, _Ti) + _c**2*(_R2*_kj**2 + _Tj)) == 0)

# ------------------- Prop. (a corner and a parallel line): the identity its proof uses
_Ss = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f'S{min(i,j)}{max(i,j)}'))          # symmetric
_av, _bv, _cv, _dv = [sp.Matrix(sp.symbols(f'{n}1 {n}2 {n}3', real=True)) for n in ('aq', 'bq', 'cq', 'dq')]
check("(a x b)^T adj(S) (c x d) = (a^T S c)(b^T S d) - (a^T S d)(b^T S c) for symmetric S",
      "prop:quadsolver",
      sp.expand((_av.cross(_bv).T * _Ss.adjugate() * _cv.cross(_dv))[0, 0]
                - (_av.T * _Ss * _cv)[0, 0] * (_bv.T * _Ss * _dv)[0, 0]
                + (_av.T * _Ss * _dv)[0, 0] * (_bv.T * _Ss * _cv)[0, 0]) == 0)

# ------------------- Thm. (several frames) and Thm. (focal length and principal point)
_sF = sp.Symbol('s_F', positive=True); _p1, _p2 = sp.symbols('p_1 p_2', real=True)
_MF = sp.Matrix([[_sF, 0, _p1], [0, _sF, _p2], [0, 0, 1]])
_XiF = _MF * _MF.T
_pv = sp.Matrix([_p1, _p2])
check("Xi = M M^T = [[s^2 I + p p^T, p], [p^T, 1]] for K' = K M, M = [[s I, p], [0, 1]]",
      "thm:fpp", sp.simplify(_XiF - sp.Matrix(sp.BlockMatrix([[_sF**2 * sp.eye(2) + _pv * _pv.T, _pv],
                                                               [_pv.T, sp.Matrix([[1]])]]))) == sp.zeros(3, 3))
_mu, _dl = sp.symbols('mu_F delta_F', nonzero=True); _db = sp.Matrix(sp.symbols('db_1 db_2', real=True))
_lam = 1 + _mu * _db.dot(_db) / _dl
_sq = _lam * (1 - _mu * _dl)
_XiT = sp.Matrix(sp.BlockMatrix([[_sq * sp.eye(2) + (_mu * _db) * (_mu * _db).T, _mu * _db],
                                 [(_mu * _db).T, sp.Matrix([[1]])]]))
_dv = sp.Matrix([_db[0], _db[1], _dl])
check("a tilted axis d = (dbar, delta) is an eigenvector, eigenvalue lambda = 1 + mu |dbar|^2/delta, "
      "when p = mu dbar and s^2 = lambda (1 - mu delta)", "eq:fppaxis",
      sp.simplify(_XiT * _dv - _lam * _dv) == sp.zeros(3, 1))
_th, _nu = sp.symbols('theta_F nu_F', real=True)
_c, _d = sp.cos(_th), sp.sin(_th)
_s2 = (1 + (_nu / _c) * _c**2 / _d) * (1 - _d * _nu / _c)
check("with d = (c u, delta), c = cos theta, delta = sin theta, mu = nu/c: s^2 = 1 + k nu - nu^2, "
      "tau = c/delta - delta/c = 2 cot 2 theta", "thm:fpp|, proof",
      sp.simplify(_s2 - (1 + (_c / _d - _d / _c) * _nu - _nu**2)) == 0
      and sp.simplify((_c / _d - _d / _c) - 2 * sp.cos(2 * _th) / sp.sin(2 * _th)) == 0)
_fx, _sk, _u0, _fy, _v0, _tt2 = sp.symbols('f_x s_k u_0 f_y v_0 t_A', positive=True)
_KA = sp.Matrix([[_fx, _sk, _u0], [0, _fy, _v0], [0, 0, 1]]); _KB = _KA.copy(); _KB[0, 0] = _fx * sp.sqrt(1 + _tt2)
_e1 = sp.Matrix([1, 0, 0])
check("K (I + t e1 e1^T) K^T = K' K'^T with K' = K except f_x -> f_x sqrt(1+t): "
      "a camera that only pitches leaves the aspect ratio free",
      "thm:frames", sp.simplify(_KA * (sp.eye(3) + _tt2 * _e1 * _e1.T) * _KA.T - _KB * _KB.T) == sp.zeros(3, 3))

# ------------------- eq. (18): derivatives of the form with respect to the two angles
_pi, _pj, _ff, _rr = sp.symbols('psi_i psi_j f_F rho_F', real=True)
_form = sp.cos(_pi - _pj) * _ff**2 + _rr**2 * sp.sin(_pi) * sp.sin(_pj)
check("form A_ij f^2 + e_i e_j = cos(psi_i - psi_j) f^2 + rho^2 sin psi_i sin psi_j has derivatives "
      "-sin(psi_i-psi_j) f^2 + rho^2 cos psi_i sin psi_j and sin(psi_i-psi_j) f^2 + rho^2 sin psi_i cos psi_j",
      "eq:miss", sp.simplify(sp.diff(_form, _pi) - (-sp.sin(_pi - _pj) * _ff**2 + _rr**2 * sp.cos(_pi) * sp.sin(_pj))) == 0
      and sp.simplify(sp.diff(_form, _pj) - (sp.sin(_pi - _pj) * _ff**2 + _rr**2 * sp.sin(_pi) * sp.cos(_pj))) == 0)

# ------------------- Thm. (where a junction bounds the focal length): the factored form of gamma_a
_e1, _e2, _e3 = sp.symbols('e_b1 e_b2 e_b3', real=True)
_Rz3 = lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]])
_Rx3 = lambda a: sp.Matrix([[1, 0, 0], [0, sp.cos(a), -sp.sin(a)], [0, sp.sin(a), sp.cos(a)]])
_RB = _Rz3(_e1) * _Rx3(_e2) * _Rz3(_e3)
_rx_, _ry_ = sp.symbols('r_x r_y', real=True); _rB = sp.Matrix([_rx_, _ry_, 1]); _zB = sp.Matrix([0, 0, 1])
_okB = True
for (_a, _p, _q) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _da, _dp, _dq = _RB[:, _a], _RB[:, _p], _RB[:, _q]
    _lhs = (_zB.dot(_da) * _rB.dot(_dp) - _rB.dot(_da) * _zB.dot(_dp)) * (_zB.dot(_da) * _rB.dot(_dq) - _rB.dot(_da) * _zB.dot(_dq))
    _rhs = -(_rB.dot(_zB.cross(_dp))) * (_rB.dot(_zB.cross(_dq)))
    _okB &= sp.simplify(sp.expand(sp.expand_trig(_lhs - _rhs))) == 0
check("(z_a zeta_p - zeta_a z_p)(z_a zeta_q - zeta_a z_q) = -(r.(zhat x d_p))(r.(zhat x d_q)) for a rotation, "
      "so gamma_a = -(r.(zhat x d_p))(r.(zhat x d_q))/((r.d_p)(r.d_q))", "eq:boundmap", _okB)

# ------------------- Thm. (where a junction bounds the focal length): the boundary is a circle
_xC, _yC, _fC = sp.symbols('x_C y_C f_C', real=True)
_dpC = sp.Matrix(sp.symbols('p_1 p_2 p_3', real=True)); _dqC = sp.Matrix(sp.symbols('q_1 q_2 q_3', real=True))
_rC = sp.Matrix([_xC / _fC, _yC / _fC, 1]); _zC = sp.Matrix([0, 0, 1])
_QC = sp.expand(_rC.dot(_zC.cross(_dpC)) * _rC.dot(_zC.cross(_dqC)) + _rC.dot(_dpC) * _rC.dot(_dqC))
_polyC = sp.Poly(_QC, _xC, _yC)
check("the boundary (r.(zhat x d_p))(r.(zhat x d_q)) + (r.d_p)(r.d_q) = 0 has no xy term and equal x^2, y^2 "
      "coefficients (dbar_p.dbar_q)/f^2, so it is a circle", "eq:boundmap",
      sp.simplify(_polyC.coeff_monomial(_xC * _yC)) == 0
      and sp.simplify(_polyC.coeff_monomial(_xC**2) - (_dpC[0] * _dqC[0] + _dpC[1] * _dqC[1]) / _fC**2) == 0
      and sp.simplify(_polyC.coeff_monomial(_yC**2) - (_dpC[0] * _dqC[0] + _dpC[1] * _dqC[1]) / _fC**2) == 0)

# ------------------- Thm. (identifiability) and Prop. (rate), part 4: on l_k, (R_i/l_i)(R_j/l_j) = -z_k^2
_g1, _g2, _g3, _zi, _zj = sp.symbols('g_1 g_2 g_3 zeta_i zeta_j', real=True)
_RL = (lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]]))(_g1) * \
      (lambda a: sp.Matrix([[1, 0, 0], [0, sp.cos(a), -sp.sin(a)], [0, sp.sin(a), sp.cos(a)]]))(_g2) * \
      (lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]]))(_g3)
_zL = sp.Matrix([0, 0, 1]); _okL = True
for (_i, _j, _k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _di, _dj, _dk = _RL[:, _i], _RL[:, _j], _RL[:, _k]
    _rL = _zi * _di + _zj * _dj                                  # a point of the edge-on line l_k
    _Ri, _Rj = _rL.dot(_zL.cross(_di)), _rL.dot(_zL.cross(_dj))
    _okL &= sp.simplify(sp.expand_trig(_Ri * _Rj + _zL.dot(_dk)**2 * _rL.dot(_di) * _rL.dot(_dj))) == 0
check("on the edge-on line l_k, (R_i/l_i)(R_j/l_j) = -z_k^2: the two diverging components of gamma have opposite signs",
      "thm:identframe", _okL)

# ------------------- Thm. (where a junction bounds the focal length): gamma_a = 1 + z_p z_q (c-v_p).(c-v_q)/(f^2 l_p l_q)
_h1, _h2, _h3, _xT, _yT, _fT = sp.symbols('h_1 h_2 h_3 x_T y_T f_T', real=True)
_RT = (lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]]))(_h1) * \
      (lambda a: sp.Matrix([[1, 0, 0], [0, sp.cos(a), -sp.sin(a)], [0, sp.sin(a), sp.cos(a)]]))(_h2) * \
      (lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]]))(_h3)
_zT = sp.Matrix([0, 0, 1]); _rT = sp.Matrix([_xT / _fT, _yT / _fT, 1]); _cT = sp.Matrix([_xT, _yT])
_dp, _dq = _RT[:, 1], _RT[:, 2]
_vp = sp.Matrix([_fT * _dp[0] / _dp[2], _fT * _dp[1] / _dp[2]]); _vq = sp.Matrix([_fT * _dq[0] / _dq[2], _fT * _dq[1] / _dq[2]])
_lhsT = _rT.dot(_zT.cross(_dp)) * _rT.dot(_zT.cross(_dq)) + _rT.dot(_dp) * _rT.dot(_dq)
_rhsT = -(_dp[2] * _dq[2] / _fT**2) * (_cT - _vp).dot(_cT - _vq)
check("(r.(zhat x d_p))(r.(zhat x d_q)) + (r.d_p)(r.d_q) = -(z_p z_q/f^2)(c-v_p).(c-v_q): the boundary gamma_a = 1 is "
      "where the edges toward v_p and v_q are perpendicular in the image, the circle with diameter v_p v_q",
      "eq:thales", sp.simplify(sp.expand_trig(sp.together(_lhsT - _rhsT))) == 0)

# ------------------- Prop. (corners that are not right-angled): t_i is a multiple of the part of d_i perpendicular to r
_d = sp.Matrix(sp.symbols('dd_1 dd_2 dd_3', real=True)); _e = sp.Matrix(sp.symbols('ee_1 ee_2 ee_3', real=True))
_rr = sp.Matrix(sp.symbols('rr_1 rr_2 rr_3', real=True)); _n2 = _rr.dot(_rr)
_ti = _d.cross(_rr).cross(_rr); _tj = _e.cross(_rr).cross(_rr)
check("((d_i x r) x r).((d_j x r) x r) = |r|^2 (|r|^2 d_i.d_j - (d_i.r)(d_j.r)): g_ij is a multiple of c_ij - a_i a_j",
      "eq:nonright", sp.expand(_ti.dot(_tj) - _n2 * (_n2 * _d.dot(_e) - _d.dot(_rr) * _e.dot(_rr))) == 0)

# ------------------- Cor. (distortion with fitted segments): second derivative of the distortion along u
_kD, _tD = sp.symbols('k_D t_D', real=True)
_xD = sp.Matrix(sp.symbols('xD_1 xD_2', real=True)); _uD = sp.Matrix(sp.symbols('uD_1 uD_2', real=True))
_pD = _xD + _tD * _uD
_DD = _pD * (1 + _kD * _pD.dot(_pD))
_second = sp.diff(_DD, _tD, 2).subs(_tD, 0)
check("the second derivative of x(1+k|x|^2) along u is k(2|u|^2 x + 4(x.u)u), of norm <= 6|k||x| for a unit u",
      "cor:chord", sp.simplify(_second - _kD * (2 * _uD.dot(_uD) * _xD + 4 * _xD.dot(_uD) * _uD)) == sp.zeros(2, 1))

# ------------------- Prop. (unknown principal point): K K^T = f^2 diag(1,1,0) + p p^T, so l_i^T w* l_j = f^2 A_ij + d_i d_j
_fp, _pxp, _pyp = sp.symbols('f_pp p_xpp p_ypp', positive=True)
_Kp = sp.Matrix([[_fp, 0, _pxp], [0, _fp, _pyp], [0, 0, 1]]); _pt = sp.Matrix([_pxp, _pyp, 1])
_li = sp.Matrix(sp.symbols('lpa_1 lpb_1 lpe_1', real=True)); _lj = sp.Matrix(sp.symbols('lpa_2 lpb_2 lpe_2', real=True))
check("l_i^T K K^T l_j = f^2 (a_i a_j + b_i b_j) + (l_i . p)(l_j . p) for a principal point p",
      "prop:pp", sp.expand((_li.T * _Kp * _Kp.T * _lj)[0] - (_fp**2 * (_li[0]*_lj[0] + _li[1]*_lj[1]) + _li.dot(_pt) * _lj.dot(_pt))) == 0)

# ------------------- Prop. (noise): the Lipschitz constant of a form in the two angles
_aL, _bL = sp.symbols('a_L b_L', real=True)
check("cos^2 a sin^2 b + sin^2 a cos^2 b = (sin^2(a+b) + sin^2(a-b))/2 <= 1, so the rho^2 part of the gradient "
      "has norm at most rho^2, and the form is Lipschitz with constant sqrt2 f^2 + rho^2",
      "eq:rigorous", sp.simplify(sp.expand_trig(sp.cos(_aL)**2*sp.sin(_bL)**2 + sp.sin(_aL)**2*sp.cos(_bL)**2
                                               - (sp.sin(_aL+_bL)**2 + sp.sin(_aL-_bL)**2)/2)) == 0)

# ------------------- Thm. (several frames): touching axes
_a1, _a2, _a3 = sp.symbols('a_r1 a_r2 a_r3', real=True)
_Rz2 = lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]])
_Rx2 = lambda a: sp.Matrix([[1, 0, 0], [0, sp.cos(a), -sp.sin(a)], [0, sp.sin(a), sp.cos(a)]])
_Rt = _Rz2(_a1) * _Rx2(_a2) * _Rz2(_a3)
_zi2, _zj2 = sp.symbols('zeta_i2 zeta_j2', real=True)
_ok_touch = True
for (_i, _j, _k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _r0 = _zi2 * _Rt[:, _i] + _zj2 * _Rt[:, _j]
    _ok_touch &= sp.simplify(_zj2 * _Rt[:, _i] - _zi2 * _Rt[:, _j] + _Rt[:, _k].cross(_r0)) == sp.zeros(3, 1)
check("on l_k, r0 = zeta_i d_i + zeta_j d_j and zeta_j d_i - zeta_i d_j = -(d_k x r0): "
      "the touch condition reads (Xi d_k).(d_k x r0) = 0", "thm:frames|, touching axes", _ok_touch)
_s3 = sp.Symbol('s_T', positive=True); _dT = sp.Matrix(sp.symbols('dT1 dT2 dT3', real=True))
_rT = sp.Matrix(sp.symbols('rT1 rT2 rT3', real=True)); _zT = sp.Matrix([0, 0, 1])
_XiT2 = _s3**2 * sp.eye(3) + (1 - _s3**2) * _zT * _zT.T
check("focal length alone: (Xi d).(d x r0) = (1 - s^2) delta zhat.(d x r0), and zhat.(d x r0) = 0 "
      "at r0 = zhat - delta d, the ray of c*", "thm:frames|, touching axes",
      sp.expand((_XiT2 * _dT).dot(_dT.cross(_rT)) - (1 - _s3**2) * _dT[2] * _zT.dot(_dT.cross(_rT))) == 0
      and sp.expand(_zT.dot(_dT.cross(_zT - _dT[2] * _dT))) == 0)

# ------------------- Lemma (sign condition): the converse direction
_g12, _g23, _g31, _R2s = sp.symbols('g12 g23 g31 R2s', nonzero=True)
_k1sq = -_g12 * _g31 / (_g23 * _R2s)                           # eq. (k2)
_k2k3 = (_g12 / _R2s) * (_g31 / _R2s) / _k1sq                  # k2 = -g12/(k1 R2), k3 = -g31/(k1 R2)
check("with k1^2 from the closed form and k2 = -g12/(k1|r|^2), k3 = -g31/(k1|r|^2): "
      "k2 k3 = -g23/|r|^2, so all three orthogonality equations hold",
      "lem:sign", sp.simplify(_k2k3 + _g23 / _R2s) == 0)

# ------------------- Theorem (the admissible cameras form a simplicial cone)
# Three lines through c, written l_k = c x p_k; generators m_k = c x l_k.
_c = sp.Matrix(sp.symbols('c1 c2 c3', real=True))
_pts = [sp.Matrix(sp.symbols(f'p{k}1 p{k}2 p{k}3', real=True)) for k in (1, 2, 3)]
_nu = sp.symbols('nu1:4', real=True)
_ln = [_c.cross(q) for q in _pts]
_mk = [_c.cross(l) for l in _ln]
_Qs = sp.zeros(3, 3)
for _k in range(3):
    _Qs += _nu[_k] * _mk[_k] * _mk[_k].T
_Dm = lambda k, i: sp.Matrix.hstack(_c, _ln[k], _ln[i]).det()
_trip = ((0, 1, 2), (1, 2, 0), (2, 0, 1))                    # (i, j, k)
check("with w* = sum_k nu_k m_k m_k^T on c-perp: l_i^T w* l_j = nu_k Delta_ki Delta_kj, "
      "Delta_ki = det(c, l_k, l_i); each pair form sees one generator only",
      "thm:convex|, generators",
      all(sp.expand((_ln[i].T * _Qs * _ln[j])[0, 0] - _nu[k] * _Dm(k, i) * _Dm(k, j)) == 0
          for i, j, k in _trip))
check("Delta_ki = -Delta_ik, so the product of the three forms is -nu1 nu2 nu3 (Delta12 Delta23 Delta31)^2",
      "thm:convex|, sign of the product",
      all(sp.expand(_Dm(k, i) + _Dm(i, k)) == 0 for i, j, k in _trip))
check("l_k^T w* l_k = nu_i Delta_ik^2 + nu_j Delta_jk^2: positive-definiteness excludes two negative nu",
      "thm:convex|, all nu positive",
      all(sp.expand((_ln[k].T * _Qs * _ln[k])[0, 0] - _nu[i] * _Dm(i, k)**2 - _nu[j] * _Dm(j, k)**2) == 0
          for i, j, k in _trip))

# ------------------- Lemma (a junction of the frame), in the basis of the vanishing points
_vv = sp.Matrix(sp.symbols('vq1 vq2 vq3', real=True)); _yy = sp.Matrix(sp.symbols('yq1 yq2 yq3', real=True))
_xp = _c.cross(_yy)                                            # an arbitrary vector of c-perp
check("m_k = c x (c x v_k) acts on c-perp as -|c|^2 v_k",
      "lem:frame", sp.expand(_c.cross(_c.cross(_vv)).dot(_xp) + _c.dot(_c) * _vv.dot(_xp)) == 0)
_ze = sp.symbols('zeta1:4', nonzero=True); _et = sp.symbols('eta1:4', real=True); _nf = sp.symbols('nuf1:4', real=True)
_Nm = sp.diag(*_nf) + sp.Matrix(_ze) * sp.Matrix(_et).T + sp.Matrix(_et) * sp.Matrix(_ze).T
def _nu_frame(a):
    p_, q_ = [b for b in range(3) if b != a]
    return (_Nm[a, a] - _ze[a] / _ze[p_] * _Nm[a, p_] - _ze[a] / _ze[q_] * _Nm[a, q_]
            + _ze[a]**2 / (_ze[p_] * _ze[q_]) * _Nm[p_, q_])
check("N = diag(nu) + zeta eta^T + eta zeta^T  =>  "
      "nu_a = N_aa - (zeta_a/zeta_p) N_ap - (zeta_a/zeta_q) N_aq + (zeta_a^2/(zeta_p zeta_q)) N_pq",
      "eq:nuframe", all(sp.simplify(_nu_frame(a) - _nf[a]) == 0 for a in range(3)))

# ------------------- Thm. (what one frame forces): the two coefficients near a vanishing line
_zi, _zj, _zk = sp.symbols('z_i z_j z_k', nonzero=True)
_Nii, _Njj, _Nkk, _Nij, _Nik, _Njk = sp.symbols('N_ii N_jj N_kk N_ij N_ik N_jk', real=True)
_Bq = (_zi / _zj) * _Njk - _Nik
_nui = _Nii - (_zi / _zj) * _Nij - (_zi / _zk) * _Nik + (_zi**2 / (_zj * _zk)) * _Njk
_nuj = _Njj - (_zj / _zi) * _Nij - (_zj / _zk) * _Njk + (_zj**2 / (_zi * _zk)) * _Nik
check("nu_i = N_ii - (zeta_i/zeta_j) N_ij + (zeta_i/zeta_k) B  and  "
      "nu_j = N_jj - (zeta_j/zeta_i) N_ij - (zeta_j^2/(zeta_i zeta_k)) B,  B = (zeta_i/zeta_j) N_jk - N_ik",
      "thm:converse|, part (1)",
      sp.simplify(_nui - (_Nii - (_zi / _zj) * _Nij + (_zi / _zk) * _Bq)) == 0
      and sp.simplify(_nuj - (_Njj - (_zj / _zi) * _Nij - (_zj**2 / (_zi * _zk)) * _Bq)) == 0)
check("the two coefficients of B have opposite signs: their product is -(zeta_j/zeta_k)^2",
      "thm:converse|, part (1)",
      sp.simplify((_zi / _zk) * (-(_zj**2) / (_zi * _zk)) + (_zj / _zk)**2) == 0)

# ------------------- Theorems (identifiability of f; several frames): a general rotation D
_e1, _e2, _e3 = sp.symbols('euler1:4', real=True)
_Rz = lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]])
_Rx = lambda a: sp.Matrix([[1, 0, 0], [0, sp.cos(a), -sp.sin(a)], [0, sp.sin(a), sp.cos(a)]])
_Dr = _Rz(_e1) * _Rx(_e2) * _Rz(_e3)
_xs, _tt = sp.symbols('x_s t_s', real=True)
_Nf = _Dr.T * sp.diag(_xs / f**2, _xs / f**2, 1) * _Dr        # K = diag(f,f,1), w* = diag(x,x,1), V = K D
_zr = _Dr[2, :]                                               # z_a = d_a . z-hat
check("w* = diag(x,x,1), K = diag(f,f,1):  N = (x/f^2) I + (1 - x/f^2) z z^T, "
      "so N_ab = (1 - x/f^2) z_a z_b off the diagonal",
      "thm:identframe",
      sp.simplify(_Nf - (_xs / f**2) * sp.eye(3) - (1 - _xs / f**2) * _zr.T * _zr) == sp.zeros(3, 3))
# ------------------- Prop. (joint admissible set in closed form) and the exceptional point
_sg = sp.Symbol('s_ratio', real=True); _zz = sp.symbols('zc1:4', real=True); _zt = sp.symbols('zt1:4', nonzero=True)
_Ng = _sg * sp.eye(3) + (1 - _sg) * sp.Matrix(_zz) * sp.Matrix(_zz).T
def _nu_g(a):
    p_, q_ = [b for b in range(3) if b != a]
    nu = (_Ng[a, a] - _zt[a] / _zt[p_] * _Ng[a, p_] - _zt[a] / _zt[q_] * _Ng[a, q_]
          + _zt[a]**2 / (_zt[p_] * _zt[q_]) * _Ng[p_, q_])
    g = (_zz[a] - _zt[a] * _zz[p_] / _zt[p_]) * (_zz[a] - _zt[a] * _zz[q_] / _zt[q_])
    return sp.simplify(nu - (_sg + (1 - _sg) * g))
check("N = s I + (1-s) z z^T in the frame lemma gives nu_a = s + (1-s) gamma_a, "
      "gamma_a = (z_a - zeta_a w_p)(z_a - zeta_a w_q), w = z/zeta",
      "eq:gform", all(_nu_g(a) == 0 for a in range(3)))
_dk = sp.Matrix(sp.symbols('dk1 dk2 dk3', real=True))
_rstar = sp.Matrix([0, 0, 1]) - _dk[2] * _dk                    # z-hat - z_k d_k
check("the ray z-hat - z_k d_k is perpendicular to d_k for unit d_k, and its image is "
      "parallel to (d_kx, d_ky): the foot of the perpendicular from the principal point",
      "thm:identframe|, exceptional point",
      sp.simplify(_rstar.dot(_dk).subs(_dk[2]**2, 1 - _dk[0]**2 - _dk[1]**2)) == 0
      and sp.simplify(_rstar[0] * _dk[1] - _rstar[1] * _dk[0]) == 0)

_d3 = _Dr[:, 2]
check("D diag(1,1,1+t) D^T = I + t d d^T for a frame D with third axis d: "
      "frames sharing an axis leave K (I + t d d^T) K^T",
      "thm:frames",
      sp.simplify(_Dr * sp.diag(1, 1, 1 + _tt) * _Dr.T - sp.eye(3) - _tt * _d3 * _d3.T) == sp.zeros(3, 3))

# ---------------- identifiability: when does a threshold hit the true f exactly?
_r = sp.Matrix(sp.symbols('rr1 rr2 rr3', real=True))
_di = sp.Matrix(sp.symbols('pp1 pp2 pp3', real=True))
_dj = sp.Matrix(sp.symbols('qq1 qq2 qq3', real=True))
check("(r x d_i).(r x d_j) = |r|^2 (d_i.d_j) - (r.d_i)(r.d_j); for orthogonal "
      "edges the pair form at the true f is -(r.d_i)(r.d_j), zero iff an edge _|_ ray",
      "eq:edgeon",
      sp.simplify(sp.expand((_r.cross(_di)).dot(_r.cross(_dj))
                  - (_r.dot(_r) * _di.dot(_dj) - _r.dot(_di) * _r.dot(_dj)))) == 0)
_al, _be, _ga = sp.symbols('alpha_e beta_e gamma_e', real=True)
def _rot(ax, t):
    c, s = sp.cos(t), sp.sin(t)
    return {'z': sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
            'y': sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])}[ax]
_Rf = _rot('z', _al) * _rot('y', _be) * _rot('z', _ga)      # a general rotation
check("for ANY orthonormal frame, sum_i (d_i . z)^2 = 1 (columns d_i of a rotation), "
      "so min_i |d_i . z| <= 1/sqrt3",
      "thm:identframe|, corollary",
      sp.simplify(sum((_Rf[:, i].dot(sp.Matrix([0, 0, 1])))**2 for i in range(3)) - 1) == 0)

# --------------- Prop. (one frame): back-projected VP directions vs the IAC
_k = sp.symbols('k11 k12 k13 k22 k23', real=True)
_Kp = sp.Matrix([[_k[0], _k[1], _k[2]], [0, _k[3], _k[4]], [0, 0, 1]])
_vi = sp.Matrix(sp.symbols('vi1 vi2 vi3', real=True))
_vj = sp.Matrix(sp.symbols('vj1 vj2 vj3', real=True))
_di = _Kp.inv() * _vi; _dj = _Kp.inv() * _vj
check("(K'^-1 v_i).(K'^-1 v_j) = v_i^T (K' K'^T)^-1 v_j : orthogonal back-projected "
      "VP directions <=> v_i^T w' v_j = 0", "prop:oneframe",
      sp.simplify(_di.dot(_dj) - (_vi.T * (_Kp * _Kp.T).inv() * _vj)[0, 0]) == 0)

# --------------- Prop. (rate): the edge-on locus is a straight line in the image
_u, _vv, _fz = sp.symbols('u_img v_img f_cam', real=True)
_dd = sp.Matrix(sp.symbols('dd1 dd2 dd3', real=True))
_ray = sp.Matrix([_u, _vv, _fz])                       # un-normalised ray of pixel (u,v)
_lin = sp.Poly(sp.expand(_ray.dot(_dd)), _u, _vv)
check("r . d_i = u d_x + v d_y + f d_z is affine in (u,v): the edge-on locus is a line",
      "prop:rate", _lin.total_degree() == 1)

# ------------- Thm. (what one frame forces): axis k's edge-on line is the
# vanishing line through v_i, v_j.
_c3 = sp.Matrix(sp.symbols('cc1 cc2 cc3', real=True))
_Kq = sp.Matrix([[sp.Symbol('q11'), sp.Symbol('q12'), sp.Symbol('q13')],
                 [0, sp.Symbol('q22'), sp.Symbol('q23')], [0, 0, 1]])
_va = sp.Matrix(sp.symbols('va1 va2 va3', real=True))
_vb = sp.Matrix(sp.symbols('vb1 vb2 vb3', real=True))
_Wq = (_Kq * _Kq.T).inv()                      # omega
_rho = _Kq.inv() * _c3
_da = _Kq.inv() * _va; _db = _Kq.inv() * _vb
# axis k's edge-on line w v_k = K^-T d_k; the vanishing line v_i x v_j = (K d_i)x(K d_j).
# The cofactor identity (Ka)x(Kb) = det(K) K^-T (a x b), with d_i x d_j = d_k for a
# right-handed frame, makes them the same line.
_pa = sp.Matrix(sp.symbols('pa1 pa2 pa3', real=True))
_pb = sp.Matrix(sp.symbols('pb1 pb2 pb3', real=True))
check("cofactor identity (K a) x (K b) = det(K) K^-T (a x b) for any K",
      "thm:converse|, edge-on line = vanishing line",
      sp.simplify((_Kq * _pa).cross(_Kq * _pb) - _Kq.det() * _Kq.T.inv() * _pa.cross(_pb))
      == sp.zeros(3, 1))
check("so with v_m = K d_m: w v_k = K^-T d_k equals (v_i x v_j)/det(K) when d_i x d_j = d_k",
      "thm:converse|, edge-on line = vanishing line",
      sp.simplify(_Wq * (_Kq * _pa.cross(_pb)) - _Kq.T.inv() * _pa.cross(_pb)) == sp.zeros(3, 1))

# ------------- Thm. (identifiability): an axis parallel to the image plane.
# Level camera: d_i = (0,1,0); d_j, d_k horizontal at yaw t.  The thresholds of
# the pairs containing i depend on the junction's abscissa u alone.
_t, _Zc = sp.symbols('t_yaw Z_c', positive=True)
_Pl = K.inv() * c * _Zc
_Dl = [sp.Matrix([0, 1, 0]), sp.Matrix([sp.cos(_t), 0, sp.sin(_t)]),
       sp.Matrix([-sp.sin(_t), 0, sp.cos(_t)])]
def _line_l(d):
    q = K * (_Pl + d); q = q / q[2]
    ex, ey = q[0] - u, q[1] - v
    return sp.Matrix([ey, -ex, -(ey * u - ex * v)])
_Ll = [_line_l(d) for d in _Dl]
def _thr_l(i, j):
    return sp.simplify(-(_Ll[i][2] * _Ll[j][2]) / (_Ll[i][0] * _Ll[j][0] + _Ll[i][1] * _Ll[j][1]))
_xi_j = f * sp.cos(_t) / sp.sin(_t); _xi_k = -f * sp.sin(_t) / sp.cos(_t)
_x01, _x02 = _thr_l(0, 1), _thr_l(0, 2)
check("axis parallel to the image plane: x_ij = -u xi_j and x_ik = -u xi_k, "
      "independent of the distance v from that axis's edge-on line",
      "thm:identframe|, excluded axis",
      sp.simplify(_x01 + u * _xi_j) == 0 and sp.simplify(_x02 + u * _xi_k) == 0
      and sp.diff(_x01, v) == 0 and sp.diff(_x02, v) == 0)
check("xi_j xi_k = -f^2, so x_ij = f^2 only at u = xi_k, on the edge-on line of axis j",
      "thm:identframe|, excluded axis",
      sp.simplify(_xi_j * _xi_k + f**2) == 0 and sp.simplify(_x01.subs(u, _xi_k) - f**2) == 0)

# ----------------------- Prop. 5, three non-concurrent lines: at most two frames
_al = sp.symbols('alpha', real=True)
_n2 = sp.Matrix(sp.symbols('n2x n2y n2z', real=True))
_n3 = sp.Matrix(sp.symbols('n3x n3y n3z', real=True))
_h1 = sp.Matrix(sp.symbols('h1x h1y h1z', real=True))
_h2 = sp.Matrix(sp.symbols('h2x h2y h2z', real=True))
_d1 = _h1 * sp.cos(_al) + _h2 * sp.sin(_al)
_cond = sp.expand(_n2.dot(_n3) - (_n3.dot(_d1)) * (_n2.dot(_d1)))
_p = sp.Poly(_cond, sp.cos(_al), sp.sin(_al))
check("the last constraint is a quadratic form in (cos a, sin a), so at most "
      "four roots in [0,2pi) and, since a ~ a+pi, at most two frames",
      "prop:threelines", _p.total_degree() == 2)

# ------------------------------------------------------- Remark 1 (radial edge)
# e1 = 0 makes two of the three factors proportional to x, so
#     L(x) = A12 A31 x^2 (A23 x + e2 e3)
# which has at most the single positive threshold -e2 e3 / A23.
Lr = sp.expand(L.subs(e1, 0))
A23 = a2 * a3 + b2 * b3
A31 = a3 * a1 + b3 * b1
check("with e1 = 0, L(x) = A12 A31 x^2 (A23 x + e2 e3): one threshold at most",
      "rem:exceptional",
      sp.simplify(Lr - A12 * A31 * x**2 * (A23 * x + e2 * e3)) == 0)

# at the principal point every e_i vanishes and L has constant sign
check("at the principal point L(x) = A12 A23 A31 x^3, of constant sign",
      "rem:exceptional",
      sp.simplify(L.subs({e1: 0, e2: 0, e3: 0})
                  - A12 * A23 * A31 * x**3) == 0)

# ----------------------------------------------------------------- Section 5.2
# w = n3 x n3' = (f p, f q, f^2 s) with v3 = l3 x l3' = (p, q, s).

a3p, b3p, e3p = sp.symbols('a3p b3p e3p', real=True)
n3 = normal(a3, b3, e3)
n3p = normal(a3p, b3p, e3p)
w = sp.expand(n3.cross(n3p))
v3 = sp.Matrix([a3, b3, e3]).cross(sp.Matrix([a3p, b3p, e3p]))
p_, q_, s_ = v3
check("w = (f p, f q, f^2 s) with (p,q,s) = l3 x l3'", "sec:elimination",
      sp.simplify(w - sp.Matrix([f * p_, f * q_, f**2 * s_])) == sp.zeros(3, 1))

n1 = normal(a1, b1, e1)
n2 = normal(a2, b2, e2)
E = sp.expand((n1.dot(n2)) * (w.dot(w)) - (n1.dot(w)) * (n2.dot(w)))

degs = sorted({m[0] for m in sp.Poly(E, f).monoms()})
check(f"E is even in f, degree 6 (exponents {degs})", "sec:elimination",
      degs == [2, 4, 6])

A12 = a1 * a2 + b1 * b2
m1 = sp.Matrix([a1, b1, e1]).dot(v3)
m2 = sp.Matrix([a2, b2, e2]).dot(v3)
c2 = A12 * s_**2
c1 = A12 * (p_**2 + q_**2) + e1 * e2 * s_**2 - m1 * m2
c0 = e1 * e2 * (p_**2 + q_**2)
Ex = sp.expand(E.subs(f, sp.sqrt(x)))
check("E = x (c2 x^2 + c1 x + c0) with the stated coefficients",
      "eq:quadratic",
      sp.simplify(Ex - sp.expand(x * (c2 * x**2 + c1 * x + c0))) == 0)
check("x = 0 is a root of E (spurious: w vanishes at f = 0)", "sec:elimination",
      sp.simplify(Ex.subs(x, 0)) == 0
      and sp.simplify(w.subs(f, 0)) == sp.zeros(3, 1))

d3 = w / sp.sqrt(w.dot(w))
cleared = sp.simplify(((n1.T * (sp.eye(3) - d3 * d3.T) * n2)[0, 0]) * w.dot(w) - E)
check("E is n1^T (I - d3 d3^T) n2 = 0 with the denominator cleared",
      "eq:E", sp.simplify(cleared) == 0)

# n_i . w = f^2 (l_i . v3), which is why E is even and factors through x.
check("n_i . w = f^2 (l_i . v3)", "sec:elimination",
      sp.simplify(sp.expand(n1.dot(w) - f**2 * m1)) == 0
      and sp.simplify(sp.expand(n2.dot(w) - f**2 * m2)) == 0)


# ------------------------------------- Section 5.2, the geometric form
# E = x s^2 [ (A12 x + e1 e2)(x + |v3|^2) - d1 d2 x ], where e_i and d_i are
# the signed distances to line i from the principal point and from the
# vertical vanishing point.
V2 = (p_**2 + q_**2) / s_**2
dd1 = (a1 * p_ + b1 * q_) / s_ + e1
dd2 = (a2 * p_ + b2 * q_) / s_ + e2
check("E = x s^2 [ (A12 x + e1 e2)(x + |v3|^2) - d1 d2 x ]",
      "eq:geosolver",
      sp.simplify(sp.together(Ex - sp.expand(
          x * s_**2 * ((A12 * x + e1 * e2) * (x + V2) - dd1 * dd2 * x)))) == 0)
check("d_i = (a_i, b_i) . (v3 - c), the signed distance from the vertical "
      "vanishing point to line i", "sec:elimination",
      sp.simplify(sp.expand(sp.together(
          (dd1 - (a1 * (p_/s_ - u) + b1 * (q_/s_ - v))).subs(
              {e1: -(a1*u + b1*v)})))) == 0)
check("if d1 d2 = 0 the only positive root is the single-corner threshold "
      "-e1 e2 / A12 of Thm. 1", "sec:elimination",
      sp.simplify(((A12 * x + e1 * e2) * (x + V2)).subs(
          x, -e1 * e2 / A12)) == 0)

# --------------------------------------------------- Section 5.3, gravity
# With gravity known, d3 = g is FIXED in camera coordinates: unlike w it does
# not scale with f, so n1^T (I - g g^T) n2 is a quadratic in f and not in f^2.
# The odd term is -f g_z (gamma_1 e_2 + gamma_2 e_1), gamma_i = a_i gx + b_i gy.
gx, gy, gz = sp.symbols('gx gy gz', real=True)
g_vec = sp.Matrix([gx, gy, gz])
G = sp.expand(n1.dot(n2) - (n1.dot(g_vec)) * (n2.dot(g_vec)))
Pg = sp.Poly(G, f)
gamma1 = a1 * gx + b1 * gy
gamma2 = a2 * gx + b2 * gy
check("gravity condition is a quadratic in f (degrees 0, 1, 2 present)",
      "sec:variants",
      sorted({m[0] for m in Pg.monoms()}) == [0, 1, 2])
check("its odd coefficient is -g_z (gamma_1 e_2 + gamma_2 e_1), nonzero in "
      "general, so the condition is not a polynomial in f^2", "sec:variants",
      sp.simplify(Pg.coeff_monomial(f)
                  + gz * (gamma1 * e2 + gamma2 * e1)) == 0)

# The vertical line gives f linearly: n3 . g = f (a3 gx + b3 gy) + e3 gz.
check("n3 . g = f (a3 gx + b3 gy) + e3 gz is linear in f", "sec:variants",
      sp.simplify(n3.dot(g_vec)
                  - (f * (a3 * gx + b3 * gy) + e3 * gz)) == 0)

# A gravity sensor does not give the horizon in pixels: K^-T g depends on f.
check("the imaged horizon K^-T g is proportional to (gx, gy, f gz), so it "
      "still depends on f", "sec:variants",
      sp.simplify(sp.Matrix(K.inv().T * g_vec) * f
                  - sp.Matrix([gx, gy, f * gz])) == sp.zeros(3, 1))


# ------------------------------------------------------- Theorem 1 (numeric)
# Hypothesis pairs are (G1 R, G2 R D) with G_i in {I, H_i}; the pair is
# consistent when the two frames agree up to a diagonal sign matrix.
# For NONDEGENERATE corners: only the true pair is consistent unless the rays
# are parallel, in which case the double flip is consistent as well.

rng = np.random.default_rng(0)


def H(rv):
    rv = np.asarray(rv, float)
    rv = rv / np.linalg.norm(rv)
    return np.eye(3) - 2 * np.outer(rv, rv)


def rand_frame():
    A = rng.standard_normal((3, 3))
    Qm, Rm = np.linalg.qr(A)
    Qm = Qm @ np.diag(np.sign(np.diag(Rm)))
    if np.linalg.det(Qm) < 0:
        Qm[:, 0] *= -1
    return Qm


def consistent(F1, F2, tol=1e-9):
    return np.allclose(np.abs(F1.T @ F2), np.eye(3), atol=tol)


def degenerate(Pt, Fr, tol=1e-3):
    """An edge along the ray, or two edges sharing one image line."""
    rv = np.asarray(Pt, float)
    rv = rv / np.linalg.norm(rv)
    for i in range(3):
        if abs(abs(rv @ Fr[:, i]) - 1) < tol:
            return True
    for i, j in ((0, 1), (1, 2), (2, 0)):
        ni = np.cross(rv, Fr[:, i])
        nj = np.cross(rv, Fr[:, j])
        if np.linalg.norm(np.cross(ni, nj)) < tol:
            return True
    return False


def hypothesis_table(P1, P2, Fr, D):
    H1, H2 = H(P1), H(P2)
    return tuple(consistent(G1 @ Fr, G2 @ (Fr @ D))
                 for G1 in (np.eye(3), H1) for G2 in (np.eye(3), H2))


ONLY_TRUE = (True, False, False, False)
DOUBLE_FLIP = (True, False, False, True)

n_generic = 0
ok_generic = True
for _ in range(2000):
    Fr = rand_frame()
    D = np.diag(rng.choice([-1.0, 1.0], 3))
    P1 = rng.uniform(-2, 2, 3) + np.array([0, 0, 4.0])
    P2 = rng.uniform(-2, 2, 3) + np.array([0, 0, 4.0])
    if degenerate(P1, Fr) or degenerate(P2, Fr @ D):
        continue
    if np.linalg.norm(np.cross(P1 / np.linalg.norm(P1),
                               P2 / np.linalg.norm(P2))) < 1e-6:
        continue
    n_generic += 1
    ok_generic &= hypothesis_table(P1, P2, Fr, D) == ONLY_TRUE
check(f"non-parallel rays: only the true pair is consistent "
      f"({n_generic} nondegenerate pairs)", "thm:interval", ok_generic)

n_par = 0
ok_par = True
for _ in range(500):
    Fr = rand_frame()
    D = np.diag(rng.choice([-1.0, 1.0], 3))
    P1 = rng.uniform(-2, 2, 3) + np.array([0, 0, 4.0])
    P2 = rng.uniform(1.2, 3.0) * P1          # same ray
    if degenerate(P1, Fr) or degenerate(P2, Fr @ D):
        continue
    n_par += 1
    ok_par &= hypothesis_table(P1, P2, Fr, D) == DOUBLE_FLIP
check(f"parallel rays: the double flip is consistent too, the single flips "
      f"are not ({n_par} pairs)", "thm:interval", ok_par)

# The configuration that the algebraic argument appears to allow -- rays 90
# degrees apart with r1 x r2 along a frame axis, so that H2 H1 = R D' R^T --
# does not arise for nondegenerate corners: both rays and the other two axes
# lie in the plane perpendicular to that axis, so at EACH corner those two
# edges share one image line.  This is why Theorem 1 has a single exception.
ok_alg, ok_deg = True, True
n_perp = 0
for _ in range(500):
    Fr = rand_frame()
    k = int(rng.integers(3))
    dk = Fr[:, k]
    Dp = -np.ones(3)
    Dp[k] = 1.0                       # 180 degrees about d_k
    Dp = np.diag(Dp)
    ek = np.cross(dk, rng.standard_normal(3))
    ek /= np.linalg.norm(ek)
    P1, P2 = 3.0 * ek, 3.0 * np.cross(dk, ek)
    n_perp += 1
    ok_alg &= np.abs(H(P2) @ H(P1) - Fr @ Dp @ Fr.T).max() < 1e-12
    ok_deg &= degenerate(P1, Fr) and degenerate(P2, Fr)
check(f"perpendicular rays with r1 x r2 along a frame axis satisfy "
      f"H2 H1 = R D' R^T ({n_perp} configurations)", "thm:interval|, discussion",
      ok_alg)
check("...but both corners are then degenerate (two edges share one image "
      "line), so this is not an exception to Thm. 1", "thm:interval|, discussion",
      ok_deg)


# ---------------------------- Proposition 2 and the interval theorem (numeric)
def corner_lines(Pt, Fr, foc):
    """Image point and the three unit line vectors (a, b, e) of a corner."""
    Km = np.diag([foc, foc, 1.0])
    cp = Km @ Pt
    cp = cp / cp[2]
    out = []
    for i in range(3):
        cq = Km @ (Pt + 1e-6 * Fr[:, i])
        cq = cq / cq[2]
        d = (cq - cp)[:2]
        d = d / np.linalg.norm(d)
        a, b = d[1], -d[0]
        out.append(np.array([a, b, -(a * cp[0] + b * cp[1])]))
    return cp, out


def coefficients(cp, L):
    """The three linear factors (A_ij, e_i e_j), with the lines re-projected
    through the junction so that n_i . r = 0 holds exactly."""
    L = [np.array([l[0], l[1], -(l[0] * cp[0] + l[1] * cp[1])]) for l in L]
    out = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        ai, bi, ei = L[i]
        aj, bj, ej = L[j]
        out.append((ai * aj + bi * bj, ei * ej))
    return out


def positive_roots(co):
    return sorted(-E / A for A, E in co if abs(A) > 1e-300 and -E / A > 1e-13)


N_SEARCH = 500 if QUICK else 20000
n_fork = n_corner = 0
ok_fork = True
max_components = 1
parity_ok = True
for _ in range(N_SEARCH):
    foc = rng.uniform(300, 1600)
    Fr = rand_frame()
    Pt = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(1.5, 8)])
    cp, L = corner_lines(Pt, Fr, foc)
    if max(abs(cp[0]), abs(cp[1])) > 1e5:
        continue
    n_corner += 1

    # Prop. 2: all three edges share a sign  <=>  all three g_ij < 0 (a fork)
    rv = np.linalg.inv(np.diag([foc, foc, 1.0])) @ cp
    tv = [np.cross(np.diag([foc, foc, 1.0]).T @ l, rv) for l in L]
    alpha = [Fr[:, i] @ rv for i in range(3)]
    same_sign = all(a > 0 for a in alpha) or all(a < 0 for a in alpha)
    all_neg = all(tv[i] @ tv[j] < 0 for i, j in ((0, 1), (1, 2), (2, 0)))
    if same_sign:
        n_fork += 1
    ok_fork &= (same_sign == all_neg)

    # Conjecture 1: count the sign changes of Q on (0, inf)
    co = coefficients(cp, L)
    rs = positive_roots(co)
    edges = [0.0] + rs + [(rs[-1] if rs else 1.0) * 1e8 + 1.0]
    signs = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        xm = np.sqrt(max(lo, hi * 1e-14) * hi)
        signs.append(np.sign(np.prod([A * xm + E for A, E in co])))
    changes = sum(1 for k in range(len(signs) - 1)
                  if signs[k] * signs[k + 1] < 0)
    prodA = np.prod([A for A, _ in co])
    parity_ok &= ((changes % 2 == 1) == (prodA < 0))
    max_components = max(max_components, changes // 2 + (changes % 2))

check(f"uniform edge sign <=> fork pattern, i.e. all g_ij < 0 "
      f"({n_fork} uniform-sign corners out of {n_corner})", "prop:concave", ok_fork)
check(f"parity: Q has an odd number of sign changes exactly when "
      f"A12 A23 A31 < 0 ({n_corner} corners)", "prop:bound|(2)", parity_ok)
check(f"F is a single interval ({n_corner} corners)", "thm:interval", max_components == 1)


# ----------------------------------------------------------------- summary
print()
bad = [row for row in results if not row[2]]
print(f"{len(results) - len(bad)}/{len(results)} checks passed")
if bad:
    for name, where, _ in bad:
        print(f"  FAILED -- {where}: {name}")
sys.exit(1 if bad else 0)
