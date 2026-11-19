import numpy as np

def rand_frame(rng):
    A = rng.standard_normal((3,3)); Q,R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0: Q[:,0] *= -1
    return Q

def Kmat(f): return np.diag([f,f,1.0])

def project(P, f):
    c = Kmat(f) @ P
    return c[:2]/c[2]

def edge_image_dir(P, d, f):
    """Exact image direction of the 3D direction d at the projection of P:
    the derivative of the projection along d, not the vanishing point."""
    X, Y, Z = P
    u0, v0 = f*X/Z, f*Y/Z
    return np.array([f*d[0] - u0*d[2], f*d[1] - v0*d[2]]) / Z

def observe_corner(P, D, f, sigma, rng, seglen=150.0, W=1000.0):
    """Image a corner: each edge is a seglen-px segment from the junction, both
    endpoints perturbed by N(0, sigma). Lines are fitted through the two noisy
    endpoints; the junction is the least-squares intersection of the three."""
    c0 = project(P, f)
    if np.max(np.abs(c0)) > W/2: return None
    lines = []
    for i in range(3):
        d = edge_image_dir(P, D[:, i], f)
        n = np.linalg.norm(d)
        if n < 1e-12: return None
        d = d/n
        p1 = c0 + rng.normal(0, sigma, 2)
        p2 = c0 + seglen*d + rng.normal(0, sigma, 2)
        e = p2 - p1; ne = np.linalg.norm(e)
        if ne < 1e-9: return None
        e = e/ne
        a, b = e[1], -e[0]                     # half-edge leaves c along (b,-a)
        lines.append(np.array([a, b, -(a*p1[0]+b*p1[1])]))
    L = np.array(lines)
    A = L[:, :2]; rhs = -L[:, 2]
    c, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return c, [np.array(l) for l in L]

def gij(c, L, f):
    r = np.linalg.inv(Kmat(f)) @ np.array([c[0], c[1], 1.0])
    t = [np.cross(Kmat(f).T @ l, r) for l in L]
    return t, r, np.array([[t[i]@t[j] for j in range(3)] for i in range(3)])

def closed_form(c, L, f):
    t, r, g = gij(c, L, f); R2 = r@r
    if abs(g[1,2])<1e-300 or abs(g[2,0])<1e-300 or abs(g[0,1])<1e-300: return None
    k1sq = -g[0,1]*g[2,0]/(g[1,2]*R2)
    if k1sq <= 0: return None
    k1 = np.sqrt(k1sq); k = [k1, -g[0,1]/(R2*k1), -g[2,0]/(R2*k1)]
    cols = []
    for i in range(3):
        w = k[i]*r + t[i]; n = np.linalg.norm(w)
        if n < 1e-12: return None
        cols.append(w/n)
    return np.column_stack(cols)

def reproject(c, L):
    """Force each line through the estimated junction, keeping its direction.

    The derivation needs every interpretation plane to contain the ray, i.e.
    n_i . r = 0, i.e. every image line through the corner point.  Three
    independently fitted noisy lines do not meet exactly, so the observed
    (a, b, e) are mutually inconsistent at the sub-pixel level.  This makes
    the observation consistent with the model before the model is applied.
    """
    return [np.array([l[0], l[1], -(l[0]*c[0] + l[1]*c[1])]) for l in L]


def coeffs(c, L, enforce=True):
    """The three LINEAR factors (A_ij, e_i e_j) of the sign condition.

    Because each interpretation plane contains the ray,
        x g_ij = (x + rho^2)(A_ij x + e_i e_j),     x = f^2,
    and the positive factor (x + rho^2) is common to all three pairs, so it
    cancels from the sign condition.  Each pair contributes one linear
    function of f^2, with at most one positive root.
    """
    if enforce:
        L = reproject(c, L)
    out = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        ai, bi, ei = L[i]
        aj, bj, ej = L[j]
        out.append((ai*aj + bi*bj, ei*ej))
    return out


def admissible(co, XMAX=1e14):
    """The admissible set as a list of intervals in x = f^2.  By the interval
    theorem there is at most one, but the sign chart is built generically so
    that a violation would show up rather than be assumed away."""
    rs = sorted(-E/A for A, E in co if abs(A) > 1e-300 and -E/A > 1e-12)
    edges = [0.0] + rs + [XMAX]
    comps = []
    for k in range(len(edges)-1):
        lo, hi = edges[k], edges[k+1]
        if hi <= lo:
            continue
        xm = np.sqrt(max(lo, hi*1e-16)*hi) if lo > 0 else min(hi/2, 1e12)
        if np.prod([A*xm + E for A, E in co]) < 0:
            if comps and abs(comps[-1][1]-lo) < 1e-9*max(1, lo):
                comps[-1] = (comps[-1][0], hi)
            else:
                comps.append((lo, hi))
    return comps


def angle_err_deg(Dh, D, c, f):
    """Compare as unoriented axes, allowing either mirror branch:
    the two closed-form solutions are D and H D with H the reflection in the
    plane through O perpendicular to the ray."""
    r = np.linalg.inv(Kmat(f)) @ np.array([c[0], c[1], 1.0])
    rh = r/np.linalg.norm(r)
    H = np.eye(3) - 2*np.outer(rh, rh)
    best = 1e9
    for S in (Dh, H @ Dh):
        errs = [np.degrees(np.arccos(np.clip(abs(S[:,i]@D[:,i]),-1,1))) for i in range(3)]
        best = min(best, float(np.mean(errs)))
    return best
