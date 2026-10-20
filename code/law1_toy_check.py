"""law1_toy_check.py -- numerical verification of docs/law1_derivation_formal.md.

Pure numpy, CPU, deterministic (seed 0), runs in ~1 s. No experiment data is used.

Linear loop h_{t+1} = J h_t + U theta + B x  =>  dF/dtheta = U (constant, A1 holds exactly).
Adjoint lambda_t = J^T lambda_{t+1}; contribution of the j-th loop counted back from the
output: c_j = U^T (J^T)^j lambda_R, j = 0..R-1.  g = sum_j c_j, g_k = sum_{j<k} c_j,
b_k = g - g_k, err_k = |b_k| / |g|,  SGD deficit D_k = (g . b_k) / |g|^2.
"""
import time
import numpy as np

T0 = time.time()
rng = np.random.default_rng(0)
TOL = 1e-10


# ---------------------------------------------------------------- helpers
def contributions(J, U, lam, R):
    """Return (C, adj) with C[j] = U^T (J^T)^j lam and adj[j] = |(J^T)^j lam|."""
    C, adj, a = [], [], lam.copy()
    for _ in range(R):
        C.append(U.T @ a)
        adj.append(np.linalg.norm(a))
        a = J.T @ a
    return np.array(C), np.array(adj)


def err_curve(C):
    g = C.sum(0)
    tails = np.cumsum(C[::-1], 0)[::-1]           # tails[k] = sum_{j>=k} c_j = b_k
    tails = np.vstack([tails, np.zeros_like(g)])   # b_R = 0
    return np.linalg.norm(tails, axis=1) / np.linalg.norm(g), tails, g


def deficit_curve(C):
    _, tails, g = err_curve(C)
    return tails @ g / (g @ g)


def rate(x, i0, i1):
    """Geometric-mean per-step decay of |x| between indices i0 < i1."""
    return (abs(x[i1]) / abs(x[i0])) ** (1.0 / (i1 - i0))


def P_formula(rho, k, R):   # parallel contributions
    return (rho ** k - rho ** R) / (1 - rho ** R)


def O_formula(rho, k, R):   # orthogonal contributions with geometric magnitudes
    return np.sqrt((rho ** (2 * k) - rho ** (2 * R)) / (1 - rho ** (2 * R)))


def rand_orth(n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def row(*cells, w=10):
    return " ".join(f"{c:>{w}}" if isinstance(c, str) else f"{c:>{w}.4g}" for c in cells)


ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ---------------------------------------------------------------- (i) parallel
print("=" * 78)
print("(i) symmetric J, dominant eigenvalue rho = 0.8, others |mu| <= 0.3")
n, p, R, rho = 6, 9, 30, 0.8
Q = rand_orth(n)
mu = np.array([rho, 0.3, 0.1, -0.2, 0.05, 0.0])
J = Q @ np.diag(mu) @ Q.T
U = rng.standard_normal((n, p))
C, _ = contributions(J, U, Q[:, 0].copy(), R)          # lambda_R = dominant eigenvector
e_par, _, _ = err_curve(C)
ks = np.arange(R + 1)
check("exact eigvec: err_k == (rho^k - rho^R)/(1 - rho^R) to 1e-10",
      np.max(np.abs(e_par - P_formula(rho, ks, R))) < TOL)
C2, adj2 = contributions(J, U, rng.standard_normal(n), R)   # generic lambda_R
e_gen, _, _ = err_curve(C2)
print(row("k", "rho^k", "exact-eig", "generic"))
for k in [1, 2, 4, 8, 12, 16]:
    print(row(k, rho ** k, e_par[k], e_gen[k]))
print(f"  generic lambda_R: adjoint rate (j 8..20) = {rate(adj2, 8, 20):.4f}, "
      f"err rate (k 4..16) = {rate(e_gen, 4, 16):.4f}")

# ---------------------------------------------------------------- (ii) orthogonal
print("=" * 78)
print("(ii) J^T = rho * cyclic shift, U = I, lambda_R = e_1  -> mutually orthogonal c_j")
R = n = 20
rho = 0.8
S = np.roll(np.eye(n), 1, axis=0)                      # S e_i = e_{i+1}
J = (rho * S).T
C, _ = contributions(J, np.eye(n), np.eye(n)[0], R)
G = C @ C.T
check("Gram matrix of contributions is diagonal", np.max(np.abs(G - np.diag(np.diag(G)))) < TOL)
e_orth, _, _ = err_curve(C)
check("err_k == sqrt((rho^2k - rho^2R)/(1 - rho^2R)) to 1e-10",
      np.max(np.abs(e_orth - O_formula(rho, ks[:R + 1], R))) < TOL)
Pk, Ok = P_formula(rho, ks[:R + 1], R), O_formula(rho, ks[:R + 1], R)
check("P_k <= O_k <= rho^k for all k", np.all(Pk <= Ok + TOL) and np.all(Ok <= rho ** ks[:R + 1] + TOL))
check("0 <= rho^k - P_k <= rho^R", np.all(rho ** ks[:R + 1] - Pk <= rho ** R + TOL))
print(row("k", "rho^k", "parallel", "orthogonal"))
for k in [1, 2, 4, 8, 12, 16, 19]:
    print(row(k, rho ** k, Pk[k], Ok[k]))

# ---------------------------------------------------------------- (iii) mixed spectrum
print("=" * 78)
print("(iii) mixed real spectrum mu = [0.9, 0.7, 0.4, -0.6, -0.2, 0.1], non-orthogonal eigvecs")
n, p, R = 6, 10, 60
mu = np.array([0.9, 0.7, 0.4, -0.6, -0.2, 0.1])
V = np.eye(n) + 0.4 * rng.standard_normal((n, n))      # right eigvecs of J
J = V @ np.diag(mu) @ np.linalg.inv(V)
Qm = np.linalg.inv(V).T                                # columns = eigvecs of J^T
U = rng.standard_normal((n, p))
lam = rng.standard_normal(n)
for label, eps in [("generic weights", 1.0), ("slow-mode weight x0.01", 0.01)]:
    q0 = Qm[:, 0:1]
    Uu = U - (1 - eps) * (q0 @ np.linalg.pinv(q0)) @ U if eps != 1.0 else U
    # NB: U^T q_0 is rescaled by eps only if q0 q0^+ is the orthogonal projector; it is.
    C, adj = contributions(J, Uu, lam, R)
    coef = V.T @ lam                                   # lambda_R = sum_i coef_i q_i
    W = (Uu.T @ Qm) * coef                             # w_i = coef_i U^T q_i (columns)
    Cmodal = np.array([W @ (mu ** j) for j in range(R)])
    if eps == 1.0:
        check("modal identity c_j = sum_i mu_i^j w_i to 1e-10 (relative)",
              np.max(np.abs(C - Cmodal)) / np.max(np.abs(C)) < TOL)
    e, _, _ = err_curve(C)
    wn = np.linalg.norm(W, axis=0) / np.abs(1 - mu)
    print(f"  {label}: |w_i|/|1-mu_i| (normalised) = {np.round(wn / wn.sum(), 3)}")
    print(f"    adjoint rate j 5..15 = {rate(adj, 5, 15):.4f}; err rate k 2..12 = {rate(e, 2, 12):.4f}; "
          f"err rate k 30..45 = {rate(e, 30, 45):.4f}")

# ---------------------------------------------------------------- (iv) damping / over-relaxation
print("=" * 78)
print("(iv) J with mu = [0.85, 0.6, 0.3, -0.3, -0.5]; J_a = (1-a)I + aJ")
n, p, R = 5, 12, 80
mu0 = np.array([0.85, 0.6, 0.3, -0.3, -0.5])
V = np.eye(n) + 0.3 * rng.standard_normal((n, n))
J0 = V @ np.diag(mu0) @ np.linalg.inv(V)
U = rng.standard_normal((n, p))
lam = rng.standard_normal(n)
print(row("a", "max|mu'|", "max mu'>0", "adj j5-15", "err k1-10", "err k40-60", "D_1", "D_2", w=10))
for a in [1.0, 0.5, 1.3]:
    Ja = (1 - a) * np.eye(n) + a * J0
    mua = 1 - a + a * mu0
    C, adj = contributions(Ja, U, lam, R)
    e, _, _ = err_curve(C)
    D = deficit_curve(C)
    print(row(a, np.max(np.abs(mua)), np.max(mua[mua > 0]), rate(adj, 5, 15), rate(e, 1, 10),
              rate(e, 40, 60), D[1], D[2], w=10))
    if a == 1.3:
        print("    mu' =", np.round(mua, 3), " 1/|1-mu'| =", np.round(1 / np.abs(1 - mua), 3))
        print("    err_k, k=1..16:", np.round(e[1:17], 4))

# ---------------------------------------------------------------- (v) SGD deficit
print("=" * 78)
print("(v) first-order SGD deficit D_k = g.b_k/|g|^2 vs err_k, err_k^2")
rho, R = 0.8, 20
Dpar = deficit_curve(np.outer(rho ** np.arange(R), rng.standard_normal(7)))
epar = P_formula(rho, np.arange(R + 1), R)
check("parallel (same sign): D_k == err_k to 1e-10", np.max(np.abs(Dpar - epar)) < TOL)
Corth = np.diag(rho ** np.arange(R)) @ rand_orth(R)    # orthogonal rows, geometric norms
Dorth = deficit_curve(Corth)
eorth, _, _ = err_curve(Corth)
check("orthogonal: D_k == err_k^2 to 1e-10", np.max(np.abs(Dorth - eorth ** 2)) < TOL)
Calt = np.outer((-0.8) ** np.arange(R), rng.standard_normal(7))   # single negative mode
Dalt = deficit_curve(Calt)
ealt, _, _ = err_curve(Calt)
check("single mode mu=-0.8: D_k == (-1)^k err_k", np.max(np.abs(Dalt - (-1.0) ** np.arange(R + 1) * ealt)) < TOL)
print(row("k", "rho^k", "par D", "orth D", "alt err", "alt D"))
for k in [1, 2, 3, 4, 8]:
    print(row(k, rho ** k, Dpar[k], Dorth[k], ealt[k], Dalt[k]))

# random trials
viol_up = viol_lo_cond = n_lo_fail = n_lo_fail_nonneg = n_tot = 0
for trial in range(3000):
    signed = trial % 2 == 0
    Z = rng.standard_normal((R, 30))
    if not signed:
        Z = np.abs(Z)                                  # nonnegative pairwise inner products
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    C = (rng.uniform(0.5, 0.95) ** np.arange(R))[:, None] * Z
    e, tails, g = err_curve(C)
    D = tails @ g / (g @ g)
    gk_dot_b = (g - tails) @ tails.T
    gk_dot_b = np.diag(gk_dot_b)
    for k in range(1, R):
        n_tot += 1
        viol_up += D[k] > e[k] + 1e-12
        if gk_dot_b[k] >= 0:
            viol_lo_cond += D[k] < e[k] ** 2 - 1e-12
        if D[k] < e[k] ** 2 - 1e-12:
            n_lo_fail += 1
            n_lo_fail_nonneg += (not signed)
check(f"D_k <= err_k in all {n_tot} random (trial,k) cells", viol_up == 0)
check("D_k >= err_k^2 whenever g_k.b_k >= 0 (all cells)", viol_lo_cond == 0)
check("nonnegative-Gram trials never violate err^2 <= D", n_lo_fail_nonneg == 0)
print(f"  signed random-direction cells with D < err^2: {n_lo_fail}/{n_tot // 2}")

# ---------------------------------------------------------------- (vi) non-normal
print("=" * 78)
print("(vi) non-normal J = Q (diag(mu) + strictly-upper T) Q^T, spectral radius 0.7")
n, p, R = 8, 12, 120
mu = np.linspace(0.7, 0.2, n)
T = np.diag(mu) + np.triu(1.5 * rng.standard_normal((n, n)), 1)
Qn = rand_orth(n)
J = Qn @ T @ Qn.T
U = rng.standard_normal((n, p))
C, adj = contributions(J, U, rng.standard_normal(n), R)
e, _, _ = err_curve(C)
print(f"  ||J||_2 = {np.linalg.norm(J, 2):.3f}, spectral radius = {np.max(np.abs(np.linalg.eigvals(J))):.3f}")
print(row("j/k", "|lam_j|/|lam_0|", "err_k", "loc. ratio adj", w=14))
for j in [0, 1, 2, 4, 6, 10, 20, 40, 80, 110]:
    lr = adj[j + 1] / adj[j]
    print(row(j, adj[j] / adj[0], e[j], lr, w=14))
check("asymptotic adjoint ratio -> spectral radius (|ratio-0.7| < 0.01 at j=110)",
      abs(adj[111] / adj[110] - 0.7) < 0.01)

# ---------------------------------------------------------------- (vii) preconditioned
print("=" * 78)
print("(vii) preconditioned deficits (diagonal D, sign descent, Adam-like one-step)")
R, p = 20, 40
viol = {"|D_D| <= sqrt(kappa) err": 0, "Wielandt (orth)": 0, "0 <= D_sign <= 2|b|_1/|g|_1": 0,
        "parallel Adam-like D in [0, err]": 0}
for trial in range(2000):
    d = np.exp(rng.uniform(-2, 2, p))
    kap = d.max() / d.min()
    C = np.diag(rng.uniform(0.6, 0.9) ** np.arange(R)) @ rand_orth(p)[:R]   # orthogonal contributions
    e, tails, g = err_curve(C)
    k = rng.integers(1, R)
    b, gk = tails[k], g - tails[k]
    nD = lambda x: np.sqrt(x @ (d * x))
    DD = (g @ (d * b)) / (g @ (d * g))
    viol["|D_D| <= sqrt(kappa) err"] += abs(DD) > np.sqrt(kap) * e[k] + 1e-12
    eD = nD(b) / nD(g)
    viol["Wielandt (orth)"] += abs(DD - eD ** 2) > (kap - 1) / (kap + 1) * nD(gk) * nD(b) / nD(g) ** 2 + 1e-12
    Ds = 1 - (g @ np.sign(gk)) / np.abs(g).sum()
    viol["0 <= D_sign <= 2|b|_1/|g|_1"] += (Ds < -1e-12) or (Ds > 2 * np.abs(b).sum() / np.abs(g).sum() + 1e-12)
    # parallel contributions + Adam-like v = x^2 + s^2
    gp = rng.standard_normal(p)
    ep = rng.uniform(0, 0.9)
    s = 10 ** rng.uniform(-3, 1)
    uk = (1 - ep) * gp / np.sqrt(((1 - ep) * gp) ** 2 + s ** 2)
    uR = gp / np.sqrt(gp ** 2 + s ** 2)
    Da = 1 - (gp @ uk) / (gp @ uR)
    viol["parallel Adam-like D in [0, err]"] += (Da < -1e-12) or (Da > ep + 1e-12)
for kname, v in viol.items():
    check(f"{kname}: violations {v}/2000", v == 0)
gp = rng.standard_normal(p)
print(row("s (v=x^2+s^2)", "err", "SGD D", "Adam-like D", w=14))
for s in [1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
    ep = 0.5
    uk = (1 - ep) * gp / np.sqrt(((1 - ep) * gp) ** 2 + s ** 2)
    uR = gp / np.sqrt(gp ** 2 + s ** 2)
    print(row(s, ep, ep, 1 - (gp @ uk) / (gp @ uR), w=14))

print("=" * 78)
print(f"ALL CHECKS {'PASSED' if ok else 'FAILED'}   ({time.time() - T0:.2f} s)")
assert ok
