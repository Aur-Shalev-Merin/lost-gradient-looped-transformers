# Law 1 for truncated backprop through loops: a formal derivation

*Status: mathematics plus a numpy toy check (`law1_toy_check.py`). No experimental results are used or claimed here. Written 2026-10-04.*

## 0. Setting and notation

A looped (weight-tied) block runs for $R$ loops, $h_{t+1}=F(h_t,e;\theta)$ for $t=0,\dots,R-1$, and the loss is $L(h_R)$. Write $J_t=\partial F/\partial h|_t$ and $M_t=\partial F/\partial\theta|_t$. The adjoint is $\lambda_R=\partial L/\partial h_R$ and $\lambda_t=J_t^\top\lambda_{t+1}$. The full gradient is $g=\sum_{t=0}^{R-1}M_t^\top\lambda_{t+1}$.

Contributions are indexed backward from the output: $c_j:=M_{R-1-j}^\top\lambda_{R-j}$ for $j=0,\dots,R-1$. Then $g=\sum_j c_j$, the $k$-loop truncated gradient is $g_k=\sum_{j<k}c_j$, the dropped piece is $b_k=g-g_k=\sum_{j\ge k}c_j$, and

$$\mathrm{err}_k=\|b_k\|/\|g\|,\qquad D_k:=\frac{g\cdot b_k}{\|g\|^2}\ \ \text{(first-order SGD deficit, §3)}.$$

The Gram matrix is $G_{ij}=c_i\cdot c_j$. With $\mathbf 1_{\ge k}$ the indicator of $\{j\ge k\}$, we have $\mathrm{err}_k^2=\mathbf 1_{\ge k}^\top G\,\mathbf 1_{\ge k}\,/\,\mathbf 1^\top G\,\mathbf 1$.

**Assumptions.**
- **(A1) Settled loop.** $J_t=J^*$ and $M_t=M$ for every $t$, as at a fixed point $h^*=F(h^*)$. Then $c_j=M^\top (J^{*\top})^j\lambda_R$.
- **(A2) One dominant direction.** $\lambda_R=v$ with $J^{*\top}v=\rho v$ and $\rho\in(0,1)$ real. §2 relaxes this to a mixture of modes.

Throughout, $P_k:=\dfrac{\rho^k-\rho^R}{1-\rho^R}$ and $O_k:=\sqrt{\dfrac{\rho^{2k}-\rho^{2R}}{1-\rho^{2R}}}$.

---

## 1. Law 1: the parallel and orthogonal extremes, and arbitrary angles

**Proposition 1 (parallel).** Under (A1)–(A2), $c_j=\rho^j c_0$ with $c_0=M^\top v$, and if $c_0\neq0$ then $\mathrm{err}_k=P_k$ exactly. Moreover $g_k=(1-P_k)\,g$.

*Proof.* $(J^{*\top})^j v=\rho^j v$, so $c_j=\rho^j c_0$. Summing the geometric series gives $b_k=c_0(\rho^k-\rho^R)/(1-\rho)$ and $g=c_0(1-\rho^R)/(1-\rho)$. The ratio is $P_k$ and $b_k=P_k\,g$. $\square$

*In words: one settled, dominant adjoint direction gives exactly geometric error, and truncation only rescales the gradient.*

**Proposition 2 (orthogonal).** Suppose $c_i\cdot c_j=0$ for $i\neq j$ and $\|c_j\|=\rho^j a$. Then $\mathrm{err}_k=O_k$ exactly.

*Proof.* By Pythagoras, $\|b_k\|^2=a^2\sum_{j\ge k}\rho^{2j}$ and $\|g\|^2=a^2\sum_{j<R}\rho^{2j}$. Take the ratio. $\square$

This case does not follow from (A1)–(A2), which force parallel contributions. It is the opposite extreme. A concrete instance has $J^{*\top}=\rho S$ with $S$ the cyclic shift, $M=I$ and $\lambda_R=e_1$, so $c_j=\rho^j e_{1+j}$. Here every eigenvalue of $J^*$ has modulus $\rho$, and the phases make the contributions orthogonal. The toy check (ii) uses this construction.

**Corollary 1 (both are $\approx\rho^k$).** For $0\le k\le R$:
$$\rho^k-\frac{\rho^R(1-\rho^k)}{1-\rho^R}\;=\;P_k\;\le\;O_k\;\le\;\rho^k .$$
So $0\le O_k-P_k\le\rho^k-P_k\le\rho^R$. As $R\to\infty$, both equal $\rho^k$ exactly. The relative gap is $P_k/\rho^k=(1-\rho^{R-k})/(1-\rho^R)$, so it matters only when $k$ approaches $R$.

*Proof.* We have $O_k^2/P_k^2=\frac{(1+\rho^{R-k})(1-\rho^R)}{(1+\rho^R)(1-\rho^{R-k})}\ge1$, because $x\mapsto(1+x)/(1-x)$ is increasing and $\rho^{R-k}\ge\rho^R$. Also $O_k\le\rho^k$, since $1-\rho^{2(R-k)}\le1-\rho^{2R}$. The left identity is algebra. $\square$

**Proposition 3 (arbitrary angles).** Let $s_j=\|c_j\|$, $S_k=\sum_{j\ge k}s_j$ and $Q_k=\sum_{j\ge k}s_j^2$.

(a) *No sign condition.* $\mathrm{err}_k\ge(s_k-S_{k+1})_+/S_0$, and there is no finite upper bound. If $g=0$ while $b_k\ne0$, then $\mathrm{err}_k=\infty$. With geometric $s_j=a\rho^j$, the lower bound is nontrivial only when $\rho<1/2$.

(b) *Nonnegative Gram ($c_i\cdot c_j\ge0$ for all $i,j$).*
$$\frac{\sqrt{Q_k}}{S_0}\;\le\;\mathrm{err}_k\;\le\;\frac{S_k}{\sqrt{Q_0}} .$$
For geometric $s_j$ and $R\to\infty$ this becomes $\rho^k\sqrt{\tfrac{1-\rho}{1+\rho}}\le\mathrm{err}_k\le\rho^k\sqrt{\tfrac{1+\rho}{1-\rho}}$. The decay rate is therefore exactly $\rho$, and the prefactor lies within $e^{\pm\,\mathrm{artanh}\,\rho}$.

*Proof.* (a) Use $\|b_k\|\ge s_k-\|\sum_{j>k}c_j\|$ and $\|g\|\le S_0$. A pair $c_0=-c_1\neq0$ with $R=2$ gives $g=0$. (b) Nonnegative off-diagonal terms give $\sum_{j\in A}s_j^2\le\|\sum_{j\in A}c_j\|^2\le(\sum_{j\in A}s_j)^2$ for any index set $A$. Apply this to the tail and to all indices. $\square$

*Tightness (corrected 2026-10-14, v15 fact-check; the earlier text said the upper bound is attained).* Neither bound is attained at finite $k$. With the tail mutually parallel and the head orthogonal to everything, $\mathrm{err}_k=S_k/\sqrt{Q_0-Q_k+S_k^2}<S_k/\sqrt{Q_0}$ unless the tail has a single term. Both bounds are approached as $k$ grows. At $\rho=0.85$, $R\gg k$: the upper bound exceeds 1 for $k\le7$; at $k=4$ the achievable range is $0.24$–$0.91$ against bounds $0.15$–$1.83$ (explicit constructions plus a random search); at $k=16$ the extremes reach $0.97$ and $1.06$ times the bounds.

*In words: Law 1 holds, with a bounded prefactor, whenever the contributions never oppose one another. It can fail badly only through cancellation. One case is $g$ nearly cancelling while the tail survives (err large). The other is the tail self-cancelling (err small).*

A single mode with a negative eigenvalue does **not** break Law 1. With $c_j=(-\rho)^jc_0$, $\mathrm{err}_k=\rho^k\,|1-(-\rho)^{R-k}|/|1-(-\rho)^R|\approx\rho^k$; the toy check (v) confirms this. Cancellation comes from mixing several modes, or from contributions whose directions vary with $t$ (A1 failing).

---

## 2. Mixtures of modes, damping, and non-normal Jacobians

**Proposition 4 (mode mixture).** Let $J^*$ be diagonalizable, with $J^{*\top}q_i=\mu_i q_i$, $\lambda_R=\sum_i\beta_iq_i$ and $w_i:=\beta_iM^\top q_i$. Then $c_j=\sum_i\mu_i^jw_i$. With $\hat w_i:=w_i/(1-\mu_i)$:
$$b_k=\sum_i\hat w_i\,\mu_i^k(1-\mu_i^{R-k}),\qquad g=\sum_i\hat w_i(1-\mu_i^R).$$
As $R\to\infty$ with $|\mu_i|<1$:

(i) $\mathrm{err}_k=\|\sum_i\hat w_i\mu_i^k\|/\|\sum_i\hat w_i\|\le\sum_i\|\hat w_i\||\mu_i|^k/\|g\|$. If the $\hat w_i$ are mutually orthogonal, $\mathrm{err}_k^2=\sum_i\|\hat w_i\|^2|\mu_i|^{2k}/\sum_i\|\hat w_i\|^2$ exactly.

(ii) $D_k=\sum_i\alpha_i\mu_i^k$ with $\alpha_i=\hat w_i\cdot g/\|g\|^2$ and $\sum_i\alpha_i=1$. The deficit is a signed linear mixture of the powers $\mu_i^k$.

(iii) Complex conjugate pairs combine into real terms $\propto|\mu|^k\cos(k\varphi+\phi)$. Asymptotically $\mathrm{err}_k\sim|\mu_\star|^k$, where $\mu_\star$ is the slowest mode with $w_\star\neq0$.

*Proof.* Expand $\lambda_R$ in the eigenvectors and sum each geometric series. $\square$

**Remark 1 (adjoint rate vs error rate).** The adjoint norm $\|\lambda_{R-j}\|=\|\sum_i\beta_i\mu_i^jq_i\|$ weights mode $i$ by $|\beta_i|\|q_i\|$. Its rate is $\max\{|\mu_i|:\beta_i\ne0\}$. The gradient error weights mode $i$ by $\|\hat w_i\|=|\beta_i|\|M^\top q_i\|/|1-\mu_i|$. The two rates differ when the slowest adjoint mode is (a) nearly annihilated by $M^\top$, or (b) has $\mu_i<0$, which gives the factor $1/|1-\mu_i|\le1$. A positive mode near 1 instead has weight up to $1/(1-\mu_i)\gg1$. If a slow mode $s$ competes with a faster but heavier mode $f$, the error follows $|\mu_f|$ until the crossover
$$k_\times=\ln(\|\hat w_f\|/\|\hat w_s\|)\,/\,\ln(|\mu_s|/|\mu_f|),$$
and follows $|\mu_s|$ after it. The toy check (iii) suppresses the 0.9 mode's weight by 100×. The adjoint rate stays 0.876, the err rate over $k\in[2,12]$ is 0.709, and the err rate over $k\in[30,45]$ is 0.889.

**Proposition 5 (damping and over-relaxation only change the rates).** Damping $x\leftarrow(1-a)x+aF(x)$ has the same fixed point. Its state Jacobian is $(1-a)I+aJ^*$, with eigenvalues $\mu_i'=1-a+a\mu_i$, and its parameter Jacobian is $aM$. So $w_i'=aw_i$ and $1-\mu_i'=a(1-\mu_i)$. **The weights $\hat w_i$ do not depend on $a$**, and so neither does the $R\to\infty$ gradient $g=\sum_i\hat w_i$, which is the implicit gradient. Only the rates move:
$$\mathrm{err}_k(a)=\Big\|\sum_i\hat w_i(1-a+a\mu_i)^k\Big\|\Big/\Big\|\sum_i\hat w_i\Big\|.$$
*Proof.* Substitute into Proposition 4. $\square$

Consequences:
- **Damping ($a<1$)** moves every $\mu_i$ toward 1. All modes slow down and $\rho$ rises.
- **Over-relaxation ($a>1$)** gives $\mu_i'-\mu_i=(1-a)(1-\mu_i)<0$, so every mode moves left. Positive modes get faster. Negative modes are pushed toward $-1$, so $|\mu_i'|$ grows; stability needs $a<2/(1-\mu_{\min})$. The adjoint norm then decays at $\max|\mu'|$, which can be set by a mode that was negative to begin with. That mode keeps its small weight $\|\hat w_i\|=\|w_i\|/(1+|\mu_i|)$, and its terms alternate in sign: in $D_k$ (Prop. 4(ii)) they enter as $\alpha_i(-|\mu_i'|)^k$ and partly cancel.
- So over the range $k<k_\times$, the gradient error decays at roughly the largest positive $\mu_i'$, which is faster than the adjoint norm. **Asymptotically the error rate still equals $\max|\mu'|$** (with nonzero weight).

The toy check (iv) uses $\mu=[.85,.6,.3,-.3,-.5]$ and $a=1.3$, giving $\mu'=[.805,.48,.09,-.69,-.95]$. The adjoint rate over $j\in[5,15]$ is 0.920. The err rate is 0.787 over $k\in[1,10]$ and 0.936 over $k\in[40,60]$. The err sequence zig-zags.

**Remark 2 (non-normal $J^*$).** If $\|J^*\|_2>1>r(J^*)$, then $\|(J^{*\top})^j\|$ can grow transiently. By Gelfand's formula, $\|J^{*j}\|^{1/j}\to r(J^*)$. For every $\epsilon>0$ there is a $K_\epsilon$ with $\|J^{*j}\|\le K_\epsilon(r+\epsilon)^j$, so $\mathrm{err}_k\le K_\epsilon\|M\|\|\lambda_R\|(r+\epsilon)^k/((1-r-\epsilon)\|g\|)$. The asymptotic rate is the spectral radius, but the prefactor $K_\epsilon$ can be huge, and $\mathrm{err}_k>1$ is possible. In the toy check (vi), $\|J\|_2=5.8$ and $r=0.7$: the adjoint grows 43× before it decays, $\mathrm{err}_k$ reaches 1.26 at $k\approx10$, and the local adjoint ratio reaches 0.700 by $j\approx80$.

---

## 3. From gradient error to first-order training progress (SGD)

For one step $\theta\leftarrow\theta-\eta g_k$, the first-order loss change is $-\eta\,g\cdot g_k$. The full-gradient step gives $-\eta\|g\|^2$. The progress ratio is
$$\frac{g\cdot g_k}{\|g\|^2}=1-D_k,\qquad D_k=\mathrm{err}_k\cos(g,b_k)=\mathrm{err}_k^2+\frac{g_k\cdot b_k}{\|g\|^2}.$$

**Proposition 6 (the bracket).**
(a) $|D_k|\le\mathrm{err}_k$ always (Cauchy–Schwarz).
(b) $D_k\ge\mathrm{err}_k^2\iff g_k\cdot b_k\ge0$. This is the exact condition.
(c) A nonnegative Gram matrix gives $g_k\cdot b_k=\sum_{i<k\le j}c_i\cdot c_j\ge0$, and hence $\mathrm{err}_k^2\le D_k\le\mathrm{err}_k$.
(d) Parallel contributions with the same sign (Prop. 1) give $D_k=\mathrm{err}_k$. Orthogonal contributions (Prop. 2) give $g_k\cdot b_k=0$, so $g\cdot b_k=\|b_k\|^2$ and $D_k=\mathrm{err}_k^2$.
(e) *Counterexamples.* A single mode $\mu=-\rho$ gives $D_k=(-1)^k\mathrm{err}_k$, so $D_k=-\mathrm{err}_k<\mathrm{err}_k^2$ for odd $k$. For random signed directions with geometric norms, about half of the $(\text{trial},k)$ cells have $D_k<\mathrm{err}_k^2$: 14 100 of 28 500 in the toy check (v).

*Proof.* (a)–(d) follow from the identity above. For (e), $b_k=(-\rho)^k\frac{1-(-\rho)^{R-k}}{1-(-\rho)^R}\,g$, and the scalar factor has sign $(-1)^k$. $\square$

*In words: when contributions never oppose one another, the first-order cost of truncation lies between $\mathrm{err}^2$ (orthogonal) and $\mathrm{err}$ (parallel). Under Law 1 it therefore decays at a rate between $\rho^{2k}$ and $\rho^k$. Opposing contributions can make the deficit negative.*

A negative $D_k$ means the truncated step has a larger first-order decrease *per unit step size*. It does not mean the truncated direction is better. A comparison normalized by step length would use $\cos(g,g_k)=(1-D_k)\,\|g\|/\|g_k\|$.

---

## 4. Preconditioned (Adam / sign) versions: partial results

Define $\mathrm{deficit}_{\rm pre}=1-\dfrac{g\cdot u_k}{g\cdot u_R}$, where $u$ is the preconditioned step built from $g_k$ or from $g$.

**Proposition 7 (fixed diagonal $D\succ0$, condition number $\kappa$).** Use the same $D$ for both runs, write $\|x\|_D^2=x^\top Dx$, and let $\mathrm{err}^D_k=\|b_k\|_D/\|g\|_D$. Then:
(a) $\mathrm{deficit}_D=g^\top Db_k/g^\top Dg$ and $|\mathrm{deficit}_D|\le\mathrm{err}^D_k\le\sqrt\kappa\,\mathrm{err}_k$.
(b) $(\mathrm{err}_k^D)^2\le\mathrm{deficit}_D\le\mathrm{err}^D_k\iff g_k^\top Db_k\ge0$.
(c) Parallel contributions give $\mathrm{deficit}_D=\mathrm{err}_k$ for every $D$.
(d) Euclidean-orthogonal contributions are not $D$-orthogonal. By Wielandt's inequality ($x\perp y\Rightarrow|x^\top Dy|\le\frac{\kappa-1}{\kappa+1}\|x\|_D\|y\|_D$; see Horn & Johnson, *Matrix Analysis*):
$$\Big|\mathrm{deficit}_D-(\mathrm{err}^D_k)^2\Big|\le\tfrac{\kappa-1}{\kappa+1}\,\|g_k\|_D\|b_k\|_D/\|g\|_D^2 .$$

*Proof.* (a)–(c) are Prop. 6 in the $D$ inner product, plus $\lambda_{\min}\|x\|^2\le\|x\|_D^2\le\lambda_{\max}\|x\|^2$. For (d), apply Wielandt to $x=g_k$ and $y=b_k$. $\square$

**Proposition 8 (sign descent, $u=\mathrm{sign}(\cdot)$; the $\beta\to0$, $\epsilon\to0$ limit of Adam).** Let $\mathrm{deficit}_{\rm sign}=1-g\cdot\mathrm{sign}(g_k)/\|g\|_1$. Then
$$0\le\mathrm{deficit}_{\rm sign}\le\frac{2}{\|g\|_1}\sum_{i:\,|b_{k,i}|\ge|g_i|}|g_i|\le2\frac{\|b_k\|_1}{\|g\|_1}.$$
With parallel contributions and $\mathrm{err}_k<1$, $\mathrm{deficit}_{\rm sign}=0$.

*Proof.* Coordinate $i$ costs $|g_i|-g_i\,\mathrm{sign}(g_{k,i})\in\{0,|g_i|,2|g_i|\}$. The cost is nonzero only if $\mathrm{sign}(g_i-b_{k,i})\ne\mathrm{sign}(g_i)$, which requires $|b_{k,i}|\ge|g_i|$. With parallel contributions, $g_k=(1-\mathrm{err}_k)g$ has the same signs as $g$. $\square$

**Proposition 9 (one-step Adam-like, $v=x^2+s^2$ per coordinate, each run's $v$ built from its own gradient).** With parallel contributions and $e=\mathrm{err}_k<1$, $\mathrm{deficit}\in[0,e]$. It tends to $0$ as $s\to0$ (the sign regime) and to $e$ as $s\to\infty$ (the SGD regime).

*Proof.* $g\cdot u_k=\sum_ir_i\,g_i^2/\sqrt{g_i^2+s^2}$ with $r_i=(1-e)\sqrt{g_i^2+s^2}/\sqrt{(1-e)^2g_i^2+s^2}\in[1-e,1]$. So $g\cdot u_k/g\cdot u_R$ is a weighted mean of the $r_i$. $\square$

The toy check (vii) shows the transition at $e=0.5$: the deficit is $2\times10^{-7}$ at $s=10^{-4}$, $0.32$ at $s=1$, and $0.50$ at $s=100$.

**Not proved / conjectures.**
- **(C1)** For real Adam, where $v$ is an EMA over noisy steps and is built from $g_k$ during truncated training, $\mathrm{deficit}_{\rm adam}$ lies between the sign-descent value and the fixed-$D$ value, approaching the fixed-$D$ value when gradient noise dominates $v$.
- **(C2)** Under orthogonal contributions and sign descent there is no bracket in terms of $\mathrm{err}_k$ alone; only the $\ell_1$ bound of Prop. 8 holds. Note $\|b\|_1/\|g\|_1\le\sqrt p\,\mathrm{err}_k$, which is weak.

---

## 5. Limits

1. **First order, one step.** Everything above compares $g\cdot u_k$ with $g\cdot u_R$ at a fixed $\theta$. It ignores curvature ($\tfrac{\eta^2}{2}u^\top Hu$), learning-rate retuning, and the trajectory. (A1) is a fixed-point idealization. Early loops that have not settled violate it, and then only Prop. 3 (arbitrary angles) applies.

2. **Biased stationary points.** Truncated training stops where $g_k=0$, not where $g=0$. *Fact:* if $\mathrm{err}_k(\theta)\le\varepsilon<1$, then $\|g\|\le\|g_k\|/(1-\varepsilon)$, so $g_k=0\Rightarrow g=0$. A purely relative bias does not move stationary points; in the exact parallel case (Prop. 1), $g_k\propto g$. At a biased stationary point, $b_k=g\ne0$ and $\mathrm{err}_k=\infty$, so **Law 1 cannot hold uniformly near such a point**. What matters there is an absolute bias floor.

   The standard result is Ajalloeian & Stich 2020 (arXiv 2008.00051, Assumption 4: $\|b(x)\|^2\le m\|\nabla f(x)\|^2+\zeta^2$, $0\le m<1$). For $L$-smooth $f$, SGD reaches $\min_t\mathbb E\|\nabla f\|^2=O(\epsilon+\zeta^2/(1-m))$, and the floor $\zeta^2/(1-m)$ is tight (their Remark 5). Under PL with constant $\mu$, $F_T\le(1-\gamma\mu(1-m))^TF_0+\tfrac12\big(\tfrac{\zeta^2}{\mu(1-m)}+\tfrac{L\gamma\sigma^2}{\mu(1-m)}\big)$ (their Thm 6). Law 1 enters through $m=\sup\mathrm{err}_k^2$, valid only where that supremum is below 1. Near stationarity, the dropped tail goes into $\zeta$.

---

## 6. Attribution

- **Aicher, Foti & Fox 2019** (arXiv 1905.07473). *Verified from the PDF, main text §3; the supplement was not read.*
  - Assumption (A-1): $\mathbb E[\phi_{k+1}]\le\beta\,\mathbb E[\phi_k]$ for $k\ge\tau$, where $\phi_k$ is the norm of the backpropagated gradient at lag $k$. This is geometric decay in expectation, weaker than contractivity.
  - Assumption (A-2): $\|\partial H/\partial\theta\|\le M$.
  - Thm 1: the absolute TBPTT bias is at most $\mathcal E(K)$, which decays geometrically in $K$, roughly $M\,\mathbb E[\phi_\tau]\beta^{K-\tau}/(1-\beta)$ (exact form garbled in my text extraction). The relative bias is at most $\mathcal E(K)/(\max_{k\le K}\|\mathbb E\hat g^k\|-\mathcal E(K))$.
  - Thm 2: SGD with relative bias $\delta<1$ converges to stationary points, paying a factor $(1-\delta)^{-1}$.
  - Adaptive $K$: the smallest $K$ with bound $<\delta$.
  - Their bounds use norms and the triangle inequality. **I found no analysis of the bias direction** (parallel/orthogonal, $\cos(g,b)$) in the main text.
- **JFB** (Fung et al. 2022, arXiv 2103.12803) and **phantom gradients** (Geng et al. 2021, arXiv 2111.05177) give descent-direction conditions for truncated or Neumann-series implicit gradients. As $R\to\infty$ with $a=1$, $g_k$ is the $k$-term Neumann truncation of the implicit gradient. *Not re-verified for this note:* the precise conditions in those papers, and whether either discusses the $\mathrm{err}$-to-deficit bracket.

| Statement | Status |
|---|---|
| Geometric decay of truncation bias under (expected) adjoint decay; relative-bias SGD convergence | **Cited**: Aicher 2019 Thm 1–2 |
| Biased-SGD floor $\zeta^2/(1-m)$, PL rate | **Cited**: Ajalloeian & Stich 2020 |
| Truncated Neumann / descent direction for implicit models | **Cited**: JFB, phantom gradients (details unverified) |
| Prop 1–2 closed forms; Cor 1 ordering $P_k\le O_k\le\rho^k$, gap $\le\rho^R$ | **Proved** (elementary; likely folklore) |
| Prop 3 Gram bounds; rate exactly $\rho$ under nonnegative Gram | **Proved** (standard linear algebra) |
| Prop 4 mode mixture; Remark 1 crossover $k_\times$ | **Proved** (standard); framing possibly new |
| Prop 5 damping leaves $\hat w_i$ and $g$ invariant; over-relaxation explanation | **Proved**; *possibly new* as an explanation |
| Remark 2 non-normal transient, rate $=r(J)$ | **Standard** (Gelfand) |
| Prop 6 bracket $\mathrm{err}^2\le D\le\mathrm{err}\iff g_k\cdot b_k\ge0$ | **Proved**; *possibly new* in this context |
| Prop 7–9 preconditioned / sign / Adam-like | **Proved** (partial, idealized) |
| C1–C2 real-Adam behavior | **Conjecture** |
| Relative bias cannot create biased stationary points | **Proved** (one line; standard) |

---

## Appendix: toy check output

`conda run -n astrocyte python law1_toy_check.py` (numpy, CPU, seed 0, 0.84 s). All assertions pass. Closed-form cases match to $10^{-10}$.

```
==============================================================================
(i) symmetric J, dominant eigenvalue rho = 0.8, others |mu| <= 0.3
  [PASS] exact eigvec: err_k == (rho^k - rho^R)/(1 - rho^R) to 1e-10
         k      rho^k  exact-eig    generic
         1        0.8     0.7998     0.8202
         2       0.64     0.6396     0.6523
         4     0.4096     0.4089     0.4162
         8     0.1678     0.1667     0.1697
        12    0.06872    0.06757    0.06877
        16    0.02815    0.02694    0.02742
  generic lambda_R: adjoint rate (j 8..20) = 0.8000, err rate (k 4..16) = 0.7972
==============================================================================
(ii) J^T = rho * cyclic shift, U = I, lambda_R = e_1  -> mutually orthogonal c_j
  [PASS] Gram matrix of contributions is diagonal
  [PASS] err_k == sqrt((rho^2k - rho^2R)/(1 - rho^2R)) to 1e-10
  [PASS] P_k <= O_k <= rho^k for all k
  [PASS] 0 <= rho^k - P_k <= rho^R
         k      rho^k   parallel orthogonal
         1        0.8     0.7977        0.8
         2       0.64     0.6358     0.6399
         4     0.4096     0.4027     0.4095
         8     0.1678     0.1581     0.1674
        12    0.06872    0.05786    0.06775
        16    0.02815    0.01681    0.02568
        19    0.01441   0.002916   0.008647
==============================================================================
(iii) mixed real spectrum mu = [0.9, 0.7, 0.4, -0.6, -0.2, 0.1], non-orthogonal eigvecs
  [PASS] modal identity c_j = sum_i mu_i^j w_i to 1e-10 (relative)
  generic weights: |w_i|/|1-mu_i| (normalised) = [0.526 0.363 0.018 0.074 0.003 0.015]
    adjoint rate j 5..15 = 0.8758; err rate k 2..12 = 0.8955; err rate k 30..45 = 0.8888
  slow-mode weight x0.01: |w_i|/|1-mu_i| (normalised) = [0.011 0.758 0.037 0.156 0.006 0.032]
    adjoint rate j 5..15 = 0.8758; err rate k 2..12 = 0.7090; err rate k 30..45 = 0.8889
==============================================================================
(iv) J with mu = [0.85, 0.6, 0.3, -0.3, -0.5]; J_a = (1-a)I + aJ
         a   max|mu'|  max mu'>0  adj j5-15  err k1-10 err k40-60        D_1        D_2
         1       0.85       0.85     0.8489     0.8559     0.8484     0.7772     0.5755
       0.5      0.925      0.925     0.9133     0.9251     0.9163     0.8882     0.7819
       1.3       0.95      0.805     0.9203     0.7874     0.9362     0.7108      0.457
    mu' = [ 0.805  0.48   0.09  -0.69  -0.95 ]  1/|1-mu'| = [5.128 1.923 1.099 0.592 0.513]
    err_k, k=1..16: [0.8053 0.5759 0.5607 0.3816 0.3904 0.2451 0.2691 0.153  0.1868 0.0937
 0.1321 0.0571 0.0958 0.036  0.0716 0.0254]
==============================================================================
(v) first-order SGD deficit D_k = g.b_k/|g|^2 vs err_k, err_k^2
  [PASS] parallel (same sign): D_k == err_k to 1e-10
  [PASS] orthogonal: D_k == err_k^2 to 1e-10
  [PASS] single mode mu=-0.8: D_k == (-1)^k err_k
         k      rho^k      par D     orth D    alt err      alt D
         1        0.8     0.7977       0.64      0.821     -0.821
         2       0.64     0.6358     0.4095     0.6358     0.6358
         3      0.512     0.5063      0.262     0.5296    -0.5296
         4     0.4096     0.4027     0.1677     0.4027     0.4027
         8     0.1678     0.1581    0.02802     0.1581     0.1581
  [PASS] D_k <= err_k in all 57000 random (trial,k) cells
  [PASS] D_k >= err_k^2 whenever g_k.b_k >= 0 (all cells)
  [PASS] nonnegative-Gram trials never violate err^2 <= D
  signed random-direction cells with D < err^2: 14100/28500
==============================================================================
(vi) non-normal J = Q (diag(mu) + strictly-upper T) Q^T, spectral radius 0.7
  ||J||_2 = 5.822, spectral radius = 0.700
           j/k |lam_j|/|lam_0|          err_k loc. ratio adj
             0              1              1          2.523
             1          2.523          1.002          3.483
             2          8.788          1.007          2.395
             4          38.27          1.051          1.212
             6          38.76          1.246         0.5841
            10             43           1.26          1.125
            20          10.64         0.1382         0.7561
            40        0.01544      0.0001761         0.7049
            80      1.051e-08      1.184e-10         0.7001
           110      2.371e-13      2.596e-15            0.7
  [PASS] asymptotic adjoint ratio -> spectral radius (|ratio-0.7| < 0.01 at j=110)
==============================================================================
(vii) preconditioned deficits (diagonal D, sign descent, Adam-like one-step)
  [PASS] |D_D| <= sqrt(kappa) err: violations 0/2000
  [PASS] Wielandt (orth): violations 0/2000
  [PASS] 0 <= D_sign <= 2|b|_1/|g|_1: violations 0/2000
  [PASS] parallel Adam-like D in [0, err]: violations 0/2000
 s (v=x^2+s^2)            err          SGD D    Adam-like D
        0.0001            0.5            0.5        2.1e-07
          0.01            0.5            0.5      0.0008588
           0.1            0.5            0.5        0.02694
             1            0.5            0.5         0.3214
            10            0.5            0.5          0.496
           100            0.5            0.5            0.5
==============================================================================
ALL CHECKS PASSED   (0.84 s)
```
