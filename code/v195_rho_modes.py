"""v195: why do the two offsetting effects behind rho^k match? (prereg docs/prereg_v195_rho_modes.md)
Frozen models, no training. Per sequence:
  traj     forward states x_0 .. x_R (no grad); step[t] = |x_{t+1} - x_t| / |x_t|
  lam[t]   backward signal dL/d(output of loop t), t = 0 .. R-1 (one full backward, parameters frozen)
  c_t      = P_t^T lam[t]  loop t's contribution to the weight gradient (parameter VJP through loop t alone)
  p_t      = P_t^T u       the same per-loop map applied to ONE fixed vector, u = lam[R-1] / |lam[R-1]|
  kap_i    = (J^T)^i lam[R-1], i = 0 .. K   Krylov sequence under the LAST loop's backward map J^T (constant-map model)
  chat_i   = P^T kap_i, i = 0 .. K-1         constant-map contributions (P^T = the last loop's parameter map)
  Arnoldi  m steps of J^T from q_0 = lam[R-1] / |lam[R-1]| (two Gram-Schmidt passes, float64): H and P^T q_l
  g_0      gradient with no loop gradient (the readout part), for all-parameter lost gradients (zero for a loop-only
           subset)
Saved: exact float64 Gram of the state vectors [lam_0 .. lam_{R-1}, kap_0 .. kap_K]; one CountSketch Gram (2^20 buckets,
hashes fixed per model) of the parameter vectors [g_0, c_0 .. c_{R-1}, p_0 .. p_{R-1}, chat_0 .. chat_{K-1},
P^T q_0 .. P^T q_{m-1}] with their exact norms; H; checks. Analysis: v195_analyze.py (frozen with the prereg).
Families:
  parcae  Parcae via the frozen v162 / v163 machinery (as v191): 16 GSM8K-val chunks, fp32, all parameters, a = 1,
          seed 777 + i before the state init, loop = core_block_forward(x, e, freqs, None, t, R); P_t includes the
          prelude through the injected e (as v191's c_j).
  probe   loop_grad_probe's adapter (as v192): Huginn-3.5B in bf16, adapter + core_block[0], probe sequences.
  python v195_rho_modes.py parcae --repo SandyResearch/parcae-140m --R 32 --K 64 --tag 140m
  LOOP_MODEL=huginn python v195_rho_modes.py probe --R 32 --K 32 --seqs 0-7,16-19 --tag huginn_part0
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

SKETCH_D, CHUNK = 2 ** 20, 2 ** 23
KEEP = os.environ.get('V195_KEEP') == '1'      # TINY test only: also keep every parameter vector for an exact Gram


# ------------------------------------------------------------------------------------------------- model families
class ParcaeFam:
    def __init__(self, repo):
        import v162_parcae_trunc_train as v
        import v163_parcae_damped_trunc as d
        self.v, self.d = v, d
        v.REPO = repo
        self.tok, self.m = v.load_model()
        self.m.eval()
        self.params = list(self.m.parameters())
        for q in self.params:
            q.requires_grad_(False)
        self.dtype, self.dev, self.out_dir = torch.float32, v.DEV, v.OUT
        data = v.gsm8k_chunks(self.tok)
        self._seqs = [(i, 'gsm8k', data['val'][i:i + 1]) for i in range(min(v.N_MEASURE, len(data['val'])))]
        self.name = repo

    def seqs(self):
        return self._seqs

    def ve(self, i):
        return self.m.value_embeds[str(i)](self.ids) if str(i) in self.m.value_embeds else None

    def prelude(self):
        m = self.m; m._current_input_ids = self.ids
        e = m.transformer.wte(self.ids)
        if m.emb_scale != 1:
            e = e * m.emb_scale
        for i, block in enumerate(m.transformer.prelude):
            e = block(e, self.freqs, None, ve=self.ve(i))
        if m.config.prelude_norm:
            e = m.transformer.ln_prelude(e)
        return e

    def start(self, ids, i, R):
        """As v163.damped_loss: seed 777 + i, prelude, random state init."""
        self.ids, self.R, self.i = ids, R, i
        self.freqs = self.m.freqs_cis[:, : ids.shape[1]]
        torch.manual_seed(777 + i)
        with torch.no_grad():
            e = self.prelude()
            x0 = self.m.initialize_state(e)
        return x0, e

    def ctx_for_params(self, e):
        return self.prelude()                       # with grad: P_t reaches the prelude through e (as v191's c_j)

    def F(self, x, t, e):
        self.m._current_input_ids = self.ids
        return self.m.core_block_forward(x, e, self.freqs, None, torch.tensor(t), torch.tensor(self.R))

    def loss(self, x):
        m = self.m
        x = m.transformer.C(x); off = m.config.n_layers_in_prelude + m.config.n_layers_in_recurrent_block
        for i, block in enumerate(m.transformer.coda):
            x = block(x, self.freqs, None, ve=self.ve(off + i))
        logits = m.lm_head(m.transformer.ln_f(x)).float() * m.config.init.logit_scale
        if m.config.logit_softcap is not None:
            logits = m.config.logit_softcap * torch.tanh(logits / m.config.logit_softcap)
        return Fn.cross_entropy(logits.view(-1, logits.shape[-1]), self.v.labels_of(self.ids).view(-1), ignore_index=-100)

    def ref_loss(self):
        """The model's loss through v163's own path (a = 1, no loop gradient), same seed: the read-out check."""
        self.d.R = self.R; torch.manual_seed(777 + self.i)
        with torch.no_grad():
            return float(self.d.damped_loss(self.m, self.ids, 0, 1.0, ckpt=False))

    def has_readout_grad(self):
        return True


class ProbeFam:
    def __init__(self, seqs_spec):
        import loop_grad_probe as L
        self.L = L
        tok, model = L.load()
        self.ad = {'huginn': L.Huginn, 'parcae': L.Parcae}[L.FAMILY](model)
        self.params = list(self.ad.subset)
        self.dtype, self.dev, self.out_dir = L.DTYPE, L.DEV, L.OUT
        allseqs = L.data(tok)
        idx = [i for part in seqs_spec.split(',') for i in range(int(part.split('-')[0]), int(part.split('-')[-1]) + 1)]
        self._seqs = [(si, allseqs[si][0], allseqs[si][1].unsqueeze(0)) for si in idx if si < len(allseqs)]
        self.name = L.REPO

    def seqs(self):
        return self._seqs

    def start(self, ids, i, R):
        self.ids, self.R = ids, R
        h0, self.ctx = self.ad.prepare(ids)
        return h0, self.ctx

    def ctx_for_params(self, ctx):
        return ctx                                  # the subset holds loop parameters only (as v192)

    def F(self, x, t, ctx):
        return self.ad.F(x, t, ctx)

    def loss(self, x):
        return self.L.ce(self.ad.readout(x, self.ctx), self.ids)

    def ref_loss(self):
        return None

    def has_readout_grad(self):
        return False


# ------------------------------------------------------------------------------------------------- vector records
class Recorder:
    """Exact float64 norms and a CountSketch of parameter vectors (lists of per-parameter tensors, None = zero).
    Hashes are drawn once per model from fixed seeds, so every vector shares them."""
    def __init__(self, params, dev):
        self.dev, self.idx, self.sgn = dev, [], []
        for pi, q in enumerate(params):
            gen = torch.Generator(device=dev); gen.manual_seed(10007 * (pi + 1))
            self.idx.append(torch.randint(0, SKETCH_D, (q.numel(),), generator=gen, device=dev, dtype=torch.int64).to(torch.int32))
            self.sgn.append((torch.randint(0, 2, (q.numel(),), generator=gen, device=dev, dtype=torch.int64) * 2 - 1).to(torch.int8))
        self.sizes = [q.numel() for q in params]
        self.sk, self.n2, self.keep = [], [], []

    def add(self, grads):
        out = torch.zeros(SKETCH_D, dtype=torch.float64, device=self.dev); n2 = 0.0
        if KEEP:
            self.keep.append(torch.cat([(g.detach().reshape(-1).double().cpu() if g is not None else torch.zeros(n, dtype=torch.float64))
                                        for g, n in zip(grads, self.sizes)]))
        for g, ix, sg in zip(grads, self.idx, self.sgn):
            if g is None:
                continue
            flat = g.detach().reshape(-1)
            for a in range(0, flat.numel(), CHUNK):
                x = flat[a:a + CHUNK].double()
                n2 += float((x * x).sum())
                out.index_add_(0, ix[a:a + CHUNK], x * sg[a:a + CHUNK].double())
        self.sk.append(out.cpu()); self.n2.append(n2)
        return math.sqrt(n2)

    def gram(self):
        S = torch.stack(self.sk); G = (S @ S.T).numpy(); self.sk = []
        E = None
        if KEEP:
            X = torch.stack(self.keep); E = (X @ X.T).numpy(); self.keep = []
        return G, np.sqrt(np.array(self.n2)), E


def vjp(out, inputs, cot, dtype):
    return torch.autograd.grad(out, inputs, grad_outputs=cot.to(dtype), retain_graph=True, allow_unused=True)


def signals(fam, x0, ctx, R):
    """lam[t] = dL/d(output of loop t), fp32 copies, parameters frozen."""
    lam = {}
    x = x0.detach().requires_grad_(True); leaf = x
    for t in range(R):
        x = checkpoint(fam.F, x, t, ctx, use_reentrant=False)
        x.register_hook(lambda g, t=t: lam.__setitem__(t, g.detach().float().clone()))
    torch.autograd.grad(fam.loss(x), leaf)
    return [lam[t] for t in range(R)]


def one_sequence(fam, ids, i, R, K, M):
    x0, ctx = fam.start(ids, i, R)
    traj = [x0]
    with torch.no_grad():
        for t in range(R):
            traj.append(fam.F(traj[-1], t, ctx))
        loss_traj = float(fam.loss(traj[-1]))
    step = [float((traj[t + 1] - traj[t]).float().norm()) / max(float(traj[t].float().norm()), 1e-30) for t in range(R)]
    ref = fam.ref_loss()
    lam = signals(fam, traj[0], ctx, R)
    rec = Recorder(fam.params, fam.dev) if not hasattr(fam, 'rec') else fam.rec
    fam.rec = rec
    rec.sk, rec.n2, rec.keep = [], [], []
    for q in fam.params:
        q.requires_grad_(True)
    # g_0: no loop gradient, only the read-out part
    if fam.has_readout_grad():
        g0 = torch.autograd.grad(fam.loss(traj[R].detach()), fam.params, allow_unused=True)
        rec.add(g0); del g0
    else:
        rec.add([None] * len(fam.params))
    ectx = fam.ctx_for_params(ctx)
    u = lam[R - 1] / lam[R - 1].norm()
    cn, pn = [0.0] * R, [0.0] * R
    for t in range(R):                                   # c_t then p_t, t = 0 .. R-1
        out = fam.F(traj[t].detach(), t, ectx)
        cn[t] = rec.add(vjp(out, fam.params, lam[t], fam.dtype))
        del out
    for t in range(R):
        out = fam.F(traj[t].detach(), t, ectx)
        pn[t] = rec.add(vjp(out, fam.params, u, fam.dtype))
        del out
    xin = traj[R - 1].detach().requires_grad_(True)
    out = fam.F(xin, R - 1, ectx)
    kap = [lam[R - 1].clone()]
    for k in range(K):                                   # chat_k and kap_{k+1}
        gs = vjp(out, [xin] + fam.params, kap[k], fam.dtype)
        rec.add(gs[1:]); kap.append(gs[0].detach().float()); del gs
    q0 = lam[R - 1].double(); lam_out = float(q0.norm()); Q = [q0 / q0.norm()]
    H = np.zeros((M + 1, M))
    for l in range(M):                                   # Arnoldi on J^T, P^T q_l recorded
        gs = vjp(out, [xin] + fam.params, Q[l].float(), fam.dtype)
        rec.add(gs[1:]); w = gs[0].detach().double(); del gs
        for _ in range(2):
            for jj in range(l + 1):
                h = float((Q[jj] * w).sum()); H[jj, l] += h; w = w - h * Q[jj]
        H[l + 1, l] = float(w.norm()); Q.append(w / w.norm())
    del out, xin
    for q in fam.params:
        q.requires_grad_(False); q.grad = None
    Qm = torch.stack([q.flatten() for q in Q])
    orth = float((Qm @ Qm.T - torch.eye(len(Q), dtype=torch.float64, device=Qm.device)).abs().max())
    # Arnoldi check: kap_i rebuilt from H vs the directly computed kap_i (i < M), via norms and inner products
    V = torch.stack([v.flatten().double() for v in lam + kap])
    SG = (V @ V.T).cpu().numpy()
    kq = (torch.stack([v.flatten().double() for v in kap[:M]]) @ Qm[:M].T).cpu().numpy()     # coordinates in Q
    del V, Qm, Q
    G, nrm, E = rec.gram()
    i_c, i_p, i_h, i_q = 1, 1 + R, 1 + 2 * R, 1 + 2 * R + K
    dg = np.sqrt(np.clip(np.diag(G), 0, None))
    nz = nrm > 0
    row = dict(i=i, adj=[float(x.norm()) for x in lam], step=step, loss_traj=loss_traj, loss_ref=ref,
               state_gram=SG.tolist(), kap_in_Q=kq.tolist(), H=H.tolist(), lam_out=lam_out, arnoldi_orth=orth,
               param_gram=G.tolist(), param_norm=nrm.tolist(),
               layout=dict(g0=0, c=i_c, p=i_p, chat=i_h, pq=i_q, n=len(nrm)),
               val_norm_maxrel=float(np.max(np.abs(dg[nz] / nrm[nz] - 1))),
               krylov_step1_cos=float(SG[R + 1, R - 2] / math.sqrt(SG[R + 1, R + 1] * SG[R - 2, R - 2])))
    if E is not None:
        row['param_gram_exact'] = E.tolist()
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('family', choices=['parcae', 'probe'])
    ap.add_argument('--repo', default=None); ap.add_argument('--R', type=int, default=32); ap.add_argument('--K', type=int, default=32)
    ap.add_argument('--M', type=int, default=None); ap.add_argument('--seqs', default='0-15'); ap.add_argument('--tag', default='x')
    ap.add_argument('--limit', type=int, default=None)
    A = ap.parse_args(); M = A.M or A.R
    t0 = time.time()
    fam = ParcaeFam(A.repo) if A.family == 'parcae' else ProbeFam(A.seqs)
    out = dict(model=fam.name, R=A.R, K=A.K, M=M, dtype=str(fam.dtype), sketch_d=SKETCH_D,
               n_params=sum(q.numel() for q in fam.params), rows=[])
    print('v195: %s | %.1fM parameters | dtype %s | R %d K %d M %d' % (fam.name, out['n_params'] / 1e6, fam.dtype, A.R, A.K, M),
          flush=True)
    for n, (i, src, ids) in enumerate(fam.seqs()):
        if A.limit is not None and n >= A.limit:
            break
        ts = time.time()
        r = one_sequence(fam, ids.to(fam.dev), i, A.R, A.K, M); r['src'] = src; r['s'] = round(time.time() - ts, 1)
        out['rows'].append(r)
        print('seq %2d %-8s loss %.4f (ref %s) | |lam| last/first %.4f | Krylov step-1 cos %.6f | Arnoldi orth %.1e | '
              'sketch norm check %.1e (%.0fs, total %.0fs)' % (
                  i, src, r['loss_traj'], 'n/a' if r['loss_ref'] is None else '%.4f' % r['loss_ref'], r['adj'][0] / r['adj'][-1],
                  r['krylov_step1_cos'], r['arnoldi_orth'], r['val_norm_maxrel'], r['s'], time.time() - t0), flush=True)
        json.dump(out, open(os.path.join(fam.out_dir, 'v195_%s_R%d.json' % (A.tag, A.R)), 'w'))
        if fam.dev == 'cuda':
            torch.cuda.empty_cache()
    print('done %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
