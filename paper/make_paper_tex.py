"""Convert paper_looped/draft.md to a TMLR LaTeX submission in paper_looped/tmlr/ and compile it with tectonic.
The converter is written for this draft's markdown (headings, paragraphs, '- ' lists, pipe tables, bracket citations,
unicode math); it is not a general markdown converter. Style files: github.com/JmlrOrg/tmlr-style-file.
  python make_paper_tex.py            # writes paper_looped/tmlr/main.tex and main.pdf (anonymous submission)
  python make_paper_tex.py --preprint # also paper_looped/arxiv/: named preprint + arxiv_source.tar.gz (with .bbl)
"""
import io, os, re, shutil, subprocess, sys, tarfile

AUTHOR_NAME = 'Aur Shalev Merin'                   # preprint only; the submission stays anonymous
AUTHOR_EMAIL = 'aur.merin@gmail.com'
AUTHOR_ADDR = 'Independent researcher'

HERE = os.path.dirname(os.path.abspath(__file__))
P = HERE
T = os.path.join(P, 'tmlr')
src = open(os.path.join(P, 'draft.md'), encoding='utf-8').read()

KEYS = {('Aicher', '2019'): 'aicher2019', ('Ajalloeian', '2020'): 'ajalloeian2020', ('Fung', '2022'): 'fung2022',
        ('Geiping', '2025'): 'geiping2025', ('Geng', '2021'): 'geng2021', ('McLeish', '2026'): 'mcleish2025',
        ('Zhu', '2025'): 'zhu2025ouro', ('Wang', '2026'): 'wang2026hrm', ('Bansal', '2022'): 'bansal2022',
        ('Prairie', '2026'): 'prairie2026parcae', ('Tallec', '2017'): 'tallec2017', ('Huang', '2026'): 'huang2026partii',
        ('Movahedi', '2026'): 'movahedi2026fprm', ('Liao', '2018'): 'liao2018', ('Yang', '2026'): 'yang2026stars',
        ('Guo', '2026'): 'guo2026', ('Schwethelm', '2026a'): 'schwethelm2026a', ('Schwethelm', '2026b'): 'schwethelm2026b',
        ('Shaban', '2019'): 'shaban2019', ('Vicol', '2022'): 'vicol2022', ('Allal', '2025'): 'allal2025smollm2',
        ('Cobbe', '2021'): 'cobbe2021gsm8k', ('Merity', '2017'): 'merity2017', ('Loshchilov', '2019'): 'loshchilov2019',
        ('Xiong', '2026'): 'xiong2026truncate', ('Li', '2026'): 'li2026policymemory', ('Shakhvalieva', '2026'): 'shakhvalieva2026',
        ('Gao', '2025'): 'gao2025urm', ('Marchenko', '2026'): 'marchenko2026', ('Pascanu', '2013'): 'pascanu2013',
        ('Wu', '2018'): 'wu2018shorthorizon', ('Bai', '2019'): 'bai2019deq', ('Dehghani', '2019'): 'dehghani2019ut',
        ('Saunshi', '2025'): 'saunshi2025looped'}

# ---------------------------------------------------------------- inline conversion
PH = []                                    # protected LaTeX fragments


def prot(tex):
    PH.append(tex)
    return '\ue000%d\ue001' % (len(PH) - 1)


def unprot(s):
    while '\ue000' in s:
        s = re.sub('\ue000(\\d+)\ue001', lambda m: PH[int(m.group(1))], s)
    return s


MATHCH = {'ρ': r'\rho', 'α': r'\alpha','λ': r'\lambda', 'μ': r'\mu', 'Δ': r'\Delta', 'Ā': r'\bar{A}', '−': '-', '·': r'\cdot ',
          '×': r'\times ', '≈': r'\approx ', '≥': r'\geq ', '≤': r'\leq ', '≫': r'\gg ', '←': r'\leftarrow ', '→': r'\to ',
          '∝': r'\propto ', '±': r'\pm ', 'ᵏ': '^{k}', 'ᴿ': '^{R}', 'ʲ': '^{j}', '²': '^{2}', '⁴': '^{4}', '⁸': '^{8}',
          'ⱼ': '_{j}', 'Σ': r'\sum'}


def m(expr):
    """unicode math expression -> protected $...$"""
    out = ''.join(MATHCH.get(ch, ch) for ch in expr)
    out = out.replace('^{2}^{k}', '^{2k}').replace('^{2}^{R}', '^{2R}')
    return prot('$' + out.strip() + '$')


MANUAL = [   # whole formulas, longest first
    ('https://github.com/Aur-Shalev-Merin/lost-gradient-looped-transformers', r'\url{https://github.com/Aur-Shalev-Merin/lost-gradient-looped-transformers}'),
    ('√Q_k / S_0 ≤ lost gradient ≤ S_k / √Q_0', r'$\sqrt{Q_k}/S_0 \leq \text{lost gradient} \leq S_k/\sqrt{Q_0}$'),
    ('S_k / √(Q_0 − Q_k + S_k²)', r'$S_k/\sqrt{Q_0 - Q_k + S_k^2}$'),
    ('√Q_k / (S_0 − S_k + √Q_k)', r'$\sqrt{Q_k}/(S_0 - S_k + \sqrt{Q_k})$'),
    ('Σ_A sⱼ² ≤ |Σ_A cⱼ|² ≤ (Σ_A sⱼ)²', r'$\sum_A s_j^2 \leq |\sum_A c_j|^2 \leq (\sum_A s_j)^2$'),
    ('|Σ cⱼ|² = Σ |cⱼ|²', r'$|\sum c_j|^2 = \sum |c_j|^2$'),
    ('S_k = Σ_{j≥k} sⱼ', r'$S_k = \sum_{j\geq k} s_j$'), ('Q_k = Σ_{j≥k} sⱼ²', r'$Q_k = \sum_{j\geq k} s_j^2$'),
    ('sⱼ = |cⱼ|', r'$s_j = |c_j|$'), ('sⱼ = ρʲ', r'$s_j = \rho^{j}$'), ('R → ∞', r'$R \to \infty$'),
    ('u = j*/R ≤ 1 (for j* ≥ R the fit gives μ_f)', r'$u = j^*/R \leq 1$ (for $j^* \geq R$ the fit gives $\mu_f$)'),
    ('|E g − E g_k| / |E g|', r'$|\mathbb{E}\,g - \mathbb{E}\,g_k| / |\mathbb{E}\,g|$'),
    ('|g − g_k| / |g − g_0| with g_0', r'$|g - g_k| / |g - g_0|$ with $g_0$'),
    ('|g − g_k| / |g − g_0|', r'$|g - g_k| / |g - g_0|$'),
    ('|Σ (G_b − T_{k,b})| / |Σ G_b|', r'$|\sum_b (G_b - T_{k,b})| / |\sum_b G_b|$'),
    ('log ρ(R) = a_s + (a_f − a_s)(3u² − 2u³), with u = j*/R.', r'$\log\rho(R) = a_s + (a_f - a_s)(3u^2 - 2u^3)$, with $u = j^*/R$.'),
    ('a_s + (a_f − a_s)·Cov(x, −(u − x)₊)/Var(x)', r'$a_s + (a_f - a_s)\,\mathrm{Cov}(x, -(u-x)_+)/\mathrm{Var}(x)$'),
    ('Cov(x, −(u − x)₊) = u²/4 − u³/6', r'$\mathrm{Cov}(x, -(u-x)_+) = u^2/4 - u^3/6$'),
    ('Var(x) = 1/12', r'$\mathrm{Var}(x) = 1/12$'), ('A·μ_f^j + B·μ_s^j', r'$A\mu_f^{\,j} + B\mu_s^{\,j}$'),
    ('a_f = log μ_f', r'$a_f = \log\mu_f$'), ('a_s = log μ_s', r'$a_s = \log\mu_s$'), ('x = j/R', r'$x = j/R$'),
    ('μ_f', r'$\mu_f$'), ('μ_s', r'$\mu_s$'), ('log sⱼ', r'$\log s_j$'), ('j* ≈ 17', r'$j^* \approx 17$'), ('crossover j*', r'crossover $j^*$'),
    ('β₁ = 0.9', r'$\beta_1 = 0.9$'), ('T_{k,b}', r'$T_{k,b}$'), ('G_b', r'$G_b$'),
    ('D_k = ρᵏ Σ_{i<R−k} ρⁱ b_{i+k}', r'$D_k = \rho^{k} \sum_{i<R-k} \rho^{i} b_{i+k}$'),
    ('D_0 = Σ_{i<R} ρⁱ bᵢ', r'$D_0 = \sum_{i<R} \rho^{i} b_i$'),
    ('Σ ρ^(i+i′) C(i′ − i) over i, i′ < N', r"$\sum_{i,i'<N} \rho^{i+i'} C(i'-i)$"),
    ('Σ_{i<N} ρⁱ b_{i+k}', r'$\sum_{i<N} \rho^{i} b_{i+k}$'),
    ('E|D_k|² / E|D_0|²', r'$\mathbb{E}|D_k|^2 / \mathbb{E}|D_0|^2$'),
    ('b_{j+1} = Q bⱼ', r'$b_{j+1} = Q b_j$'), ('cⱼ = ρʲ bⱼ', r'$c_j = \rho^{j} b_j$'),
    ('ρ^(R−k)', r'$\rho^{R-k}$'), ('Qᵏ', r'$Q^{k}$'),
    ('D_k = g − g_k', r'$D_k = g - g_k$'), ('log|λⱼ|', r'$\log|\lambda_j|$'),
    ('(j = 0 is the last loop)', r'($j = 0$ is the last loop)'), ('after loop j (the adjoint)', r'after loop $j$ (the adjoint)'),
    ("loop j's contribution", r"loop $j$'s contribution"), ('the loops j ≥ k', r'the loops $j \geq k$'),
    ('g, the full gradient, and', r'$g$, the full gradient, and'), ('the loop j steps', r'the loop $j$ steps'),
    ('1/(1 − δ)', r'$1/(1-\delta)$'), ('δ < 1', r'$\delta < 1$'),
    ('x ← (1 − a)·x + a·F(x)', r'$x \leftarrow (1-a)\,x + a\,F(x)$'),
    ('x ← (1 − a)x + aF(x)', r'$x \leftarrow (1-a)\,x + a\,F(x)$'),
    ('√((ρ²ᵏ − ρ²ᴿ)/(1 − ρ²ᴿ))', r'$\sqrt{(\rho^{2k}-\rho^{2R})/(1-\rho^{2R})}$'),
    ('√((1 + ρ)/(1 − ρ))', r'$\sqrt{(1+\rho)/(1-\rho)}$'),
    ('ρᵢᵏ', r'$\rho_i^{k}$'),
    ('√(Σ_{j≥k} sⱼ² / Σⱼ sⱼ²)', r'$\sqrt{\sum_{j\geq k} s_j^2 / \sum_j s_j^2}$'),
    ('Σ_{j≥k} sⱼ / Σⱼ sⱼ', r'$\sum_{j\geq k} s_j / \sum_j s_j$'), ('sⱼ', r'$s_j$'),
    ('√(Σ_{j≥k} |cⱼ|²) / Σⱼ |cⱼ|', r'$\sqrt{\sum_{j\geq k} |c_j|^2} \,/\, \sum_j |c_j|$'),
    ('Σ_{j≥k} |cⱼ| / √(Σⱼ |cⱼ|²)', r'$\sum_{j\geq k} |c_j| \,/\, \sqrt{\sum_j |c_j|^2}$'),
    ('Σ_{j≥k} |cⱼ| / Σⱼ |cⱼ|', r'$\sum_{j\geq k} |c_j| \,/\, \sum_j |c_j|$'),
    ('|c_k| = |D_k| √(1 + r² − 2r cos θ_k)', r'$|c_k| = |D_k|\sqrt{1 + r^2 - 2r\cos\theta_k}$'),
    ('c_k = D_k − D_{k+1}', r'$c_k = D_k - D_{k+1}$'), ('|D_k| / |D_0|', r'$|D_k| / |D_0|$'),
    ('ρ^(−j)', r'$\rho^{-j}$'), ('D_k', r'$D_k$'), ('θ_k', r'$\theta_k$'),
    ('(ρᵏ − ρᴿ)/(1 − ρᴿ)', r'$(\rho^{k}-\rho^{R})/(1-\rho^{R})$'),
    ('g = Σⱼ cⱼ', r'$g = \sum_j c_j$'), ('g_k = Σ_{j<k} cⱼ', r'$g_k = \sum_{j<k} c_j$'),
    ('|cⱼ| ∝ ρʲ', r'$|c_j| \propto \rho^{j}$'), ('cⱼ', r'$c_j$'),
    ('lost² = 1 + r² − 2r·cos', r'$\mathrm{lost}^2 = 1 + r^2 - 2r\cos$'),
    ('g = g_full +\nλ(g_k − g_full)', r'$g = g_\mathrm{full} + \lambda(g_k - g_\mathrm{full})$'),
    ('g = g_full + λ(g_k − g_full)', r'$g = g_\mathrm{full} + \lambda(g_k - g_\mathrm{full})$'),
    ('H + α(F(H) − H)', r'$H + \alpha(F(H) - H)$'),
    ('cost = 0.339 · (lost gradient)^3.38', r'$\mathrm{cost} = 0.339\cdot(\text{lost gradient})^{3.38}$'),
    ('cost = 1.33 · (1 − Ā)^0.94', r'$\mathrm{cost} = 1.33\cdot(1-\bar{A})^{0.94}$'),
    ('φ = ', r'$\varphi = $'),
    ('|mean ΔĀ| ≤ 0.01', r'$|\text{mean}\ \Delta\bar{A}| \leq 0.01$'),
    ('1 − a + aμ', r'$1 - a + a\mu$'), ('1.4 × 10⁸', r'$1.4\times 10^{8}$'),
    ('a·k', r'$a\cdot k$'), ('a·b', r'$a\cdot b$'), ('R − k', r'$R-k$'),
    ('max(0.015, 0.25 × cost)', r'$\max(0.015,\ 0.25\times\mathrm{cost})$'),
    ('loop Jacobian J', r'loop Jacobian $J$'),
    ('With blending a,', r'With blending $a$,'), ('does not depend on a ', r'does not depend on $a$ '),
    ('about a times smaller', r'about $a$ times smaller'), ('let b be', r'let $b$ be'), ('ratio r', r'ratio $r$'),
    ('μ to', r'$\mu$ to'), ('H,', r'$H$,'),
    ('ρᵢ > 1', r'$\rho_i > 1$'), ('ρᵢ', r'$\rho_i$'), ('λⱼ', r'$\lambda_j$'), ('g_k', r'$g_k$'), ('D_0', r'$D_0$'), ('bⱼ', r'$b_j$'), ('bᵢ', r'$b_i$'),
]
VAR = r'(?:ρ[ᵏᴿʲ⁴⁸²]*|Ā|λ|α|[kRa])'
NUM = r'(?:[−-]?\d+(?:\.\d+)?|[kR1])'
EQ = re.compile(r'(?<![\w$\ue001])(' + VAR + r' (?:=|<|>|≥|≤|≫) ' + NUM + r'(?: / ' + NUM + r')*)(?![\w])')


def cite_part(part):
    part = part.strip()
    if part == 'Parcae, App. I':
        return r'\citealp[App.~I]{prairie2026parcae}'
    mm = re.match(r'^([A-Z][\w\u00fc]+)(?: et al\.| and [A-Z]\w+)? (\d{4}[ab]?)(?:, (.+))?$', part)
    if not mm:
        return None
    key = KEYS[(mm.group(1), mm.group(2))]
    loc = mm.group(3)
    return (r'\citealp[%s]{%s}' % (loc.replace('. ', '.~'), key)) if loc else (r'\citealp{%s}' % key)


def cites(s):
    s = s.replace('Parcae uses about the last half [Sec. 4.2]', 'Parcae uses about the last half ' + prot(r'\citep[Sec.~4.2]{prairie2026parcae}'))
    s = s.replace('Parcae (App. I) reports', 'Parcae ' + prot(r'\citep[App.~I]{prairie2026parcae}') + ' reports')
    s = s.replace("Geng et al.'s", prot(r"\citeauthor{geng2021}'s"))
    s = s.replace('Shalev Merin [2026]', prot(r'\citet{shalevmerin2026present}'))

    def narr(mm):          # Name et al. [2019, Thm. 1]
        key = KEYS[(mm.group(1), mm.group(2))]
        loc = mm.group(3)
        return prot((r'\citet[%s]{%s}' % (loc.replace('. ', '.~'), key)) if loc else (r'\citet{%s}' % key))
    s = re.sub(r'\b([A-Z]\w+)(?: et al\.| and [A-Z]\w+) \[(\d{4}[ab]?)(?:, ([^\]]+))?\]', narr, s)

    def paren(mm):         # [Name et al. 2019, loc; ...]
        parts = mm.group(1).split(';')
        conv = [cite_part(p) for p in parts]
        if all(c is None for c in conv):
            return mm.group(0)
        bits = [c if c else re.sub(r'\bSection (\d)\b', r'Section~\\ref{sec:\1}', p.strip()) for c, p in zip(conv, parts)]
        return prot('(' + '; '.join(bits) + ')')
    return re.sub(r'\[([^\[\]]*?\d{4}[ab]?[^\[\]]*?)\]', paren, s)


def esc(s):
    for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_'), ('$', r'\$'),
                 ('~', r'\textasciitilde{}'), ('^', r'\^{}')]:
        s = s.replace(a, b)
    return s


def inline(s):
    s = re.sub(r'`([^`]+)`', lambda mm: prot(r'\texttt{%s}' % esc(mm.group(1))), s)
    s = cites(s)
    # cross-references -> \ref (citation locators such as "[..., Table 4]" are already protected by cites())
    s = re.sub(r'\bTables (\d) and (\d)\b', lambda mm: prot(r'Tables~\ref{tab:%s} and~\ref{tab:%s}' % mm.groups()), s)
    s = re.sub(r'\bSections (\d)–(\d)\b', lambda mm: prot(r'Sections~\ref{sec:%s}--\ref{sec:%s}' % mm.groups()), s)
    for word, kind in (('Figure', 'fig'), ('Table', 'tab'), ('Section', 'sec')):
        s = re.sub(r'\b%s (\d+)\b' % word, lambda mm, w=word, k=kind: prot(r'%s~\ref{%s:%s}' % (w, k, mm.group(1))), s)
    s = re.sub(r'\bAppendix ([A-L])\b',lambda mm: prot(r'Appendix~\ref{app:%s}' % mm.group(1)), s)
    for a, b in MANUAL:
        s = s.replace(a, prot(b))
    s = EQ.sub(lambda mm: m(mm.group(1)), s)
    s = re.sub(r'ρ[ᵏᴿʲ⁴⁸²]*', lambda mm: m(mm.group(0)), s)
    s = re.sub(r'(?<=[\s"/])([kR])(?=[\s,.;:)?"])', lambda mm: m(mm.group(1)), s)
    s = re.sub(r'(?<=\()([kR])(?=[\s,.;:)])', lambda mm: m(mm.group(1)), s)
    s = re.sub(r'(R)²', lambda mm: m('R²'), s)
    s = re.sub(r'\*\*(.+?)\*\*', lambda mm: prot(r'\textbf{') + mm.group(1) + prot('}'), s)
    s = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', lambda mm: prot(r'\emph{') + mm.group(1) + prot('}'), s)
    s = re.sub(r'(?<=\w)\*(?=\s|$)', lambda mm: prot(r'$^{*}$'), s)                       # Huginn-3.5B*
    s = re.sub(r'"([^"]+)"', lambda mm: prot('``') + mm.group(1) + prot("''"), s)
    s = re.sub(r'\^(\d+(?:\.\d+)?)', lambda mm: prot('$^{%s}$' % mm.group(1)), s)
    s = s.replace('†', prot(r'\textsuperscript{\dag}')).replace('‡', prot(r'\textsuperscript{\ddag}'))
    s = s.replace('–', prot('--')).replace('ü', prot(r'{\"u}'))
    s = re.sub('[−·×≈≥≤≫←→∝±λμαΔĀ]+', lambda mm: m(mm.group(0)), s)
    s = esc(s)
    return unprot(s)


# ---------------------------------------------------------------- pull out the pieces
lines = src.split('\n')
title = lines[0].lstrip('# ').strip()
body = '\n'.join(lines[1:])
body = re.sub(r'^\*Draft v\d.*?\*\s*$', '', body, count=1, flags=re.M | re.S)
foot = re.search(r'^\*Footnote, first page[^*]*\*:?\s*(.*?)\n\n', body, re.S | re.M)
footnote = ' '.join(foot.group(1).split()) if foot else ''
body = body.replace(foot.group(0), '') if foot else body
caps_raw = re.search(r'## Figure captions\n(.*?)\n---', body, re.S).group(1)
body = re.sub(r'## Figure captions\n.*?\n---', '', body, flags=re.S)
body = body.split('## References')[0]


def caption(n):
    c = re.search(r'\*\*Figure %d\.\*\*(.*?)(?=\n- \*\*Figure|\Z)' % n, caps_raw, re.S).group(1)
    c = re.sub(r'\n\s*- \*\*(Left|Right):\*\*', r' \1:', c)
    return inline(' '.join(c.split()))


FIGS = {1: ('fig3_fit.pdf', 0.55), 2: ('fig1_rule.pdf', 0.55), 3: ('fig4_cost_vs_lost.pdf', 0.55), 4: ('fig2_cost.pdf', 0.95)}   # numbered in citation order


def figure(n):
    f, w = FIGS[n]
    return ('\\begin{figure}[t]\n\\centering\n\\includegraphics[width=%.2f\\linewidth]{%s}\n\\caption{%s}\n'
            '\\label{fig:%d}\n\\end{figure}\n' % (w, f, caption(n), n))


# ---------------------------------------------------------------- block conversion
def is_item(l):
    return re.match(r'^\s*(- |\d+\. )', l)


def conv_list(ls):
    out, stack = [], []
    for l in ls:
        mm = re.match(r'^(\s*)(- |\d+\. )(.*)$', l)
        if mm:
            depth = len(mm.group(1)) // 2
            env = 'enumerate' if mm.group(2)[0].isdigit() else 'itemize'
            while len(stack) > depth + 1:
                out.append('\\end{%s}' % stack.pop())
            if len(stack) < depth + 1:
                stack.append(env)
                out.append('\\begin{%s}' % env)
            out.append('\\item ' + mm.group(3).strip())
        else:
            out[-1] += ' ' + l.strip()
    while stack:
        out.append('\\end{%s}' % stack.pop())
    return '\n'.join(o if o.startswith('\\begin') or o.startswith('\\end') else
                     ('\\item ' + inline(o[6:]) if o.startswith('\\item ') else inline(o)) for o in out)


def conv_table(ls, cap=None, notes=None, label=None, place='t'):
    rows = [[c.strip() for c in l.strip().strip('|').split('|')] for l in ls if not re.match(r'^\|[-|\s]+\|$', l)]

    def cell(c):                                 # header cells ending in a variable: 'loops R', 'blending a', 'k'
        mm = re.match(r'^(.*?)( ?)([kRar])$', c)
        return (inline(mm.group(1)) + mm.group(2) + '$%s$' % mm.group(3)) if mm else inline(c)
    ncol = len(rows[0])
    width = [max(len(r[j]) for r in rows) for j in range(ncol)]
    wrap = max(width) > 45                       # text-heavy table: wrap the long columns
    spec = ''.join('X' if (wrap and w > 25) else 'l' for w in width)
    if wrap and len(rows) > 12:                  # long text-heavy table (Table 4): break across pages, repeat the head
        head = ' & '.join(cell(c) for c in rows[0]) + ' \\\\'
        s = '{\\small\n\\begin{xltabular}{\\linewidth}{%s}\n' % spec
        if cap:
            s += '\\caption{%s}%s\\\\\n' % (cap, '\\label{%s}' % label if label else '')
        s += '\\toprule\n%s\n\\midrule\n\\endfirsthead\n' % head
        s += '\\multicolumn{%d}{l}{\\footnotesize\\itshape (continued)}\\\\\n\\toprule\n%s\n\\midrule\n\\endhead\n' % (ncol, head)
        s += '\\bottomrule\n\\endlastfoot\n'
        s += '\n'.join(' & '.join(cell(c) for c in r) + ' \\\\' for r in rows[1:]) + '\n\\end{xltabular}}\n'
        if notes:
            s += '\\par\\smallskip{\\footnotesize\\raggedright\n' + '\n\n'.join(notes) + '\\par}\n'
        return s
    t = [('\\begin{tabularx}{\\linewidth}{%s}' % spec) if wrap else ('\\begin{tabular}{%s}' % spec), '\\toprule',
         ' & '.join(cell(c) for c in rows[0]) + ' \\\\', '\\midrule']
    t += [' & '.join(cell(c) for c in r) + ' \\\\' for r in rows[1:]]
    t += ['\\bottomrule', '\\end{tabularx}' if wrap else '\\end{tabular}']
    tab = '\n'.join(t)
    s = '\\begin{table}[%s]\n\\centering\\small\n' % place
    if cap:
        s += '\\caption{%s}\n' % cap
    s += '\\resizebox{\\linewidth}{!}{%%\n%s}\n' % tab if (ncol >= 6 and not wrap) else tab + '\n'
    if notes:
        s += '\\par\\smallskip\\footnotesize\\raggedright\n' + '\n\n'.join(notes) + '\n'
    if label:
        s += '\\label{%s}\n' % label
    return s + '\\end{table}\n'


DISPLAY = {
    'log ρ(R) = a_s + (a_f − a_s)(3u² − 2u³), with u = j*/R.':
        r'\begin{equation*}\log\rho(R) = a_s + (a_f - a_s)(3u^2 - 2u^3),\qquad u = j^*/R.\end{equation*}',
    'lost gradient ≈ ρᵏ.': r'\begin{equation*}\text{lost gradient}\approx\rho^{k}.\end{equation*}',
    'lost gradient = |g − g_k| / |g|.': r'\begin{equation*}\text{lost gradient}=\frac{|g-g_k|}{|g|}.\end{equation*}',
    'cost = (loss with k loops − loss with all loops) / (loss with no loop gradient − loss with all loops).':
        r'\begin{equation*}\text{cost}=\frac{\text{loss with $k$ loops}-\text{loss with all loops}}'
        r'{\text{loss with no loop gradient}-\text{loss with all loops}}.\end{equation*}',
}

blocks = re.split(r'\n\s*\n', body.strip())
out, abstract, in_abstract, appendix_started, pending_cap = [], [], False, False, None
i = 0
while i < len(blocks):
    b = blocks[i].strip('\n')
    first = b.lstrip()
    i += 1
    if not first or first == '---':
        continue
    # figures go before these anchors (same placement as the HTML build)
    if first.startswith('**A dial for ρ.**'):
        out.append(figure(1))
    if first.startswith('**Where the rule needs care.**'):
        out.append(figure(2))
    if first.startswith('**Across models (S11).**'):
        out.append(figure(3))
    if first.startswith('**The earlier tests.**'):
        out.append(figure(4))
    if first.startswith('## '):
        h = first[3:].strip()
        if h == 'Abstract':
            in_abstract = True
            continue
        in_abstract = False
        label = ''
        if h.startswith('Appendix'):
            if not appendix_started:
                out.append('\\appendix')
                appendix_started = True
            label = '\\label{app:%s}' % h.split()[1].rstrip('.')
            h = re.sub(r'^Appendix [A-Z]\.\s*', '', h)
        else:
            mn = re.match(r'^(\d+)\.\s*', h)
            label = '\\label{sec:%s}' % mn.group(1) if mn else ''
            h = re.sub(r'^\d+\.\s*', '', h)
        out.append('\\section{%s}%s' % (inline(h), label))
        continue
    if in_abstract:
        abstract.append(inline(' '.join(b.split())))
        continue
    if first.strip() in DISPLAY:
        out.append(DISPLAY[first.strip()])
        continue
    mt = re.match(r'^\*\*Table (\d+)\. (.*?)\*\*\s*(.*)$', ' '.join(b.split()), re.S)
    if mt:                                   # table caption paragraph: hold until the table block
        rest = mt.group(3)                   # '(...)' continues the title; a sentence follows a full stop
        cap = mt.group(2).rstrip('.') + ((' ' if rest.startswith('(') else '. ') + rest if rest else '')
        pending_cap = (int(mt.group(1)), inline(cap.rstrip('.') + '.'), [])
        continue
    if first.startswith('|'):
        notes = []
        while i < len(blocks) and re.match(r'^(\\\*|†|‡)', blocks[i].lstrip()):
            for para in re.split(r'\n(?=†|‡)', blocks[i].strip()):
                para = re.sub(r'^\\\*', '*', para.strip())
                mk = {'*': '$^{*}$', '†': r'\textsuperscript{\dag}', '‡': r'\textsuperscript{\ddag}'}[para[0]]
                notes.append(mk + ' ' + inline(' '.join(para[1:].split())))
            i += 1
        n, cap, between = pending_cap if pending_cap else (None, None, [])
        out.append(conv_table(b.split('\n'), cap, between + notes, 'tab:%d' % n if n else None,
                              '!htbp' if appendix_started else 't'))    # appendix tables stay with their text
        pending_cap = None
        continue
    if pending_cap and not first.startswith('|'):   # explanatory text between caption and table: a table note
        pending_cap[2].append(inline(' '.join(b.split())))
        continue
    ls = b.split('\n')
    k = next((j for j, l in enumerate(ls) if is_item(l)), None)
    if k is None:
        out.append(inline(' '.join(b.split())))
    else:
        if k:
            out.append(inline(' '.join(' '.join(ls[:k]).split())))
        out.append(conv_list(ls[k:]))

abstract_tex = '\n\n'.join(abstract)
ai_footnote = ('\\footnote{%s}' % inline(footnote)) if footnote else ''     # TMLR requires it on page 1; the arXiv build drops it
abstract_tex += ai_footnote

tex = r"""\documentclass[10pt]{article}
\usepackage{tmlr}
%% \usepackage[accepted]{tmlr}   %% camera-ready
%% \usepackage[preprint]{tmlr}   %% de-anonymised preprint
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{xltabular}
\renewcommand{\tabularxcolumn}[1]{>{\raggedright\arraybackslash}p{#1}}
\usepackage{hyperref}
\usepackage{url}

\title{%s}

\author{\name Anonymous authors}

\def\month{MM}
\def\year{YYYY}
\def\openreview{\url{https://openreview.net/forum?id=XXXX}}

\begin{document}
\maketitle

\begin{abstract}
%s
\end{abstract}

%s

\bibliography{main}
\bibliographystyle{tmlr}

\end{document}
""" % (inline(title).replace(': ', ':\\\\ ', 1), abstract_tex, '\n\n'.join(out))
tex = tex.replace('\\bibliography{main}\n\\bibliographystyle{tmlr}\n', '')
# references go before the appendices
tex = tex.replace('\\appendix', '\\bibliography{main}\n\\bibliographystyle{tmlr}\n\n\\clearpage\n\\appendix', 1)

os.makedirs(T, exist_ok=True)
ANON_URL = (r'\url{https://github.com/Aur-Shalev-Merin/lost-gradient-looped-transformers}', 'an anonymised repository (link withheld for double-blind review)')
assert tex.count(ANON_URL[0]) == 1
open(os.path.join(T, 'main.tex'), 'w', encoding='utf-8').write(tex.replace(*ANON_URL))   # the preprint keeps the link
for f, _ in FIGS.values():
    shutil.copy(os.path.join(P, f), os.path.join(T, f))
left = re.findall(r'[^\x00-\x7f]', tex)
print('non-ascii left in main.tex:', sorted(set(left)))
if not os.path.exists(os.path.join(T, 'main.bib')):
    raise SystemExit('main.bib missing: wrote main.tex only')
tectonic = shutil.which('tectonic') or 'tectonic'
r = subprocess.run([tectonic, '--keep-logs', 'main.tex'], cwd=T, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-3000:])
for f in ('main.log', 'main.blg'):                  # logs are only for debugging; keep them out of git
    if os.path.exists(os.path.join(T, f)):
        os.remove(os.path.join(T, f))

# ---------------------------------------------------------------- arXiv preprint (de-anonymised): --preprint
if '--preprint' in sys.argv:
    A = os.path.join(P, 'arxiv')
    os.makedirs(A, exist_ok=True)
    pre = tex.replace('\\usepackage{tmlr}\n', '\\usepackage[preprint]{tmlr}\n', 1)
    if ai_footnote:
        assert pre.count(ai_footnote) == 1
        pre = pre.replace(ai_footnote, '')
    pre = pre.replace('\\author{\\name Anonymous authors}',
                      '\\author{\\name ' + AUTHOR_NAME + ' \\email ' + AUTHOR_EMAIL + ' \\\\\n      \\addr ' + AUTHOR_ADDR + '}', 1)
    open(os.path.join(A, 'main.tex'), 'w', encoding='utf-8').write(pre)   # local build: tectonic (XeTeX)
    for f in ['main.bib', 'tmlr.sty', 'tmlr.bst', 'fancyhdr.sty'] + [f for f, _ in FIGS.values()]:
        shutil.copy(os.path.join(T, f), os.path.join(A, f))
    r = subprocess.run([tectonic, '--keep-intermediates', 'main.tex'], cwd=A, capture_output=True, text=True)
    print('[preprint]', r.stderr[-1500:])
    # arXiv does not run BibTeX: ship the .bbl with the sources
    # tmlr.bst is unused (main.bbl is shipped) but arXiv's scan flags \bibliographystyle{tmlr} as missing without it
    src_files = ['main.bbl', 'tmlr.sty', 'tmlr.bst', 'fancyhdr.sty'] + [f for f, _ in FIGS.values()]
    with tarfile.open(os.path.join(A, 'arxiv_source.tar.gz'), 'w:gz') as tf:
        data = ('\\pdfoutput=1\n' + pre).encode('utf-8')            # arXiv: compile with pdfLaTeX
        info = tarfile.TarInfo('main.tex')
        info.size, info.mtime = len(data), int(__import__("time").time())
        tf.addfile(info, io.BytesIO(data))
        for f in src_files:
            tf.add(os.path.join(A, f), arcname=f)
    for f in os.listdir(A):                          # keep main.bbl (in the tarball) and drop the other intermediates
        if f.startswith('main.') and f.split('.')[-1] in ('aux', 'log', 'blg', 'out', 'xdv', 'toc'):
            os.remove(os.path.join(A, f))
    print('[preprint] wrote', os.path.join(A, 'main.pdf'), 'and arxiv_source.tar.gz')
