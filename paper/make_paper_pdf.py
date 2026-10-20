"""Render paper_looped/draft.md to paper_looped/paper.pdf (HTML + headless Chrome; no LaTeX installed).
Figures are placed inline with captions; the draft's status line and caption list are dropped.
  PYTHONPATH=<dir with the markdown package> python make_paper_pdf.py
"""
import base64, os, re, subprocess
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
P = HERE
src = open(os.path.join(P, 'draft.md'), encoding='utf-8').read()

title = src.splitlines()[0].lstrip('# ').strip()
body = src.split('\n', 1)[1]
body = re.sub(r'^\*Draft v\d.*?\*\s*$', '', body, count=1, flags=re.M | re.S)       # drop the status line
caps = re.search(r'## Figure captions\n(.*?)\n---', body, re.S).group(1)


def cap_of(n):                                       # caption text of Figure n, sub-bullets flattened
    c = re.search(r'\*\*Figure %d\.\*\*(.*?)(?=\n- \*\*Figure|\Z)' % n, caps, re.S).group(1)
    return ' '.join(l.strip().lstrip('- ').strip() for l in c.strip().splitlines())


cap1, cap2, cap3, cap4 = (cap_of(n) for n in (1, 2, 3, 4))
body = re.sub(r'## Figure captions\n.*?\n---', '---', body, flags=re.S)


def fig(name, cap, n, width):
    b64 = base64.b64encode(open(os.path.join(P, name), 'rb').read()).decode()
    cap = markdown.markdown(cap).replace('<p>', '').replace('</p>', '')
    return ('\n\n<figure><img src="data:image/png;base64,%s" style="width:%s"><figcaption><b>Figure %d.</b> %s</figcaption>'
            '</figure>\n\n' % (b64, width, n, cap))


body = body.replace('**A dial for ρ.**', fig('fig3_fit.png', cap1, 1, '55%') + '**A dial for ρ.**', 1)
body = body.replace('**Where the rule needs care.**', fig('fig1_rule.png', cap2, 2, '52%') + '**Where the rule needs care.**', 1)
body = body.replace('**Across models (S11).**', fig('fig4_cost_vs_lost.png', cap3, 3, '52%') + '**Across models (S11).**', 1)
body = body.replace('**The earlier tests.**', fig('fig2_cost.png', cap4, 4, '92%') + '**The earlier tests.**', 1)
body = body.replace('**Figure 2** plots', 'Figure 2 plots')

lines, out_lines = body.split('\n'), []
for i, line in enumerate(lines):                     # Python-Markdown needs a blank line before a list
    is_item = re.match(r'^\s*(- |\d+\. )', line)
    prev = out_lines[-1] if out_lines else ''
    if is_item and prev.strip() and not re.match(r'^\s*(- |\d+\. )', prev) and not prev.startswith('  '):
        out_lines.append('')
    m = re.match(r'^  (\S.*)$', line)                # two-space-indented display lines -> centred equations
    if m and not prev.strip():
        line = '<p class="eq">%s</p>' % m.group(1)
    out_lines.append(line)
body = '\n'.join(out_lines)
body = re.sub(r'^  (- |\d+\. )', r'    \1', body, flags=re.M)                     # 2-space nested items -> 4 spaces
body = re.sub(r'\^(\d+(?:\.\d+)?)', r'<sup>\1</sup>', body)                         # x^3.38 -> superscript

html_body = markdown.markdown(body, extensions=['tables', 'sane_lists'])
css = """
@page { size: Letter; margin: 0.9in 0.95in; }
body { font-family: 'Times New Roman', Times, serif; font-size: 10.6pt; line-height: 1.38; color: #111; }
h1 { font-size: 16pt; text-align: center; margin: 0 0 4pt 0; line-height: 1.25; }
.author { text-align: center; font-size: 10.5pt; margin-bottom: 14pt; color: #333; }
h2 { font-size: 12pt; margin: 14pt 0 4pt 0; break-after: avoid; page-break-after: avoid; }
h2:first-of-type { text-align: center; font-size: 11pt; }
p { margin: 0 0 6pt 0; text-align: justify; }
ul, ol { margin: 0 0 6pt 0; padding-left: 18pt; }
li { margin-bottom: 2pt; }
table { border-collapse: collapse; margin: 6pt auto 4pt auto; font-size: 9pt; }
th, td { border-top: 0.6pt solid #888; border-bottom: 0.6pt solid #888; padding: 2.5pt 5pt; text-align: left; }
th { border-top: 1.2pt solid #000; background: #f3f3f3; }
td:nth-child(n+3), th:nth-child(n+3) { white-space: nowrap; }
figure { margin: 8pt 0 10pt 0; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 9pt; text-align: left; margin: 3pt 4% 0 4%; }
hr { border: 0; border-top: 0.6pt solid #aaa; margin: 14pt 0; page-break-after: always; }
code { font-family: Consolas, monospace; font-size: 9pt; }
p.eq { text-align: center; margin: 6pt 0; }
"""
html = ('<!doctype html><html><head><meta charset="utf-8"><title>%s</title><style>%s</style></head><body>'
        '<h1>%s</h1><div class="author">Anonymous authors</div>%s</body></html>') % (title, css, title, html_body)
hp = os.path.join(P, 'paper.html'); open(hp, 'w', encoding='utf-8').write(html)
chrome = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
out = os.path.join(P, 'paper.pdf')
subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', '--print-to-pdf=' + out,
                'file:///' + hp.replace('\\', '/')], check=True, capture_output=True, timeout=120)
print('wrote', out, os.path.getsize(out) // 1024, 'KB')
