"""Fig. 1 (pgfplots) from results/e1.json. Writes figures/fig_coverage.tex."""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
r = json.load(open(os.path.join(ROOT, "results", "e1.json")))
rows = r["synthetic"]["rows"]


def pts(method):
    return " ".join(f"({x['N']},{x['coverage']:.4f})" for x in rows if x["method"] == method)


curve = " ".join(f"({N},{c:.4f})" for N, c in r["eq2"]["curve"])
tex = rf"""%% Fig. 1 (E1), written by experiments/make_figure.py from results/e1.json ({r['replicates']:,} replicates per size) and Eq. (2)
\begin{{tikzpicture}}
\begin{{semilogxaxis}}[scale only axis, width=0.84\columnwidth, height=2.3cm,
  xmin=110, xmax=6500, ymin=-0.03, ymax=1.05,
  xtick={{143,300,1000,2000,5000}}, xticklabels={{143,300,1000,2000,5000}},
  ytick={{0,0.25,0.5,0.75,1}}, yticklabels={{0,0.25,0.50,0.75,1}},
  tick label style={{font=\small}}, label style={{font=\small}},
  xlabel={{Cancers in source validation study}}, ylabel={{Coverage of $\mathrm{{Sens}}_T$}},
  ylabel style={{yshift=-3pt}}, xlabel style={{yshift=2pt}},
  legend style={{font=\small, draw=none, fill=white, fill opacity=0.85, text opacity=1, at={{(0.99,0.07)}}, anchor=south east, row sep=-2pt}},
  legend cell align=left, axis lines*=left, tick align=outside]
\addplot[gray, dashed, thin, forget plot] coordinates {{(110,0.95) (6500,0.95)}};
\addplot[orange!80!black, thin, opacity=0.7] coordinates {{{curve}}};
\addlegendentry{{Pooled, Eq.~(2)}}
\addplot[only marks, mark=square*, mark size=1.4pt, orange!80!black] coordinates {{{pts('Pooled')}}};
\addlegendentry{{Pooled, sim.}}
\addplot[densely dotted, mark=triangle*, mark size=1.6pt, yellow!60!black] coordinates {{{pts('Wald')}}};
\addlegendentry{{Std.\ Wald}}
\addplot[mark=*, mark size=1.3pt, blue!70!black] coordinates {{{pts('Jeffreys')}}};
\addlegendentry{{Jeffreys cert.}}
\addplot[dashed, mark=diamond*, mark size=1.7pt, green!50!black] coordinates {{{pts('Exact')}}};
\addlegendentry{{Exact cert.}}
\end{{semilogxaxis}}
\end{{tikzpicture}}
"""
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
open(os.path.join(ROOT, "figures", "fig_coverage.tex"), "w").write(tex)
print(tex)
