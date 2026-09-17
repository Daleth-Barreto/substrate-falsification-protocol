"""make_nc_v2.py -- deterministic port of paper/main_v2.tex to NC style.

Neural Computation (MIT Press) wants a single bibliography-less-source file:
12pt, 1 3/8 in margins, double-spaced, author-date (APA) citations, expanded
tables. Hand-maintaining a second copy is how the v1 NC file drifted
(a duplicate \\end{document} and an author-less reference). Instead this script
*derives* main_nc_v2.tex from the canonical main_v2.tex so the content can
never disagree with the audited manuscript:

  * preamble swapped to the NC layout;
  * \\cite{key,...} -> (Author, Year; ...) via an explicit key map;
  * \\input{tables/X.tex} -> inlined table body;
  * \\path{...} -> \\texttt{...};
  * numbered bibliography replaced by a sorted author-date reference list.

Run from this directory (paper/):
    ..\\..\\02_cl\\.venv312\\Scripts\\python.exe make_nc_v2.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "main_v2.tex")
DST = os.path.join(HERE, "main_nc_v2.tex")

TITLE = ("A falsification protocol for simulated embodied spiking "
         "substrates: Separating substrate-carried information from "
         "readout-channel information")

# key -> inline author-date, used for \cite conversion
CITE = {
    "embodied_neurocomp": "Zhou et al., 2026",
    "kagan2022dishbrain": "Kagan et al., 2022",
    "balci2023neurons": "Balci et al., 2023",
    "kagan2023semantics": "Kagan et al., 2023",
    "smirnova2023organoid": "Smirnova et al., 2023",
    "brooks1991": "Brooks, 1991",
    "pfeifer2006how": "Pfeifer & Bongard, 2006",
    "dangelo2026benchmark": "D'Angelo et al., 2026",
    "bartolozzi2022embodied": "Bartolozzi et al., 2022",
    "neurobench2025": "Yik et al., 2025",
    "maass1997": "Maass, 1997",
    "izhikevich2003": "Izhikevich, 2003",
    "izhikevich2004": "Izhikevich, 2004",
    "gerstner2002": "Gerstner & Kistler, 2002",
    "maass2002liquid": "Maass et al., 2002",
    "jaeger2001echo": "Jaeger, 2001",
    "lukosevicius2009": "Lukosevicius & Jaeger, 2009",
    "wringe2025rc": "Wringe et al., 2025",
    "yan2024rc": "Yan et al., 2024",
    "steinmetz2019": "Steinmetz et al., 2019",
    "shannon1948": "Shannon, 1948",
    "kraskov2004": "Kraskov et al., 2004",
    "panzeri2007": "Panzeri et al., 2007",
    "schreiber2000": "Schreiber, 2000",
    "walker2018": "Walker & Newhall, 2018",
    "theiler1992": "Theiler et al., 1992",
    "ernst2004": "Ernst, 2004",
    "hurlbert1984": "Hurlbert, 1984",
    "georgopoulos1986": "Georgopoulos et al., 1986",
    "quiroga2009": "Quiroga & Panzeri, 2009",
    "paninski2004": "Paninski et al., 2004",
    "hoerl1970": "Hoerl & Kennard, 1970",
    "carmena2003": "Carmena et al., 2003",
    "nicolelis2003": "Nicolelis, 2003",
}

# (sort-key, reference entry) -- sorted author-date list
REFS = [
    ("Balci", "Balci, F., Ben Hamed, S., Boraud, T., Bouret, S., Brochier, T., "
              "Brun, C., & others. (2023). A response to claims of emergent "
              "intelligence and sentience in a dish. Neuron, 111(5), 604-605."),
    ("Bartolozzi", "Bartolozzi, C., Indiveri, G., & Donati, E. (2022). "
                   "Embodied neuromorphic intelligence. Nature Communications, "
                   "13, 1024."),
    ("Brooks", "Brooks, R. A. (1991). Intelligence without representation. "
               "Artificial Intelligence, 47, 139-159."),
    ("Carmena", "Carmena, J. M., & others. (2003). Learning to control a "
                "brain-machine interface for reaching and grasping by primates. "
                "PLoS Biology, 1(2), e42."),
    ("D'Angelo", "D'Angelo, G., Pedersen, J. E., Hassan, T., & others. (2026). "
                 "A benchmarking framework for embodied neuromorphic agents. "
                 "Nature Machine Intelligence, 8, 300-312."),
    ("Ernst", "Ernst, M. D. (2004). Permutation methods: A basis for exact "
              "inference. Statistical Science, 19(4), 676-685."),
    ("Georgopoulos", "Georgopoulos, A. P., Schwartz, A. B., & Kettner, R. E. "
                     "(1986). Neuronal population coding of movement direction. "
                     "Science, 233, 1416-1419."),
    ("Gerstner", "Gerstner, W., & Kistler, W. M. (2002). Spiking neuron models: "
                 "Single neurons, populations, plasticity. Cambridge University "
                 "Press."),
    ("Hoerl", "Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased "
              "estimation for nonorthogonal problems. Technometrics, 12, 55-67."),
    ("Hurlbert", "Hurlbert, S. H. (1984). Pseudoreplication and the design of "
                 "ecological field experiments. Ecological Monographs, 54(2), "
                 "187-211."),
    ("Izhikevich-a", "Izhikevich, E. M. (2003). Simple model of spiking "
                     "neurons. IEEE Transactions on Neural Networks, 14(6), "
                     "1569-1572."),
    ("Izhikevich-b", "Izhikevich, E. M. (2004). Which model to use for cortical "
                     "spiking neurons? IEEE Transactions on Neural Networks, "
                     "15(5), 1063-1070."),
    ("Jaeger", "Jaeger, H. (2001). The echo state approach to analysing and "
               "training recurrent neural networks (GMD-Report 148). GMD."),
    ("Kagan-a", "Kagan, B. J., & others. (2023). Scientific communication and "
                "the semantics of sentience. Neuron, 111(5). [Letter]."),
    ("Kagan-b", "Kagan, B. J., Kitchen, A. C., Tran, N. T., Habibollahi, F., "
                "Khajehnejad, M., Parker, B. J., Bhat, A., Rollo, B., Razi, A., "
                "& Friston, K. J. (2022). In vitro neurons learn and exhibit "
                "sentience when embodied in a simulated game-world. Neuron, "
                "110(22), 3952-3969."),
    ("Kraskov", "Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). "
                "Estimating mutual information. Physical Review E, 69, 066138."),
    ("Lukosevicius", "Lukosevicius, M., & Jaeger, H. (2009). Reservoir "
                     "computing approaches to recurrent neural network training. "
                     "Computer Science Review, 3(3), 127-149."),
    ("Maass-a", "Maass, W. (1997). Networks of spiking neurons: The third "
                "generation of neural network models. Neural Networks, 10(9), "
                "1659-1671."),
    ("Maass-b", "Maass, W., Natschlaeger, T., & Markram, H. (2002). Real-time "
                "computing without stable states: A new framework for neural "
                "computation based on perturbations. Neural Computation, 14(11), "
                "2531-2560."),
    ("Nicolelis", "Nicolelis, M. A. L. (2003). Brain-machine interfaces to "
                  "restore motor function and probe neural circuits. Nature "
                  "Reviews Neuroscience, 4, 417-422."),
    ("Paninski", "Paninski, L., & others. (2004). Superlinear population "
                 "encoding of dynamic hand trajectory in primary motor cortex. "
                 "Journal of Neuroscience, 24(39), 8551-8561."),
    ("Panzeri", "Panzeri, S., Senatore, R., Montemurro, M. A., & Petersen, "
                "R. S. (2007). Correcting for the sampling bias in spike-train "
                "information measures. Journal of Neurophysiology, 98, "
                "1064-1072."),
    ("Pfeifer", "Pfeifer, R., & Bongard, J. (2006). How the body shapes the way "
                "we think: A new view of intelligence. MIT Press."),
    ("Quiroga", "Quiroga, R. Q., & Panzeri, S. (2009). Extracting information "
                "from neuronal populations: Information theory and decoding "
                "approaches. Nature Reviews Neuroscience, 10, 173-185."),
    ("Schreiber", "Schreiber, T. (2000). Measuring information transfer. "
                  "Physical Review Letters, 85(2), 461-464."),
    ("Shannon", "Shannon, C. E. (1948). A mathematical theory of communication. "
                "Bell System Technical Journal, 27, 379-423."),
    ("Smirnova", "Smirnova, L., Caffo, B. S., Gracias, D. H., Huang, Q., "
                 "Morales Pantoja, I. E., & others. (2023). Organoid "
                 "intelligence (OI): The new frontier in biocomputing and "
                 "intelligence-in-a-dish. Frontiers in Science, 1, 1017235."),
    ("Steinmetz", "Steinmetz, N. A., Zatka-Haas, P., Carandini, M., & Harris, "
                  "K. D. (2019). Distributed coding of choice, action and "
                  "engagement across the mouse brain. Nature, 576, 266-273."),
    ("Theiler", "Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & "
                "Farmer, J. D. (1992). Testing for nonlinearity in time series: "
                "The method of surrogate data. Physica D, 58, 77-94."),
    ("Walker", "Walker, B. L., & Newhall, K. A. (2018). Inferring information "
               "flow in spike-train data sets using a trial-shuffle method. "
               "PLoS ONE, 13(11), e0206977."),
    ("Wringe", "Wringe, C., Trefzer, M., & Stepney, S. (2025). Reservoir "
               "computing benchmarks: A tutorial review and critique. "
               "International Journal of Parallel, Emergent and Distributed "
               "Systems, 40(4), 313-351."),
    ("Yan", "Yan, M., Huang, C., Bienstman, P., Ti\u0148o, P., Lin, W., & Sun, "
            "J. (2024). Emerging opportunities and challenges for the future of "
            "reservoir computing. Nature Communications, 15, 2056."),
    ("Yik", "Yik, J., Van den Berghe, K., den Blanken, D., & others. (2025). "
            "The NeuroBench framework for benchmarking neuromorphic computing "
            "algorithms and systems. Nature Communications, 16, 1545."),
    ("Zhou", "Zhou, J., Tanneberg, D., Habibollahi, F., Loeffler, A., Lawson, "
             "K., Baccetti, V., & others. (2026). Embodied neurocomputation: A "
             "framework for interfacing biological neural cultures with scaled "
             "task-driven validation. [Preprint]. arXiv:2605.13315."),
]

PREAMBLE = r"""%% ====================================================================
%%  Neural Computation (MIT Press) submission manuscript  --  v2
%%  GENERATED from main_v2.tex by make_nc_v2.py. Do not edit by hand:
%%  edit main_v2.tex, re-run the converter, then re-run
%%  verify/verify_paper_numbers.py.
%%  - 12pt, 1 3/8 in margins, author-date (APA) citations, expanded tables
%%  - Draft mode (double-spaced, full width) as required for submission
%%  Build: pdflatex main_nc_v2.tex  (twice)
%% ====================================================================
\documentclass[12pt]{article}

\usepackage[margin=1.375in,top=1.375in,bottom=1.375in]{geometry}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{xcolor}
\usepackage[colorlinks=true,urlcolor=blue,citecolor=blue,linkcolor=blue]{hyperref}
\usepackage{parskip}
\usepackage{setspace}
\usepackage{enumitem}
\usepackage{siunitx}
\usepackage{float}

% Neural Computation submission: draft = double-spaced, full width
\doublespacing

% Macros used by the (inlined) tables, mirroring main_v2.tex
\newcommand{\sub}[1]{\ensuremath{\mathit{#1}}}
\newcommand{\RMSE}{\ensuremath{\mathrm{RMSE}}}
\newcommand{\MI}{\ensuremath{\mathrm{MI}}}
\newcommand{\TE}{\ensuremath{\mathrm{TE}}}
\newcommand{\dfloor}{\ensuremath{\Delta\RMSE_{\mathrm{floor}}}}
\newcommand{\palt}{0.75\,u}

\newenvironment{hangparas}{\begin{list}{}{\setlength{\leftmargin}{1.5em}\setlength{\itemindent}{-1.5em}\setlength{\itemsep}{0pt}\setlength{\parsep}{0pt}}}{\end{list}}

\title{TITLE_PLACEHOLDER}
\author{Alan Daleth Hern\'andez Barreto}
\date{}
\begin{document}
\maketitle
"""


def inline_tables(body):
    def repl(m):
        path = os.path.join(HERE, "tables", m.group(1))
        with open(path, encoding="utf-8") as f:
            return f.read().rstrip("\n")
    return re.sub(r"\\input\{tables/([^}]+)\}", repl, body)


def convert_cites(body):
    used = set()

    def repl(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        for k in keys:
            if k not in CITE:
                raise KeyError(f"no author-date mapping for cite key {k!r}")
            used.add(k)
        return "(" + "; ".join(CITE[k] for k in keys) + ")"

    body = re.sub(r"\\cite\{([^}]*)\}", repl, body)
    unused = set(CITE) - used
    if unused:
        print("WARNING: cite keys defined but unused:", sorted(unused))
    return body


def convert_paths(body):
    def repl(m):
        s = m.group(1).replace("\\_", "_").replace("_", "\\_")
        return "\\texttt{" + s + "}"

    return re.sub(r"\\path\{([^}]*)\}", repl, body)


def main():
    with open(SRC, encoding="utf-8") as f:
        src = f.read()

    doc = src.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
    abstract = doc.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    body = doc.split("\\end{abstract}", 1)[1]

    body = body.split("\\bibliographystyle", 1)[0]
    body = inline_tables(body)
    body = convert_cites(body)
    body = convert_paths(body)

    refs = "\n".join(
        "\\noindent " + r + "\n\\par" for _, r in sorted(REFS, key=lambda t: t[0])
    )

    out = (PREAMBLE.replace("TITLE_PLACEHOLDER", TITLE)
           + "\n\\begin{abstract}\n" + abstract.strip() + "\n\\end{abstract}\n"
           + body.rstrip() + "\n\n\\section*{References}\n\n" + refs
           + "\n\\end{document}\n")
    with open(DST, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"WROTE {DST} ({out.count(chr(10)) + 1} lines, "
          f"{len(REFS)} references)")


if __name__ == "__main__":
    main()
