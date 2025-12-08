# ml_pipeline/training/report_generator.py

from __future__ import annotations
import shutil
from pathlib import Path
import datetime
import subprocess
import cairosvg  # <-- NEW for SVG → PDF conversion

TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage{graphicx}
\usepackage{float}
\usepackage{geometry}
\geometry{margin=1in}

\title{Model Report: \texttt{{MODEL_NAME}}}
\date{{DATE}}
\author{Robot Localization Failure Detection Pipeline}

\begin{document}

\maketitle

\section*{1. Experiment Summary}

\begin{itemize}
    \item \textbf{Model Name}: \texttt{{MODEL_NAME}}
    \item \textbf{Algorithm}: {BEST_ALGO}
    \item \textbf{Optuna Trials}: {N_TRIALS}
    \item \textbf{Train Maps}: {TRAIN_MAPS}
    \item \textbf{Eval Maps}: {EVAL_MAPS}
    \item \textbf{Odometry}: {ODOMETRY}
    \item \textbf{Notes}: {NOTES}
\end{itemize}

\section*{2. Feature Toggles}

\begin{itemize}
    \item Temporal Features: {USE_TEMPORAL}
    \item Scan-map Features: {USE_SCANMAP}
    \item Particle Features: {USE_PARTICLE}
    \item AMCL Pose Features: {USE_AMCL}
\end{itemize}

\section*{3. Metrics (Threshold = 0.5)}

\begin{itemize}
    \item Precision: {PREC_T05}
    \item Recall: {REC_T05}
    \item F1 Score: {F1_T05}
\end{itemize}

\section*{4. Best F1 Threshold}

\begin{itemize}
    \item Best Threshold: {BEST_THR}
    \item Precision: {PREC_BEST}
    \item Recall: {REC_BEST}
    \item F1 Score: {F1_BEST}
\end{itemize}

\section*{5. Plots}

\subsection*{Dataset Class Imbalance}
\begin{figure}[H]
\centering
\includegraphics[width=0.6\linewidth]{figures/dataset_imbalance.pdf}
\end{figure}

\subsection*{ROC Curve}
\begin{figure}[H]
\centering
\includegraphics[width=0.8\linewidth]{figures/roc_curve.pdf}
\end{figure}

\subsection*{Precision--Recall Curve}
\begin{figure}[H]
\centering
\includegraphics[width=0.8\linewidth]{figures/pr_curve.pdf}
\end{figure}

\subsection*{Confusion Matrix (Counts)}
\begin{figure}[H]
\centering
\includegraphics[width=0.75\linewidth]{figures/confusion_matrix_counts.pdf}
\end{figure}

\subsection*{Confusion Matrix (Normalized)}
\begin{figure}[H]
\centering
\includegraphics[width=0.75\linewidth]{figures/confusion_matrix_normalized.pdf}
\end{figure}

\subsection*{Threshold Curve — F1 vs Threshold}
\begin{figure}[H]
\centering
\includegraphics[width=0.8\linewidth]{figures/threshold_f1_curve.pdf}
\end{figure}

\subsection*{Feature Importances}
\begin{figure}[H]
\centering
\includegraphics[width=0.8\linewidth]{figures/feature_importances.pdf}
\end{figure}

\end{document}
"""


def generate_pdf_report(
    model_dir: Path,
    model_name: str,
    best_algo: str,
    n_trials: int,
    metrics_t05: dict,
    best_thr: float,
    best_metrics: dict,
    train_maps,
    eval_maps,
    odometry,
    notes,
    feature_flags: dict,
    svg_paths: dict,
):
    """
    Creates a thesis-friendly PDF report with all plots.
    Converts SVGs → PDF using CairoSVG.
    """

    report_dir = model_dir / "report"
    figs_dir = report_dir / "figures"

    report_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Convert SVGs → PDFs inside report/figures
    for key, path in svg_paths.items():
        if path.suffix.lower() != ".svg":
            continue

    # Convert SVGs → PDFs INSIDE report/figures WITHOUT COPYING PNG/SVG
    for key, path in svg_paths.items():
        if path.suffix.lower() != ".svg":
            continue

        target_pdf = figs_dir / (path.stem + ".pdf")

        cairosvg.svg2pdf(
            url=str(path),          # ← use original SVG directly
            write_to=str(target_pdf)
        )


    # Escape underscores for LaTeX
    model_name_tex = model_name.replace("_", r"\_")

    tex = TEMPLATE
    tex = tex.replace("{MODEL_NAME}", model_name_tex)
    tex = tex.replace("{DATE}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    tex = tex.replace("{BEST_ALGO}", best_algo)
    tex = tex.replace("{N_TRIALS}", str(n_trials))

    tex = tex.replace("{TRAIN_MAPS}", ", ".join(train_maps))
    tex = tex.replace("{EVAL_MAPS}", ", ".join(eval_maps))
    tex = tex.replace("{ODOMETRY}", odometry)
    tex = tex.replace("{NOTES}", notes)

    tex = tex.replace("{USE_TEMPORAL}", str(feature_flags["temporal"]))
    tex = tex.replace("{USE_SCANMAP}", str(feature_flags["scanmap"]))
    tex = tex.replace("{USE_PARTICLE}", str(feature_flags["particle"]))
    tex = tex.replace("{USE_AMCL}", str(feature_flags["amcl"]))

    tex = tex.replace("{PREC_T05}", f"{metrics_t05['precision']:.4f}")
    tex = tex.replace("{REC_T05}", f"{metrics_t05['recall']:.4f}")
    tex = tex.replace("{F1_T05}", f"{metrics_t05['f1']:.4f}")

    tex = tex.replace("{BEST_THR}", f"{best_thr:.3f}")
    tex = tex.replace("{PREC_BEST}", f"{best_metrics['precision']:.4f}")
    tex = tex.replace("{REC_BEST}", f"{best_metrics['recall']:.4f}")
    tex = tex.replace("{F1_BEST}", f"{best_metrics['f1']:.4f}")

    tex_path = report_dir / "model_report.tex"
    tex_path.write_text(tex)

    # Compile using XeLaTeX
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=report_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"✔ PDF report generated → {report_dir / 'model_report.pdf'}")
