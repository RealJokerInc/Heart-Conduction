# Archived: monolithic textbook (pre-website-fork)

**Archived 2026-07-02.** These files are **superseded and stale** — do not edit, cite, or ship them.

## What these are

| File | What it was |
|------|-------------|
| `Bidomain_Textbook.html` | The old single-file (~12,300-line) textbook source. |
| `Bidomain_Textbook.pdf` | PDF render of that file (Feb 2026). |
| `Cardiac_Computational_Modeling.pdf` | Later PDF render of that file (Mar 2026), the former "official" PDF. |

## Why they were archived

The textbook forked into two copies that diverged. All rewriting after ~Feb 2026 happened in the **split website source** (`../../website/chapters/*.html`), not in this monolithic file. A chapter-by-chapter content comparison (2026-07-02) found:

- **Part I (Ch 1–6):** identical in both copies.
- **Part II (Ch 7–11):** identical except Ch 8, where the split has corrected appendix cross-refs (→ Appendix C) and this file still points at the old single-appendix numbering.
- **Part III (Ch 12–17):** this file still contains the **session-13b-DELETED** chapters — Ch 16 "Elliptic Solvers: The Schur Complement" and Ch 17 "Implementation Roadmap" — which describe a monolithic 2N×2N / FGMRES / AMG / Schur solver architecture **that does not exist in the engine** (Engine V1 uses decoupled N×N SPD solves). The split source replaced these with the correct 4-chapter Feynman rewrite (Ch 12–15).
- **Part IV (Ch 18–20):** identical in both copies.
- **Appendices:** this file has only **2** lettered appendices (A = Differential Equations, B = PyTorch). The split source has **4** (A Diff-Eqs / B Linear-Algebra / C Numerical-Analysis / D PyTorch) — ~10,000 words of Linear-Algebra + Numerical-Analysis content are **absent** from this file.

Every place the two copies differ, the split/website copy is newer; this file has no unique content worth keeping (its only "extra" material is the fictional Part III architecture that was deliberately deleted).

## What replaced them

- **Canonical source:** `../../website/chapters/*.html` (edit here) + `../../website/toc.json`.
- **Rendered whole-book HTML:** `../../Cardiac_Textbook_Website.html` (open in a browser).
- The `/textbook-edit` and `/textbook-compile` skills and the tracking docs (README / INDEX / KNOWLEDGE / IDEALOG / CHANGELOG) were repointed to the website source on 2026-07-02.

## Known gap (separate to-do)

The PDF build script `html_to_pdf_v3.py` referenced by `/textbook-compile` is **missing from the repo**, so there is currently no working PDF build for either copy. A correct PDF must be regenerated from the website source once that pipeline is rebuilt.
