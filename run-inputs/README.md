# Kaggle run inputs branch

Purpose: hold ONLY the inputs a Kaggle run needs (patches, diagnostics,
repro scripts). This branch is NOT product code.

Branch discipline (commander order 2026-10-24, standing):
- agent-v1 stays product-only. Diagnostic patches NEVER land on agent-v1.
- QA tests product ONLY by cloning agent-v1.
- NOTHING reaches Kaggle by browser upload anymore (upload dialogs retired
  2026-10-24 after approval-card blocks and a dataset-copy hash failure).
  The Kaggle session only FETCHES from GitHub.

Fetch procedure on Kaggle (product + inputs):
1. git clone -b agent-v1 --depth 1 https://github.com/nguyenminh20000/Litetorch- lt
2. cd lt && git fetch origin kaggle-run-inputs --depth 1
3. git show origin/kaggle-run-inputs:run-inputs/<file> > /tmp/<file>
   (or raw fetch from the branch; never an upload dialog, never a
   pre-existing Kaggle dataset copy)
4. Verify size AND sha256 against the QA verdict BEFORE applying.
   Mismatch means STOP and report. No fallback.

Layout: run-inputs/<run>/<file>. Each run gets its own directory.
