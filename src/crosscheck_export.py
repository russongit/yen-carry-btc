"""Export the H1a estimation sample for an independent OLS cross-check (DEVIATIONS.md, 2026-10-20).

Writes outputs/crosscheck/h1a_sample.csv (date, dependent variable, regressors; no constant) and
outputs/crosscheck/h1a_statsmodels_reference.csv (coefficients, classical and HC1 standard errors,
fitted on that CSV as read back). Stops if the CSV's coefficients differ from the H1a full-sample
coefficients in the committed first run (outputs/tables/regressions.csv).
Verification only: HC1 errors are not the pre-registered Newey-West errors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_projections as lp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "crosscheck"


def main() -> None:
    s = lp.build_samples("full")["H1a"]
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.concat([s.y.rename(s.spec.y), s.X], axis=1)
    data.index.name = "date"
    data.to_csv(OUT / "h1a_sample.csv", date_format="%Y-%m-%d", float_format="%.17g")

    back = pd.read_csv(OUT / "h1a_sample.csv", index_col="date")
    X = sm.add_constant(back[list(s.spec.x)])
    classical = sm.OLS(back[s.spec.y], X).fit()
    hc1 = sm.OLS(back[s.spec.y], X).fit(cov_type="HC1")

    run = pd.read_csv(ROOT / "outputs" / "tables" / "regressions.csv")
    committed = run[(run["spec"] == "H1a") & (run["subsample"] == "full")].set_index("term")["coef"]
    np.testing.assert_allclose(classical.params[committed.index], committed, rtol=1e-9, atol=1e-15)

    ref = pd.DataFrame({"term": classical.params.index, "coef": classical.params.values,
                        "se_classical": classical.bse.values, "se_hc1": hc1.bse.values})
    ref.to_csv(OUT / "h1a_statsmodels_reference.csv", index=False, float_format="%.10g")
    print(f"{len(back)} rows, {len(s.spec.x)} regressors + constant; coefficients match the committed run")
    print(ref.to_string(index=False))


if __name__ == "__main__":
    main()
