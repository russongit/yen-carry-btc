"""Pre-freeze check of the DXY ex-yen formula in PREREGISTRATION.md section 4.

Data: Federal Reserve H.10 noon buying rates, weekly release dated 2024-08-12
(the source of FRED's DEX* series), and ICE DXY daily low/high from
Yahoo Finance (DX-Y.NYB). Values typed from those pages on 2026-10-19.

H.10 quote directions: EUR and GBP in USD per unit; JPY, CAD, SEK, CHF in
units per USD. This matches FRED: DEXUSEU, DEXUSUK (USD per unit) and
DEXJPUS, DEXCAUS, DEXSDUS, DEXSZUS (units per USD).
"""
import numpy as np

dates = ["2024-08-05", "2024-08-06", "2024-08-07", "2024-08-08", "2024-08-09"]
fx = {
    "EUR": np.array([1.0957, 1.0926, 1.0925, 1.0914, 1.0925]),          # DEXUSEU
    "JPY": np.array([143.9500, 145.1100, 147.4200, 147.0800, 146.4900]),  # DEXJPUS
    "GBP": np.array([1.2765, 1.2695, 1.2718, 1.2741, 1.2766]),          # DEXUSUK
    "CAD": np.array([1.3820, 1.3791, 1.3729, 1.3742, 1.3725]),          # DEXCAUS
    "SEK": np.array([10.5628, 10.5270, 10.4514, 10.5292, 10.5073]),     # DEXSDUS
    "CHF": np.array([0.8528, 0.8533, 0.8652, 0.8656, 0.8639]),          # DEXSZUS
}
dxy_low = np.array([102.16, 102.70, 102.92, 102.92, 103.03])
dxy_high = np.array([103.28, 103.23, 103.37, 103.55, 103.30])

# ICE exponents. Negative for pairs quoted as USD per unit (EUR, GBP).
W = {"EUR": -0.576, "JPY": 0.136, "GBP": -0.119, "CAD": 0.091, "SEK": 0.042, "CHF": 0.036}
CONST = 50.14348112


def dxy(weights):
    level = np.full(len(dates), CONST)
    for ccy, w in weights.items():
        level = level * fx[ccy] ** w
    return level


# Test 1: full DXY rebuilt from noon rates must sit inside each day's actual range.
recon = dxy(W)
print("Test 1: rebuilt DXY at New York noon vs actual daily range")
for d, r, lo, hi in zip(dates, recon, dxy_low, dxy_high):
    print(f"  {d}  rebuilt {r:8.3f}  range [{lo:.2f}, {hi:.2f}]  {'PASS' if lo <= r <= hi else 'FAIL'}")
assert np.all((recon >= dxy_low) & (recon <= dxy_high))

# Test 2: flipping the sign of any one exponent must break Test 1 (detects quote-direction errors).
print("Test 2: sign flip on each currency")
for ccy in W:
    flipped = dict(W)
    flipped[ccy] = -W[ccy]
    err = np.max(np.abs(dxy(flipped) / recon - 1))
    print(f"  flip {ccy}: max level error {err:7.1%}")
    assert not np.all((dxy(flipped) >= dxy_low) & (dxy(flipped) <= dxy_high))

# Test 3: ex-yen weights rescale to 1 and the return formula equals the log change of the index.
ex = {c: w for c, w in W.items() if c != "JPY"}
scale = sum(abs(w) for w in ex.values())
assert abs(scale - 0.864) < 1e-12
w_tilde = {c: w / scale for c, w in ex.items()}
assert abs(sum(abs(w) for w in w_tilde.values()) - 1) < 1e-12

r_formula = sum(w_tilde[c] * np.diff(np.log(fx[c])) for c in ex)
index_ex = np.ones(len(dates))
for c, w in w_tilde.items():
    index_ex = index_ex * fx[c] ** w
r_index = np.diff(np.log(index_ex))
assert np.allclose(r_formula, r_index)

print("Test 3: r_dxyx (positive = dollar stronger vs non-yen basket)")
for d, r in zip(dates[1:], r_formula):
    print(f"  {d}  r_dxyx {r:+.5f}")
print(f"  rescale factor 1/{scale:.3f}; rescaled weights: "
      + ", ".join(f"{c} {w:+.4f}" for c, w in w_tilde.items()))
print("ALL TESTS PASSED")
