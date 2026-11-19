"""gta6-trophy-rarity-forecast-01: forecast Exophase earn rates for GTA VI trophies.

Model (per case, deterministic, stdlib only, no network):
  Rates are earners / players_tracked.  We forecast both counts to the target,
  then take the ratio, with a logit-normal spread around it.

  1. Players N(t): the tracked-player count grows fast just after release and
     slows as the launch flood passes.  When the history shows growth we fit
     log N against log(1 + hours since release) (a power law) and extrapolate
     to the target, capped at 8x the last count.  When the history is FLAT
     (every read identical) the page is stale: Exophase refreshes on its own
     schedule, hours apart, so the next refresh reveals the launch flood that
     accumulated since the snapshot.  Then N(target) = N_last * M with
     M = ((h_target + 0.5) / (h_snap + 0.5)) ** 0.8, capped at 6x, where
     h_snap is the hours since release at the first read.  On launch night
     this gives M ~ 6, matching the refresh the house recorded (638 -> 3,868
     players); the flat page is a snapshot from before the release, not proof
     that nothing changed.

  2. Earners E_i(t) per trophy: fit a log-log slope of earners against
     log(1 + hours since release) over the case's reads, shrunk toward a
     per-kind prior slope (story trophies are earned fast by new players,
     grinds and the platinum barely move in week one).  In the flat/stale
     regime there is no slope to fit, so earners grow by a per-kind factor
     (early story trophies gain the most, grinds/platinum almost nothing).
     Earners never decrease; a trophy's projected rate is capped by a
     per-kind ceiling (platinum ~0.3%, 100% completion ~0.5%, grinds ~8%,
     story chapters ~65-80%).

  3. Spread: logit-normal, sigma = sqrt((0.5*sqrt(d))^2 + (0.45 + 0.10*d)^2
     (+ 0.25^2 when the history is flat)), d = horizon in days, plus a
     little extra for the platinum / 100%-completion trophies.  A 4.5-day
     horizon is far more uncertain than a 90-minute one, so the band is
     much wider there.  A trophy sitting at 100% in the last read keeps a
     median of 99.99 with a tight band: every tracked player earned it,
     including through the launch flood.

  Quantiles use the exact standard-normal quantiles of the case's levels.
  Two runs write byte-identical files: no randomness, no wall-clock use,
  inputs read in name order, outputs written with fixed formatting.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim.
Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output): {"trophies": [{"id": n, "q": [q05, q25,
q50, q75, q95]}, ...]}, one row per trophy, five finite numbers 0..100,
non-decreasing, at most 65,536 bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import. (This entry ships alone.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Standard levels and their exact normal quantiles; other levels go through
# the Acklam approximation in norm_ppf().
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_STD = {
    0.05: -1.6448536269514722,
    0.25: -0.6744897501960817,
    0.5: 0.0,
    0.75: 0.6744897501960817,
    0.95: 1.6448536269514722,
}


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p in Z_STD:
        return Z_STD[p]
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as the starter does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    """ISO 8601 (date or datetime, 'Z' or offset) -> aware UTC datetime."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------- kinds
# What sort of trophy this is, from its tier and description.  The kind
# sets the prior earner-growth slope, the flat-regime earner growth, the
# rate ceiling at the target, and a little extra spread.
GRIND_KEYS = (
    "250 stunts", "100 locations", "collectible card", "50 challenges",
    "every spot", "explore all", "win 20", "every location",
    "mission strands", "gold medal in all", "top fun", "purchase all",
    "register 10", "discover 100", "meditate", "earn $", "spend $",
)
STORY_MISSION_KEYS = (
    "complete breakfast", "complete the cartel", "complete child",
    "complete headliners", "complete vcia", "complete operation",
    "complete the final mission for", "complete an arcade",
)


def kind_of(trophy):
    if trophy.get("tier") == "platinum":
        return "platinum"
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    if "100% completion" in text:
        return "completion100"
    if "chapter 1" in text:
        return "story_ch1"
    if "chapter" in text:
        return "story_chapter"
    if "final mission" in text:
        return "story_end"
    if any(k in text for k in STORY_MISSION_KEYS):
        return "story_mission"
    if any(k in text for k in GRIND_KEYS):
        return "grind"
    return "activity"


# Per kind: prior log-log earner slope, flat-regime earner growth factor,
# rate ceiling at target (percent), extra logit spread.
KIND = {
    "platinum":      {"slope": 0.30, "flat_g": 0.00, "cap": 0.30, "extra": 0.30},
    "completion100": {"slope": 0.20, "flat_g": 0.00, "cap": 0.50, "extra": 0.30},
    "story_ch1":     {"slope": 1.60, "flat_g": 0.65, "cap": 80.0, "extra": 0.00},
    "story_chapter": {"slope": 1.30, "flat_g": 0.25, "cap": 65.0, "extra": 0.00},
    "story_end":     {"slope": 1.00, "flat_g": 0.05, "cap": 30.0, "extra": 0.10},
    "story_mission": {"slope": 1.10, "flat_g": 0.25, "cap": 55.0, "extra": 0.00},
    "grind":         {"slope": 0.60, "flat_g": 0.025, "cap": 8.0, "extra": 0.15},
    "activity":      {"slope": 1.00, "flat_g": 0.25, "cap": 65.0, "extra": 0.00},
}


def least_squares(xs, ys):
    """Slope and intercept of y on x; None when the fit is degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return b, my - b * mx


def forecast_case(case, name):
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or LEVELS
    zs = [norm_ppf(tau) for tau in levels]
    horizon_days = case["horizon_minutes"] / 1440.0

    release = parse_time(case["game"]["release_date"]) if case.get("game", {}).get("release_date") else None
    target = parse_time(case["target"])
    if release is None:
        release = parse_time(history[0]["at"]) if history else target

    def hours_since_release(dt):
        return max(0.0, (dt - release).total_seconds() / 3600.0)

    # --- history series -------------------------------------------------
    reads = []
    for h in history:
        try:
            reads.append((parse_time(h["at"]), h["players_tracked"], h["rates"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not reads:
        # No history at all: fall back to kind priors with a wide spread.
        rows = []
        for t in trophies:
            k = kind_of(t)
            med = {"platinum": 0.05, "completion100": 0.05, "story_ch1": 15.0,
                   "story_chapter": 5.0, "story_end": 1.0, "story_mission": 3.0,
                   "grind": 0.3, "activity": 2.0}[k]
            sig = 1.5
            rows.append({"id": t["id"], "q": [round4(clamp(inv_logit(logit(med) + z * sig))) for z in zs]})
        return {"trophies": rows}

    h_last = hours_since_release(reads[-1][0])
    h_target = hours_since_release(target)
    n_last = reads[-1][1]
    rates_last = reads[-1][2]

    # Last non-null rate per trophy (the last read can lack a trophy).
    last_rate = []
    for i in range(len(trophies)):
        r = None
        for _, _, rates in reversed(reads):
            if i < len(rates) and rates[i] is not None:
                r = rates[i]
                break
        last_rate.append(r)

    # --- is the history flat (a stale page)? -----------------------------
    ns = [n for _, n, _ in reads]
    flat = (max(ns) == min(ns)) or (max(ns) <= 1.02 * min(ns))
    if flat:
        h_snap = hours_since_release(reads[0][0])
        mult = ((h_target + 0.5) / (h_snap + 0.5)) ** 0.8
        n_target = n_last * min(6.0, max(1.0, mult))
    else:
        # Power-law fit of log N on log(1 + hours since release).
        xs = [math.log1p(hours_since_release(at)) for at, _, _ in reads]
        ys = [math.log(max(1, n)) for _, n, _ in reads]
        fit = least_squares(xs, ys)
        if fit is not None:
            b, a = fit
            b = min(3.0, max(0.0, b))
            n_target = math.exp(a + b * math.log1p(h_target))
        else:
            n_target = float(n_last)
        n_target = min(n_target, n_last * 8.0)
        n_target = max(n_target, float(n_last))
    n_target = max(n_target, 1.0)

    # --- spread ----------------------------------------------------------
    sigma_rw = 0.5 * math.sqrt(horizon_days)
    sigma_model = 0.45 + 0.10 * horizon_days
    sigma_flat = 0.25 if flat else 0.0

    # --- per-trophy forecast ---------------------------------------------
    rows = []
    for i, t in enumerate(trophies):
        k = kind_of(t)
        kp = KIND[k]
        r_last = last_rate[i]
        if r_last is None:
            r_last = 0.0
        e_last = r_last * n_last / 100.0

        if r_last >= 99.99:
            # Everybody tracked earned it, through the launch flood too.
            med, sig = 99.99, 0.12
        else:
            if flat:
                age_h = (target - reads[0][0]).total_seconds() / 3600.0
                g = kp["flat_g"] * math.sqrt(max(1.0, age_h) / 4.7)
                e_target = e_last * (1.0 + g)
            else:
                # Fit this trophy's earner slope, shrunk to the kind prior.
                xs, ys = [], []
                for at, n, rates in reads:
                    if i < len(rates) and rates[i] is not None:
                        e = rates[i] * n / 100.0
                        xs.append(math.log1p(hours_since_release(at)))
                        ys.append(math.log(max(e, 0.05)))
                fit = least_squares(xs, ys) if len(xs) >= 3 else None
                if fit is not None:
                    b_fit, _ = fit
                    b_fit = min(3.0, max(-0.5, b_fit))
                    w_fit = len(xs)
                    b = (w_fit * b_fit + 4.0 * kp["slope"]) / (w_fit + 4.0)
                else:
                    b = kp["slope"]
                e_target = e_last * ((1.0 + h_target) / (1.0 + h_last)) ** b
            # Earners never fall; a trophy with no earners yet can gain a few.
            e_target = max(e_target, e_last, n_target * 0.0001)
            e_target = min(e_target, n_target * kp["cap"] / 100.0)
            med = clamp(100.0 * e_target / n_target)
            sig = math.sqrt(sigma_rw ** 2 + sigma_model ** 2 + sigma_flat ** 2
                            + kp["extra"] ** 2)
        med_lg = logit(med)
        qs = [round4(clamp(inv_logit(med_lg + z * sig))) for z in zs]
        # Keep the row non-decreasing after rounding.
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
