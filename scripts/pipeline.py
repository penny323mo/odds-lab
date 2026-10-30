#!/usr/bin/env python3
"""
pipeline.py - Orchestrates the entire Odds Lab v4 data pipeline:
1. Fetches or loads Macau Slot live data (next 3 days, top 5 + UEFA cups)
2. Fetches or loads The Odds API data (guarded daily run, 500/mo budget)
3. Loads football-data historical matches & upcoming sharp AH fixtures
4. Fits Dixon-Coles model on top-5 leagues with sample size attenuation for promotees
5. Cross-source matches fixtures with normalized team names (strictly avoiding collisions)
6. Calculates consensus: Pinnacle-weighted (0.6 Pinnacle + 0.4 other books median)
7. Fits Poisson/DC lambda goal expectations to consensus 1X2 + totals
8. Sharp Asian Handicap interpolation to Macau quarter lines (via score matrix calibration)
9. Blends consensus with Dixon-Coles model for 1X2, Over/Under, AND Asian Handicap
10. Strict "值博 sanity" & Pinnacle Anchor:
    - Never badge as 值博 if Macau odds <= Pinnacle odds on that selection
    - Never badge as 值博 if raw EV > 15% (mark as 數據不足 instead of loophole clamp)
11. Builds bookmaker discrepancy matrix and best price highlights (prioritizing Pinnacle in booksList)
12. Records and updates snapshots in data/snapshots/ (with valid match keys and sparklines)
13. Updates track record in data/track_record.json (freezing CLV at kickoff, strictly date-gated results)
14. Schema validation and error handling:
    - On Macau failure, preserves previous good data.js without corrupting with unpriced samples
    - On Odds API error, surfaces overall warning in health status
"""

import csv
import json
import math
import os
import sys
import re
from datetime import datetime, timezone, timedelta

# Import helper modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in (CURRENT_DIR, PARENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from team_names import TEAM_NAMES, get_team_cantonese, list_missing_cantonese, teams_match, normalize_team_name
from fetch_macau import fetch_macau_data, COMPETITION_MAP
from fetch_odds_api import fetch_odds_api
from math_engine import (
    remove_margin_multiplicative, remove_margin_power, remove_margin,
    solve_poisson_lambda_for_ou25, convert_ou_line_probs,
    calc_ou_expected_return, settle_ah_single, settle_ou_single,
    blend_probabilities, classify_ev, calc_kelly_stake,
    calc_dc_matrix, fit_lambdas_from_consensus, calibrate_lambdas_to_sharp_ah,
    derive_ah_fair_from_matrix, blend_ah_distributions
)
from snapshots_tracker import save_snapshot, load_snapshots_history
from record_tracker import update_track_record, load_track_record

ROOT_DIR = PARENT_DIR
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(ROOT_DIR, "raw")

LEAGUES_TOP5 = [
    {"id": "E0", "key": "epl", "name": "英超", "fullName": "英格蘭超級聯賽"},
    {"id": "SP1", "key": "laliga", "name": "西甲", "fullName": "西班牙甲組聯賽"},
    {"id": "I1", "key": "seriea", "name": "意甲", "fullName": "意大利甲組聯賽"},
    {"id": "D1", "key": "bundesliga", "name": "德甲", "fullName": "德國甲組聯賽"},
    {"id": "F1", "key": "ligue1", "name": "法甲", "fullName": "法國甲組聯賽"},
]

def load_previous_fixtures_from_data_js(data_js_path):
    """Extracts last known good macauFixtures from data/data.js."""
    if not os.path.exists(data_js_path):
        return []
    try:
        with open(data_js_path, "r", encoding="utf-8") as f:
            content = f.read()
        m = re.search(r"const\s+ODDS_DATA\s*=\s*(\{.*?\});\s*(?:if|$)", content, re.DOTALL)
        if m:
            data = json.loads(m.group(1))
            return data.get("macauFixtures", [])
    except Exception as e:
        print(f"[Pipeline] Warning: failed to load previous fixtures from data.js: {e}")
    return []

def validate_macau_response(macau_result):
    """Schema check for Macau Slot API response: must have events with actual odds."""
    if not isinstance(macau_result, dict):
        return False, "Macau result is not a dictionary"
    events = macau_result.get("events")
    if events is None or not isinstance(events, list):
        return False, "Macau result missing 'events' list"
    if len(events) == 0:
        return True, "OK (no events scheduled)"

    required_keys = ["ev_id", "start_time", "league_code", "home_en", "away_en"]
    for idx, ev in enumerate(events):
        for k in required_keys:
            if k not in ev or not ev[k]:
                return False, f"Event at index {idx} missing required key '{k}'"

    # Require that at least 1 event has priced markets (not unpriced target_events)
    has_odds = any(ev.get("odds_1x2") or ev.get("main_ah") for ev in events)
    if not has_odds:
        return False, "Macau events contain no odds markets (unpriced events)"

    return True, "OK"

def validate_odds_api_response(api_matches):
    """Schema check for The Odds API response."""
    if not isinstance(api_matches, list):
        return False, "The Odds API matches must be a list"
    for idx, m in enumerate(api_matches):
        if not isinstance(m, dict):
            return False, f"Match at index {idx} is not a dictionary"
        for k in ["id", "sport_key", "commence_time", "home_team", "away_team"]:
            if k not in m:
                return False, f"Match at index {idx} missing required key '{k}'"
    return True, "OK"

def load_football_data_ah_fixtures(csv_path):
    """
    Loads upcoming fixtures with Pinnacle and Bet365 Asian Handicap odds from football-data fixtures.csv.
    Returns dict: (home, away) -> {"line": float, "h": float, "a": float, "book": str}
    """
    if not os.path.exists(csv_path):
        return {}
    ah_map = {}
    try:
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = (row.get("HomeTeam") or "").strip()
                a = (row.get("AwayTeam") or "").strip()
                if not h or not a:
                    continue
                line_str = row.get("AHh") or row.get("AHCh")
                if not line_str:
                    continue
                try:
                    line = float(line_str)
                except ValueError:
                    continue

                pcahh = row.get("PCAHH") or row.get("PAHH")
                pcaha = row.get("PCAHA") or row.get("PAHA")
                b365ahh = row.get("B365CAHH") or row.get("B365AHH")
                b365aha = row.get("B365CAHA") or row.get("B365AHA")
                avgahh = row.get("AvgCAHH") or row.get("AvgAHH")
                avgaha = row.get("AvgCAHA") or row.get("AvgAHA")

                h_odds = None
                a_odds = None
                book = None

                if pcahh and pcaha:
                    try:
                        h_val, a_val = float(pcahh), float(pcaha)
                        if h_val > 1.0 and a_val > 1.0:
                            h_odds, a_odds, book = h_val, a_val, "Pinnacle"
                    except ValueError:
                        pass
                if not h_odds and b365ahh and b365aha:
                    try:
                        h_val, a_val = float(b365ahh), float(b365aha)
                        if h_val > 1.0 and a_val > 1.0:
                            h_odds, a_odds, book = h_val, a_val, "Bet365"
                    except ValueError:
                        pass
                if not h_odds and avgahh and avgaha:
                    try:
                        h_val, a_val = float(avgahh), float(avgaha)
                        if h_val > 1.0 and a_val > 1.0:
                            h_odds, a_odds, book = h_val, a_val, "MarketAvg"
                    except ValueError:
                        pass

                if h_odds and a_odds:
                    ah_map[(h, a)] = {"line": line, "h": h_odds, "a": a_odds, "book": book}
    except Exception as e:
        print(f"[Pipeline] Warning reading fixtures.csv AH: {e}")
    return ah_map

def format_day_label(date_str, now_date):
    """Returns friendly Cantonese date grouping label."""
    try:
        match_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        diff_days = (match_date - now_date).days
        if diff_days == 0:
            return "今日 (Today)"
        elif diff_days == 1:
            return "聽日 (Tomorrow)"
        elif diff_days == 2:
            return "後日 (Day After)"
        else:
            return f"{match_date.strftime('%m月%d日')} ({diff_days}日後)"
    except Exception:
        return date_str

def fit_dixon_coles_python(matches, teams, as_of_ts, half_life_days=365):
    """
    Fits Dixon-Coles parameters (attack alpha, defence beta, home advantage gamma)
    using iterative re-weighted least squares / Newton-Raphson approximation.
    """
    n_teams = len(teams)
    if n_teams == 0 or len(matches) == 0:
        return {}, {}, 0.25

    xi = math.log(2.0) / (half_life_days * 86400.0)

    alpha = {t["id"]: 0.0 for t in teams}
    beta = {t["id"]: 0.0 for t in teams}
    gamma = 0.25

    for iteration in range(15):
        grad_alpha = {t["id"]: 0.0 for t in teams}
        grad_beta = {t["id"]: 0.0 for t in teams}
        hess_alpha = {t["id"]: 1e-4 for t in teams}
        hess_beta = {t["id"]: 1e-4 for t in teams}
        grad_gamma = 0.0
        hess_gamma = 1e-4

        for m in matches:
            date_iso, h_id, a_id, fthg, ftag = m[0], m[1], m[2], m[3], m[4]
            m_dt = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            m_ts = m_dt.timestamp()

            if m_ts >= as_of_ts:
                continue

            dt_sec = as_of_ts - m_ts
            w = math.exp(-xi * dt_sec) if dt_sec > 0 else 1.0

            log_lh = alpha[h_id] - beta[a_id] + gamma
            log_la = alpha[a_id] - beta[h_id]

            lh = math.exp(max(-4.0, min(3.0, log_lh)))
            la = math.exp(max(-4.0, min(3.0, log_la)))

            gh = w * (fthg - lh)
            ga = w * (ftag - la)

            grad_alpha[h_id] += gh
            grad_beta[a_id] -= gh
            grad_gamma += gh
            hess_alpha[h_id] += w * lh
            hess_beta[a_id] += w * lh
            hess_gamma += w * lh

            grad_alpha[a_id] += ga
            grad_beta[h_id] -= ga
            hess_alpha[a_id] += w * la
            hess_beta[h_id] += w * la

        # Newton updates
        for tid in alpha:
            alpha[tid] += grad_alpha[tid] / (hess_alpha[tid] + 1.0)
            beta[tid] += grad_beta[tid] / (hess_beta[tid] + 1.0)
        gamma += grad_gamma / (hess_gamma + 1.0)
        gamma = max(0.05, min(0.60, gamma))

        # Constraint: sum(alpha) == 0
        mean_alpha = sum(alpha.values()) / n_teams
        for tid in alpha:
            alpha[tid] -= mean_alpha

    return alpha, beta, gamma

def derive_probs_from_matrix(matrix):
    """Derives 1X2, Over 2.5, and Expected Goals from matrix."""
    max_goals = len(matrix) - 1
    p_h = 0.0
    p_d = 0.0
    p_a = 0.0
    p_o25 = 0.0
    xg_h = 0.0
    xg_a = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = matrix[h][a]
            xg_h += h * p
            xg_a += a * p
            if h > a:
                p_h += p
            elif h == a:
                p_d += p
            else:
                p_a += p
            if h + a > 2.5:
                p_o25 += p

    return {
        "p_h": p_h, "p_d": p_d, "p_a": p_a,
        "p_o25": p_o25, "p_u25": 1.0 - p_o25,
        "xg_h": round(xg_h, 2), "xg_a": round(xg_a, 2)
    }

def run_pipeline(force_odds_api=False):
    """Main execution method for Odds Lab v4 pipeline."""
    now_utc = datetime.now(timezone.utc)
    now_macau = datetime.now(timezone(timedelta(hours=8)))
    now_iso_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    health_status = {
        "status": "healthy",
        "checked_at": now_iso_utc,
        "issues": [],
        "sources": {}
    }

    # Step 1: Fetch Macau Slot Data
    print("[Pipeline] 1/7 Fetching live Macau Slot coming events & markets...")
    macau_result = {"events": []}
    macau_fetch_failed = False
    try:
        macau_result = fetch_macau_data(sample_save_dir=os.path.join(DATA_DIR, "raw_samples"))
        valid_macau, err_macau = validate_macau_response(macau_result)
        if not valid_macau:
            print(f"[Pipeline] ERROR: Macau Slot schema validation failed: {err_macau}", file=sys.stderr)
            health_status["status"] = "error"
            health_status["issues"].append(f"Macau schema error: {err_macau}")
            health_status["sources"]["macau_slot"] = {
                "status": "error",
                "last_fetch": now_iso_utc,
                "events_count": len(macau_result.get("events", [])),
                "message": f"Schema validation failed: {err_macau}"
            }
            macau_fetch_failed = True
        else:
            health_status["sources"]["macau_slot"] = {
                "status": "healthy",
                "last_fetch": now_iso_utc,
                "events_count": len(macau_result.get("events", [])),
                "message": f"Fetched {len(macau_result.get('events', []))} events"
            }
    except Exception as e:
        print(f"[Pipeline] ERROR in Macau Slot fetch: {e}", file=sys.stderr)
        health_status["status"] = "error"
        health_status["issues"].append(f"Macau Slot fetch failure: {e}")
        health_status["sources"]["macau_slot"] = {
            "status": "error",
            "last_fetch": now_iso_utc,
            "events_count": 0,
            "message": str(e)
        }
        macau_fetch_failed = True

    # If Macau fetch failed, NEVER overwrite with unpriced raw_samples!
    if macau_fetch_failed:
        print("[Pipeline] Macau fetch failed. Restoring last-good fixtures from previous data.js...")
        prev_fixtures = load_previous_fixtures_from_data_js(os.path.join(DATA_DIR, "data.js"))
        if prev_fixtures:
            print(f"[Pipeline] Successfully restored {len(prev_fixtures)} previous fixtures.")
            # Write health status and return with preserved fixtures
            health_file = os.path.join(DATA_DIR, "health.json")
            with open(health_file, "w", encoding="utf-8") as f:
                json.dump(health_status, f, ensure_ascii=False, indent=2)
            track_rec = load_track_record(filepath=os.path.join(DATA_DIR, "track_record.json"))
            return {
                "generatedAt": now_iso_utc,
                "health": health_status,
                "macauFixtures": prev_fixtures,
                "trackRecord": track_rec,
                "sourcesStatus": health_status["sources"]
            }

    # Step 2: Fetch The Odds API Data (Guarded)
    print("[Pipeline] 2/7 Querying The Odds API (daily guard)...")
    odds_api_res = fetch_odds_api(data_dir=DATA_DIR, force=force_odds_api)
    api_matches = odds_api_res.get("matches", [])
    api_state = odds_api_res.get("state", {})

    valid_odds, err_odds = validate_odds_api_response(api_matches)
    if not valid_odds and len(api_matches) > 0:
        health_status["status"] = "error"
        health_status["issues"].append(f"Odds API schema error: {err_odds}")

    # Surface Odds API warning in overall health status if API had an error
    is_api_err = (api_state.get("status") == "error") or (odds_api_res.get("status") == "error")
    if is_api_err and health_status["status"] == "healthy":
        health_status["status"] = "warning"
        health_status["issues"].append(f"The Odds API warning: {api_state.get('message', 'API request failed')}")

    health_status["sources"]["the_odds_api"] = {
        "status": "healthy" if api_state.get("status") in ("success", "skipped") else "warning",
        "last_fetch": api_state.get("last_run_timestamp", now_iso_utc),
        "requests_remaining": api_state.get("requests_remaining"),
        "requests_used": api_state.get("requests_used"),
        "events_count": len(api_matches),
        "message": api_state.get("message", "")
    }

    # Step 3: Load historical match data & fit Dixon-Coles models for Top 5 leagues
    print("[Pipeline] 3/7 Loading historical data and fitting Dixon-Coles models...")
    dc_models = {}
    league_teams_map = {}
    finished_matches_for_tracker = []

    for lg in LEAGUES_TOP5:
        div = lg["id"]
        matches_list = []
        teams_map = {}
        teams_list = []

        # Load 2526 and 2627
        for season in ["2526", "2627"]:
            csv_path = os.path.join(RAW_DIR, f"{div}_{season}.csv")
            if not os.path.exists(csv_path):
                continue
            with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as fp:
                reader = csv.DictReader(fp)
                for r in reader:
                    h = (r.get("HomeTeam") or "").strip()
                    a = (r.get("AwayTeam") or "").strip()
                    if not h or not a:
                        continue
                    try:
                        fthg = int(r.get("FTHG", 0))
                        ftag = int(r.get("FTAG", 0))
                    except (ValueError, TypeError):
                        continue

                    # Register team
                    for t_name in (h, a):
                        if t_name not in teams_map:
                            t_idx = len(teams_list)
                            teams_map[t_name] = t_idx
                            teams_list.append({"id": t_idx, "name": t_name})

                    h_idx = teams_map[h]
                    a_idx = teams_map[a]

                    # Parse date
                    date_raw = r.get("Date", "")
                    parts = date_raw.split("/")
                    date_iso = f"{int(parts[2]):04d}-{int(parts[1]):02d}-{int(parts[0]):02d}" if len(parts) == 3 else date_raw

                    matches_list.append([date_iso, h_idx, a_idx, fthg, ftag])
                    # Strictly store finished matches with calendar date (P0-1)
                    finished_matches_for_tracker.append({
                        "date": date_iso,
                        "home": h,
                        "away": a,
                        "fthg": fthg,
                        "ftag": ftag
                    })

        league_teams_map[div] = (teams_map, teams_list)
        as_of_ts = now_utc.timestamp()
        alpha, beta, gamma = fit_dixon_coles_python(matches_list, teams_list, as_of_ts)

        # Track number of matches per team in the training set
        team_match_counts = {t["id"]: 0 for t in teams_list}
        for m in matches_list:
            team_match_counts[m[1]] += 1
            team_match_counts[m[2]] += 1

        dc_models[div] = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "teams_map": teams_map,
            "match_counts": team_match_counts
        }

    # Load upcoming sharp AH data from fixtures.csv (P2-20)
    sharp_ah_fixtures = load_football_data_ah_fixtures(os.path.join(RAW_DIR, "fixtures.csv"))

    health_status["sources"]["football_data"] = {
        "status": "healthy",
        "finished_matches_count": len(finished_matches_for_tracker),
        "sharp_ah_count": len(sharp_ah_fixtures)
    }

    # Step 4: Cross-source match mapping & Consensus generation
    print("[Pipeline] 4/7 Cross-source matching and calculating consensus...")
    macau_events = macau_result.get("events", [])
    processed_fixtures = []
    unmatched_events = []
    missing_team_names = []  # English names with no Cantonese mapping
    all_value_bets = []

    # Load snapshot history for sparklines
    snapshot_histories = load_snapshots_history(base_data_dir=DATA_DIR)

    for mev in macau_events:
        m_dt = datetime.fromisoformat(mev["kickoff_iso"])
        m_ts = m_dt.timestamp()
        h_m = mev["home_en"]
        a_m = mev["away_en"]
        lg_code = mev["league_code"]
        ev_id = mev["ev_id"]

        # Ensure Cantonese team names prefer CJK; fall back to English and collect misses (P1-4)
        h_zh = mev.get("home_zh") or ""
        a_zh = mev.get("away_zh") or ""
        if not re.search(r"[\u4e00-\u9fff]", h_zh) or "?" in h_zh:
            h_zh = get_team_cantonese(h_m, collect_missing=missing_team_names)
        if not re.search(r"[\u4e00-\u9fff]", a_zh) or "?" in a_zh:
            a_zh = get_team_cantonese(a_m, collect_missing=missing_team_names)
        mev["home_zh"] = h_zh
        mev["away_zh"] = a_zh

        match_key = f"{lg_code}_{h_m}_{a_m}"

        # Match against The Odds API (strict collision-free matcher)
        found_odds_match = None
        for om in api_matches:
            try:
                o_dt = datetime.fromisoformat(om["commence_time"].replace("Z", "+00:00"))
                o_ts = o_dt.timestamp()
                if abs(m_ts - o_ts) <= 3 * 3600:
                    if teams_match(h_m, om["home_team"]) and teams_match(a_m, om["away_team"]):
                        found_odds_match = om
                        break
            except Exception:
                continue

        # Bookmaker Odds aggregation & Consensus
        books_list = []
        pinnacle_book = None
        all_1x2_probs = {"h": [], "d": [], "a": []}
        other_1x2_probs = {"h": [], "d": [], "a": []}
        all_ou25_probs = {"over": [], "under": []}
        pinnacle_totals = {}

        if found_odds_match:
            for bm in found_odds_match.get("bookmakers", []):
                b_key = bm.get("key")
                b_title = bm.get("title")

                # Extract 1X2
                b_1x2 = None
                for mkt in bm.get("markets", []):
                    if mkt.get("key") == "h2h":
                        rates = {}
                        for oc in mkt.get("outcomes", []):
                            name = oc.get("name")
                            price = oc.get("price")
                            if price and price > 1.0:
                                if name == "Draw":
                                    rates["d"] = price
                                elif teams_match(h_m, name) or name == found_odds_match.get("home_team"):
                                    rates["h"] = price
                                elif teams_match(a_m, name) or name == found_odds_match.get("away_team"):
                                    rates["a"] = price
                        if "h" in rates and "d" in rates and "a" in rates:
                            devig_probs, margin = remove_margin([rates["h"], rates["d"], rates["a"]], method="multiplicative")
                            b_1x2 = {
                                "h": rates["h"], "d": rates["d"], "a": rates["a"],
                                "margin": round(margin * 100.0, 1),
                                "prob_h": devig_probs[0], "prob_d": devig_probs[1], "prob_a": devig_probs[2]
                            }
                            all_1x2_probs["h"].append(devig_probs[0])
                            all_1x2_probs["d"].append(devig_probs[1])
                            all_1x2_probs["a"].append(devig_probs[2])
                            # P2-10: Collect non-Pinnacle probabilities separately for median
                            if b_key != "pinnacle":
                                other_1x2_probs["h"].append(devig_probs[0])
                                other_1x2_probs["d"].append(devig_probs[1])
                                other_1x2_probs["a"].append(devig_probs[2])

                    elif mkt.get("key") == "totals":
                        rates = {}
                        point = None
                        for oc in mkt.get("outcomes", []):
                            name = oc.get("name")
                            point = oc.get("point")
                            price = oc.get("price")
                            if price and price > 1.0:
                                if name == "Over":
                                    rates["over"] = price
                                elif name == "Under":
                                    rates["under"] = price
                        if "over" in rates and "under" in rates:
                            # Use power de-vig (Shin-lite) for two-way totals markets (P2-11)
                            devig_probs, margin = remove_margin([rates["over"], rates["under"]], method="power")
                            if point == 2.5:
                                all_ou25_probs["over"].append(devig_probs[0])
                                all_ou25_probs["under"].append(devig_probs[1])
                            if b_key == "pinnacle" and point is not None:
                                pinnacle_totals[float(point)] = {
                                    "over": rates["over"], "under": rates["under"],
                                    "prob_over": devig_probs[0], "prob_under": devig_probs[1]
                                }

                if b_1x2:
                    book_entry = {
                        "key": b_key,
                        "title": b_title,
                        "1x2": b_1x2
                    }
                    books_list.append(book_entry)
                    if b_key == "pinnacle":
                        pinnacle_book = book_entry
        else:
            unmatched_events.append({
                "ev_id": ev_id,
                "kickoff": mev["start_time"],
                "league": lg_code,
                "home": h_m,
                "away": a_m
            })

        # Consensus 1X2 Probabilities (Pinnacle-weighted 0.6 / 0.4 other books median)
        if pinnacle_book and pinnacle_book.get("1x2"):
            pin_1x2 = pinnacle_book["1x2"]
            p_pin_h = pin_1x2["prob_h"]
            p_pin_d = pin_1x2["prob_d"]
            p_pin_a = pin_1x2["prob_a"]
            # Median of other (non-Pinnacle) books (P2-10)
            med_pool_h = other_1x2_probs["h"] if other_1x2_probs["h"] else [p_pin_h]
            med_pool_d = other_1x2_probs["d"] if other_1x2_probs["d"] else [p_pin_d]
            med_pool_a = other_1x2_probs["a"] if other_1x2_probs["a"] else [p_pin_a]
            med_h = sorted(med_pool_h)[len(med_pool_h) // 2]
            med_d = sorted(med_pool_d)[len(med_pool_d) // 2]
            med_a = sorted(med_pool_a)[len(med_pool_a) // 2]
            raw_c_h = 0.6 * p_pin_h + 0.4 * med_h
            raw_c_d = 0.6 * p_pin_d + 0.4 * med_d
            raw_c_a = 0.6 * p_pin_a + 0.4 * med_a
        elif all_1x2_probs["h"]:
            raw_c_h = sorted(all_1x2_probs["h"])[len(all_1x2_probs["h"]) // 2]
            raw_c_d = sorted(all_1x2_probs["d"])[len(all_1x2_probs["d"]) // 2]
            raw_c_a = sorted(all_1x2_probs["a"])[len(all_1x2_probs["a"]) // 2]
        elif mev.get("odds_1x2"):
            m1x2 = mev["odds_1x2"]
            dev, _ = remove_margin([m1x2["h"], m1x2["d"], m1x2["a"]], method="multiplicative")
            raw_c_h, raw_c_d, raw_c_a = dev[0], dev[1], dev[2]
        else:
            raw_c_h, raw_c_d, raw_c_a = 0.38, 0.28, 0.34

        tot_c = raw_c_h + raw_c_d + raw_c_a
        cons_p_h = raw_c_h / tot_c
        cons_p_d = raw_c_d / tot_c
        cons_p_a = raw_c_a / tot_c

        # Consensus Over 2.5
        if all_ou25_probs["over"]:
            cons_p_o25 = sorted(all_ou25_probs["over"])[len(all_ou25_probs["over"]) // 2]
        elif mev.get("main_ou"):
            m_ou = mev["main_ou"]
            line = m_ou["line"]
            dev, _ = remove_margin([m_ou["over"], m_ou["under"]], method="power")
            cons_p_o25 = dev[0] if abs(line - 2.5) < 0.1 else 0.50
        else:
            cons_p_o25 = 0.50
        cons_p_u25 = 1.0 - cons_p_o25

        # Dixon-Coles pre-match probabilities for top-5 league
        dc_probs = None
        dc_matrix = None
        dc_weight = 0.30
        if lg_code in dc_models:
            model = dc_models[lg_code]
            t_map = model["teams_map"]
            alpha = model["alpha"]
            beta = model["beta"]
            gamma = model["gamma"]
            counts = model.get("match_counts", {})

            h_id = None
            a_id = None
            for name, tid in t_map.items():
                if teams_match(h_m, name):
                    h_id = tid
                if teams_match(a_m, name):
                    a_id = tid

            if h_id is not None and a_id is not None:
                lh = math.exp(alpha.get(h_id, 0.0) - beta.get(a_id, 0.0) + gamma)
                la = math.exp(alpha.get(a_id, 0.0) - beta.get(h_id, 0.0))
                dc_matrix = calc_dc_matrix(lh, la)
                dc_probs = derive_probs_from_matrix(dc_matrix)

                # Promotee / Small-Sample Dixon-Coles Attenuation:
                # If either club has < 20 matches in top-flight training data,
                # smoothly attenuate DC weight to avoid noisy promotee distortions (Hull City bug fix).
                nh = counts.get(h_id, 0)
                na = counts.get(a_id, 0)
                min_n = min(nh, na)
                if min_n < 20:
                    dc_weight = 0.30 * max(0.0, (min_n - 5) / 15.0)

        # Blend Consensus with Dixon-Coles
        cons_weight = 1.0 - dc_weight
        if dc_probs:
            blended_1x2 = blend_probabilities(
                [cons_p_h, cons_p_d, cons_p_a],
                [dc_probs["p_h"], dc_probs["p_d"], dc_probs["p_a"]],
                weight_consensus=cons_weight
            )
            fair_p_h, fair_p_d, fair_p_a = blended_1x2[0], blended_1x2[1], blended_1x2[2]
            fair_p_o25 = cons_weight * cons_p_o25 + dc_weight * dc_probs["p_o25"]
            xg_home = dc_probs["xg_h"]
            xg_away = dc_probs["xg_a"]
        else:
            fair_p_h, fair_p_d, fair_p_a = cons_p_h, cons_p_d, cons_p_a
            fair_p_o25 = cons_p_o25
            xg_home = round(1.2 + 0.6 * (fair_p_h - fair_p_a), 2)
            xg_away = round(1.0 - 0.5 * (fair_p_h - fair_p_a), 2)

        fair_p_u25 = 1.0 - fair_p_o25
        fair_odds_h = round(1.0 / fair_p_h, 2)
        fair_odds_d = round(1.0 / fair_p_d, 2)
        fair_odds_a = round(1.0 / fair_p_a, 2)

        # Step 5: Classify Macau Slot Markets
        match_opportunities = []

        # 1. 1X2 Classification (with Pinnacle Anchor)
        macau_1x2_classified = None
        if mev.get("odds_1x2"):
            m_h = mev["odds_1x2"]["h"]
            m_d = mev["odds_1x2"]["d"]
            m_a = mev["odds_1x2"]["a"]

            ev_h = (fair_p_h * m_h) - 1.0
            ev_d = (fair_p_d * m_d) - 1.0
            ev_a = (fair_p_a * m_a) - 1.0

            best_1x2_ev = max(ev_h, ev_d, ev_a)
            best_sel = "H" if best_1x2_ev == ev_h else ("D" if best_1x2_ev == ev_d else "A")
            sel_name = f"主勝 ({mev['home_zh']})" if best_sel == "H" else ("和局" if best_sel == "D" else f"客勝 ({mev['away_zh']})")
            sel_odds = m_h if best_sel == "H" else (m_d if best_sel == "D" else m_a)
            sel_fair = fair_odds_h if best_sel == "H" else (fair_odds_d if best_sel == "D" else fair_odds_a)
            sel_p = fair_p_h if best_sel == "H" else (fair_p_d if best_sel == "D" else fair_p_a)

            # Pinnacle Anchor Rule:
            # If Pinnacle is present, Macau odds MUST beat Pinnacle odds on that outcome!
            # Never badge as 值博 if Macau is offering worse price than Pinnacle.
            is_pin_anchored = True
            if pinnacle_book and pinnacle_book.get("1x2"):
                pin_1x2 = pinnacle_book["1x2"]
                pin_rates = {"H": pin_1x2["h"], "D": pin_1x2["d"], "A": pin_1x2["a"]}
                pin_p = {"H": pin_1x2["prob_h"], "D": pin_1x2["prob_d"], "A": pin_1x2["prob_a"]}[best_sel]
                pin_o = pin_rates[best_sel]
                ev_vs_pin = (pin_p * sel_odds) - 1.0
                if sel_odds <= pin_o or ev_vs_pin <= 0.0:
                    is_pin_anchored = False
            else:
                raw_cons_p = {"H": cons_p_h, "D": cons_p_d, "A": cons_p_a}[best_sel]
                if (raw_cons_p * sel_odds - 1.0) <= 0.0:
                    is_pin_anchored = False

            # Classification with clamp/loophole elimination
            if best_1x2_ev > 0.15 or best_1x2_ev < -0.25:
                badge_1x2 = {
                    "label": "數據不足",
                    "tier": "insufficient",
                    "badgeClass": "badge-val-neutral",
                    "ev_pct": round(best_1x2_ev * 100.0, 1)
                }
            elif best_1x2_ev > 0.03 and is_pin_anchored:
                badge_1x2 = classify_ev(best_1x2_ev)
            elif best_1x2_ev > 0.03 and not is_pin_anchored:
                badge_1x2 = {
                    "label": "一般",
                    "tier": "neutral",
                    "badgeClass": "badge-val-neutral",
                    "ev_pct": round(best_1x2_ev * 100.0, 1)
                }
            else:
                badge_1x2 = classify_ev(best_1x2_ev)

            k_frac, k_units = calc_kelly_stake(sel_p, sel_odds)

            macau_1x2_classified = {
                "h": m_h, "d": m_d, "a": m_a,
                "margin": mev["odds_1x2"]["margin"],
                "fair_h": fair_odds_h, "fair_d": fair_odds_d, "fair_a": fair_odds_a,
                "ev_h": round(ev_h * 100.0, 1),
                "ev_d": round(ev_d * 100.0, 1),
                "ev_a": round(ev_a * 100.0, 1),
                "best_sel": best_sel,
                "max_ev": round(best_1x2_ev * 100.0, 1),
                "badge": badge_1x2
            }

            if badge_1x2["label"] == "值博":
                match_opportunities.append({
                    "market": "1X2",
                    "market_name": "主客和",
                    "selection": best_sel,
                    "selection_name": sel_name,
                    "line": None,
                    "macau_odds": sel_odds,
                    "fair_odds": sel_fair,
                    "ev": round(best_1x2_ev, 4),
                    "badge": badge_1x2,
                    "kelly": {"fraction_pct": k_frac, "stake_units": k_units}
                })

        # 2. Asian Handicap Classification (Sharp Interpolation & Pinnacle Anchor)
        macau_ah_classified = None
        if mev.get("main_ah"):
            mah = mev["main_ah"]
            line = mah["line"]
            m_ah_h = mah["h"]
            m_ah_a = mah["a"]

            # Derive consensus AH from consensus 1X2 + totals via fitted Poisson/DC expectations
            cons_lh, cons_la = fit_lambdas_from_consensus(cons_p_h, cons_p_d, cons_p_a, cons_p_o25)
            cons_matrix = calc_dc_matrix(cons_lh, cons_la, rho=-0.05)
            cons_ah = derive_ah_fair_from_matrix(cons_matrix, line)

            # Check if football-data fixtures.csv has sharp Pinnacle/Bet365 AH
            sharp_ah_info = sharp_ah_fixtures.get((h_m, a_m))
            if not sharp_ah_info:
                for (sh_h, sh_a), info in sharp_ah_fixtures.items():
                    if teams_match(h_m, sh_h) and teams_match(a_m, sh_a):
                        sharp_ah_info = info
                        break

            # Interpolate sharp AH line to Macau's line if within +/- 0.5
            if sharp_ah_info:
                sharp_l = sharp_ah_info["line"]
                if abs(sharp_l - line) <= 0.5:
                    dev_probs, _ = remove_margin([sharp_ah_info["h"], sharp_ah_info["a"]], method="power")
                    if dev_probs:
                        p_sharp_h = dev_probs[0]
                        mu_totals = solve_poisson_lambda_for_ou25(cons_p_o25)
                        sharp_lh, sharp_la = calibrate_lambdas_to_sharp_ah(mu_totals, sharp_l, p_sharp_h)
                        sharp_matrix = calc_dc_matrix(sharp_lh, sharp_la, rho=-0.05)
                        sharp_ah_at_macau = derive_ah_fair_from_matrix(sharp_matrix, line)
                        # Blend sharp-calibrated AH distribution with consensus AH
                        cons_ah = blend_ah_distributions(cons_ah, sharp_ah_at_macau, weight_consensus=0.5)

            # Model AH from Dixon-Coles
            dc_ah = derive_ah_fair_from_matrix(dc_matrix, line) if dc_matrix else None

            # Blend Consensus with Dixon-Coles (using promotee-attenuated weights)
            blended_ah = blend_ah_distributions(cons_ah, dc_ah, weight_consensus=cons_weight)

            # Expected return under 1u stake
            er_h = (
                blended_ah["p_win"] * m_ah_h +
                blended_ah["p_hw"] * (1.0 + (m_ah_h - 1.0) / 2.0) +
                blended_ah["p_push"] * 1.0 +
                blended_ah["p_hl"] * 0.5
            )
            er_a = (
                blended_ah["p_loss"] * m_ah_a +
                blended_ah["p_hl"] * (1.0 + (m_ah_a - 1.0) / 2.0) +
                blended_ah["p_push"] * 1.0 +
                blended_ah["p_hw"] * 0.5
            )

            raw_ev_ah_h = er_h - 1.0
            raw_ev_ah_a = er_a - 1.0
            best_raw_ah_ev = max(raw_ev_ah_h, raw_ev_ah_a)
            best_ah_sel = "H" if best_raw_ah_ev == raw_ev_ah_h else "A"
            best_ah_name = f"上盤 ({mev['home_zh']} {mah['hcap_disp']})" if best_ah_sel == "H" else f"下盤 ({mev['away_zh']})"
            sel_ah_odds = m_ah_h if best_ah_sel == "H" else m_ah_a

            # Sharp AH anchor check
            is_sharp_ah_anchored = True
            if sharp_ah_info and abs(sharp_ah_info["line"] - line) <= 0.25:
                sharp_odds = sharp_ah_info["h"] if best_ah_sel == "H" else sharp_ah_info["a"]
                if sel_ah_odds < sharp_odds:
                    is_sharp_ah_anchored = False

            # Loophole fix: if raw EV > 15% or < -25%, mark as 數據不足, NEVER 值博!
            if best_raw_ah_ev > 0.15 or best_raw_ah_ev < -0.25:
                badge_ah = {
                    "label": "數據不足",
                    "tier": "insufficient",
                    "badgeClass": "badge-val-neutral",
                    "ev_pct": round(best_raw_ah_ev * 100.0, 1)
                }
            elif best_raw_ah_ev > 0.03 and is_sharp_ah_anchored:
                badge_ah = classify_ev(best_raw_ah_ev)
            elif best_raw_ah_ev > 0.03 and not is_sharp_ah_anchored:
                badge_ah = {
                    "label": "一般",
                    "tier": "neutral",
                    "badgeClass": "badge-val-neutral",
                    "ev_pct": round(best_raw_ah_ev * 100.0, 1)
                }
            else:
                badge_ah = classify_ev(best_raw_ah_ev)

            fair_ah_h = round(m_ah_h / er_h, 2) if er_h > 0.05 else blended_ah["fair_h"]
            fair_ah_a = round(m_ah_a / er_a, 2) if er_a > 0.05 else blended_ah["fair_a"]
            sel_ah_fair = fair_ah_h if best_ah_sel == "H" else fair_ah_a
            win_rate_h = blended_ah["eff_win_h"]
            win_rate_a = blended_ah["eff_win_a"]
            sel_ah_p = win_rate_h if best_ah_sel == "H" else win_rate_a

            k_frac, k_units = calc_kelly_stake(sel_ah_p, sel_ah_odds)

            macau_ah_classified = {
                "line": line,
                "hcap_disp": mah["hcap_disp"],
                "h": m_ah_h, "a": m_ah_a,
                "margin": mah["margin"],
                "fair_h": fair_ah_h, "fair_a": fair_ah_a,
                "ev_h": round(raw_ev_ah_h * 100.0, 1),
                "ev_a": round(raw_ev_ah_a * 100.0, 1),
                "best_sel": best_ah_sel,
                "max_ev": round(best_raw_ah_ev * 100.0, 1),
                "badge": badge_ah
            }

            if badge_ah["label"] == "值博":
                match_opportunities.append({
                    "market": "AH",
                    "market_name": "讓球",
                    "selection": best_ah_sel,
                    "selection_name": best_ah_name,
                    "line": line,
                    "macau_odds": sel_ah_odds,
                    "fair_odds": sel_ah_fair,
                    "ev": round(best_raw_ah_ev, 4),
                    "badge": badge_ah,
                    "kelly": {"fraction_pct": k_frac, "stake_units": k_units}
                })

        # 3. Over/Under Classification at Macau's exact line
        macau_ou_classified = None
        if mev.get("main_ou"):
            mou = mev["main_ou"]
            line = mou["line"]
            m_ou_o = mou["over"]
            m_ou_u = mou["under"]

            ou_line_probs = convert_ou_line_probs(fair_p_o25, line)
            er_o = calc_ou_expected_return(ou_line_probs, line, m_ou_o, is_over=True)
            er_u = calc_ou_expected_return(ou_line_probs, line, m_ou_u, is_over=False)

            ev_ou_o = er_o - 1.0
            ev_ou_u = er_u - 1.0
            best_ou_ev = max(ev_ou_o, ev_ou_u)
            best_ou_sel = "Over" if best_ou_ev == ev_ou_o else "Under"
            best_ou_name = f"大波 (>{mou['hcap_disp']})" if best_ou_sel == "Over" else f"細波 (<{mou['hcap_disp']})"
            sel_ou_odds = m_ou_o if best_ou_sel == "Over" else m_ou_u

            # Pinnacle Anchor for Totals:
            # If Pinnacle has odds on this match for this line or 2.5:
            is_pin_anchored_ou = True
            pin_line_info = pinnacle_totals.get(float(line)) or (pinnacle_totals.get(2.5) if abs(line - 2.5) < 0.3 else None)
            if pin_line_info:
                pin_odds_sel = pin_line_info["over"] if best_ou_sel == "Over" else pin_line_info["under"]
                pin_prob_sel = pin_line_info["prob_over"] if best_ou_sel == "Over" else pin_line_info["prob_under"]
                ev_vs_pin_ou = (pin_prob_sel * sel_ou_odds) - 1.0
                if sel_ou_odds <= pin_odds_sel or ev_vs_pin_ou <= 0.0:
                    is_pin_anchored_ou = False

            # Tail variance guard: if line is far from 2.5 (|line - 2.5| >= 0.75), Poisson tails are unreliable
            is_far_tail = abs(line - 2.5) >= 0.75

            if best_ou_ev > 0.15 or best_ou_ev < -0.25 or (is_far_tail and not is_pin_anchored_ou):
                badge_ou = {
                    "label": "數據不足" if (best_ou_ev > 0.15 or best_ou_ev < -0.25) else "一般",
                    "tier": "insufficient" if (best_ou_ev > 0.15 or best_ou_ev < -0.25) else "neutral",
                    "badgeClass": "badge-val-neutral",
                    "ev_pct": round(best_ou_ev * 100.0, 1)
                }
            elif best_ou_ev > 0.03 and is_pin_anchored_ou:
                badge_ou = classify_ev(best_ou_ev)
            elif best_ou_ev > 0.03 and not is_pin_anchored_ou:
                badge_ou = {
                    "label": "一般",
                    "tier": "neutral",
                    "badgeClass": "badge-val-neutral",
                    "ev_pct": round(best_ou_ev * 100.0, 1)
                }
            else:
                badge_ou = classify_ev(best_ou_ev)

            fair_ou_o = round(m_ou_o / er_o, 2) if er_o > 0.05 else 99.0
            fair_ou_u = round(m_ou_u / er_u, 2) if er_u > 0.05 else 99.0
            sel_ou_fair = fair_ou_o if best_ou_sel == "Over" else fair_ou_u

            # True Poisson win probability for Kelly stake
            is_quarter = abs((line * 2) % 1) > 1e-5
            if not is_quarter:
                p_dict = ou_line_probs.get("p", {})
                if best_ou_sel == "Over":
                    sel_ou_p = p_dict.get("over", 0.0) + 0.5 * p_dict.get("push", 0.0)
                else:
                    sel_ou_p = p_dict.get("under", 0.0) + 0.5 * p_dict.get("push", 0.0)
            else:
                p1 = ou_line_probs.get("p1", {})
                p2 = ou_line_probs.get("p2", {})
                if best_ou_sel == "Over":
                    sel_ou_p = 0.5 * (p1.get("over", 0.0) + 0.5 * p1.get("push", 0.0)) + 0.5 * (p2.get("over", 0.0) + 0.5 * p2.get("push", 0.0))
                else:
                    sel_ou_p = 0.5 * (p1.get("under", 0.0) + 0.5 * p1.get("push", 0.0)) + 0.5 * (p2.get("under", 0.0) + 0.5 * p2.get("push", 0.0))

            k_frac, k_units = calc_kelly_stake(sel_ou_p, sel_ou_odds)

            macau_ou_classified = {
                "line": line,
                "hcap_disp": mou["hcap_disp"],
                "over": m_ou_o, "under": m_ou_u,
                "margin": mou["margin"],
                "fair_o": fair_ou_o, "fair_u": fair_ou_u,
                "ev_o": round(ev_ou_o * 100.0, 1),
                "ev_u": round(ev_ou_u * 100.0, 1),
                "best_sel": best_ou_sel,
                "max_ev": round(best_ou_ev * 100.0, 1),
                "badge": badge_ou
            }

            if badge_ou["label"] == "值博":
                match_opportunities.append({
                    "market": "OU",
                    "market_name": "大細",
                    "selection": best_ou_sel,
                    "selection_name": best_ou_name,
                    "line": line,
                    "macau_odds": sel_ou_odds,
                    "fair_odds": sel_ou_fair,
                    "ev": round(best_ou_ev, 4),
                    "badge": badge_ou,
                    "kelly": {"fraction_pct": k_frac, "stake_units": k_units}
                })

        # Determine Best Opportunity for this match card
        best_opp = None
        if match_opportunities:
            match_opportunities.sort(key=lambda o: o["ev"], reverse=True)
            best_opp = match_opportunities[0]
            for opp in match_opportunities:
                if opp["ev"] > 0.03 and opp.get("badge", {}).get("label") == "值博":
                    line_suffix = f"_{opp['line']}" if opp.get('line') is not None else ""
                    bet_id = f"{mev['start_time'][:10]}_{ev_id}_{opp['market']}_{opp['selection']}{line_suffix}"
                    all_value_bets.append({
                        "id": bet_id,
                        "league": lg_code,
                        "league_name": mev["league_zh"],
                        "home": h_m,
                        "away": a_m,
                        "home_zh": mev["home_zh"],
                        "away_zh": mev["away_zh"],
                        "kickoff": mev["start_time"],
                        "market": opp["market"],
                        "market_name": opp["market_name"],
                        "selection": opp["selection"],
                        "selection_name": opp["selection_name"],
                        "line": opp.get("line"),
                        "macau_odds": opp["macau_odds"],
                        "fair_odds": opp["fair_odds"],
                        "ev": opp["ev"]
                    })
        elif macau_1x2_classified:
            # Fallback display card when no market is 值博
            best_opp = {
                "market": "1X2",
                "market_name": "主客和",
                "selection": macau_1x2_classified["best_sel"],
                "selection_name": f"主勝 ({mev['home_zh']})" if macau_1x2_classified["best_sel"] == "H" else ("和局" if macau_1x2_classified["best_sel"] == "D" else f"客勝 ({mev['away_zh']})"),
                "macau_odds": macau_1x2_classified[macau_1x2_classified["best_sel"].lower()],
                "fair_odds": macau_1x2_classified[f"fair_{macau_1x2_classified['best_sel'].lower()}"],
                "ev": round(macau_1x2_classified["max_ev"] / 100.0, 4),
                "badge": macau_1x2_classified["badge"],
                "kelly": {"fraction_pct": 0.0, "stake_units": 0.0}
            }

        # Bookmaker Discrepancy & Best Price across market
        outliers = []
        best_market_prices = {
            "h": (mev.get("odds_1x2", {}).get("h", 1.0), "Macau Slot"),
            "d": (mev.get("odds_1x2", {}).get("d", 1.0), "Macau Slot"),
            "a": (mev.get("odds_1x2", {}).get("a", 1.0), "Macau Slot")
        }

        for b in books_list:
            b1 = b.get("1x2", {})
            b_name = b.get("title", b.get("key"))
            for outcome, f_prob in [("h", fair_p_h), ("d", fair_p_d), ("a", fair_p_a)]:
                price = b1.get(outcome)
                if price and price > 1.0:
                    if price > best_market_prices[outcome][0]:
                        best_market_prices[outcome] = (price, b_name)
                    fair_o = 1.0 / f_prob
                    discrepancy = (price / fair_o) - 1.0
                    if discrepancy > 0.04:
                        outliers.append({
                            "book": b_name,
                            "selection": outcome.upper(),
                            "odds": price,
                            "fair_odds": round(fair_o, 2),
                            "discrepancy_pct": round(discrepancy * 100.0, 1)
                        })

        # P2: Prioritize Pinnacle in booksList display table
        display_books = []
        if pinnacle_book:
            display_books.append(pinnacle_book)
        for b in books_list:
            if b.get("key") != "pinnacle" and len(display_books) < 8:
                display_books.append(b)

        fixture_obj = {
            "id": ev_id,
            "matchKey": match_key,
            "date": mev["start_time"][:10],
            "time": mev["start_time"][11:16],
            "kickoffIso": mev["kickoff_iso"],
            "dayLabel": format_day_label(mev["start_time"][:10], now_macau.date()),
            "league": lg_code,
            "leagueName": mev["league_zh"],
            "leagueEn": mev["league_en"],
            "home": h_m,
            "away": a_m,
            "homeZh": mev["home_zh"],
            "awayZh": mev["away_zh"],
            "xg": {"home": xg_home, "away": xg_away},
            "macau": {
                "odds_1x2": macau_1x2_classified,
                "main_ah": macau_ah_classified,
                "all_ah": mev.get("all_ah", []),
                "main_ou": macau_ou_classified,
                "all_ou": mev.get("all_ou", [])
            },
            "consensus": {
                "pinnaclePresent": pinnacle_book is not None,
                "booksCount": len(books_list),
                "fair_1x2": {"h": fair_odds_h, "d": fair_odds_d, "a": fair_odds_a},
                "prob_1x2": {"h": round(fair_p_h * 100.0, 1), "d": round(fair_p_d * 100.0, 1), "a": round(fair_p_a * 100.0, 1)},
                "prob_ou25": {"over": round(fair_p_o25 * 100.0, 1), "under": round(fair_p_u25 * 100.0, 1)}
            },
            "bestOpportunity": best_opp,
            "booksList": display_books,
            "outliers": outliers,
            "sparkline": snapshot_histories.get(match_key, [])
        }
        processed_fixtures.append(fixture_obj)

    # Sort fixtures by kickoff time
    processed_fixtures.sort(key=lambda f: f["kickoffIso"])

    # Fallback to last-good fixtures if pipeline failed (P1-7)
    if not processed_fixtures and health_status["status"] == "error":
        processed_fixtures = load_previous_fixtures_from_data_js(os.path.join(DATA_DIR, "data.js"))
        print(f"[Pipeline] Retained {len(processed_fixtures)} last-good fixtures from previous data.js due to health error.")

    # Step 6: Save snapshot and update track record
    print(f"[Pipeline] 5/7 Saving snapshot and updating track record ({len(all_value_bets)} value bets detected)...")
    save_snapshot(processed_fixtures, base_data_dir=DATA_DIR)

    track_record = update_track_record(
        all_value_bets,
        finished_matches=finished_matches_for_tracker,
        filepath=os.path.join(DATA_DIR, "track_record.json"),
        current_time_utc=now_utc
    )

    # Log unmatched matches
    unmatched_file = os.path.join(DATA_DIR, "unmatched.json")
    with open(unmatched_file, "w", encoding="utf-8") as f:
        json.dump(unmatched_events, f, ensure_ascii=False, indent=2)

    # Missing Cantonese team names (promoted / new clubs): warn, do not fail the build
    # Also recompute from fixture English names so Macau-provided CJK does not hide gaps in TEAM_NAMES.
    fixture_team_names = []
    for fx in processed_fixtures:
        fixture_team_names.append(fx.get("home") or "")
        fixture_team_names.append(fx.get("away") or "")
    missing_sorted = list_missing_cantonese(fixture_team_names)
    # Merge any names collected during zh resolution (e.g. early path)
    for n in missing_team_names:
        if n and n not in missing_sorted:
            missing_sorted.append(n)
    missing_sorted = sorted(set(missing_sorted))
    missing_payload = {
        "generated_at": now_iso_utc,
        "teams": missing_sorted,
    }
    missing_file = os.path.join(DATA_DIR, "missing_team_names.json")
    with open(missing_file, "w", encoding="utf-8") as f:
        json.dump(missing_payload, f, ensure_ascii=False, indent=2)

    strict_team_names = os.environ.get("ODDS_LAB_STRICT_TEAM_NAMES", "").strip() == "1"
    if missing_sorted:
        miss_msg = f"Missing Cantonese team names ({len(missing_sorted)}): " + ", ".join(missing_sorted)
        print(f"[Pipeline] WARNING: {miss_msg}")
        for t in missing_sorted:
            print(f"::warning title=Missing Cantonese team name::{t}")
        print(f"::warning title=Missing Cantonese team names::{miss_msg}")
        health_status["issues"].append(miss_msg)
        if strict_team_names:
            health_status["status"] = "error"
            print("[Pipeline] ODDS_LAB_STRICT_TEAM_NAMES=1 → treating missing names as error")
        elif health_status["status"] == "healthy":
            health_status["status"] = "warning"

    # Step 7: Write health status
    health_file = os.path.join(DATA_DIR, "health.json")
    with open(health_file, "w", encoding="utf-8") as f:
        json.dump(health_status, f, ensure_ascii=False, indent=2)

    print(f"[Pipeline] 6/7 Health status: {health_status['status']} ({len(health_status['issues'])} issues)")

    return {
        "generatedAt": now_iso_utc,
        "health": health_status,
        "macauFixtures": processed_fixtures,
        "trackRecord": track_record,
        "sourcesStatus": health_status["sources"]
    }

if __name__ == "__main__":
    res = run_pipeline()
    print(f"[Pipeline] Completed! Processed {len(res['macauFixtures'])} live fixtures.")
