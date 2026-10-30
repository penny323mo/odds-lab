#!/usr/bin/env python3
"""
test_v4.py - Unit tests for Odds Lab v4 Python modules:
- Asian Handicap quarter-line settlement (0, ±0.25, ±0.5, ±0.75, ±1.0)
- Over/Under quarter-line settlement (2.0, 2.25, 2.5, 2.75, 3.0)
- Multiplicative and Power de-vigging margin removal
- Poisson expected goals solver and totals line conversion
- Team name normalization and cross-source matching
- EV classification thresholds (+3% / -3%)
- Data and health schema validation
"""

import unittest
import json
import os
import sys
from datetime import datetime, timezone

# Add root and scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from math_engine import (
    remove_margin_multiplicative, remove_margin_power, remove_margin,
    solve_poisson_lambda_for_ou25, convert_ou_line_probs,
    settle_ah_single, settle_ou_single, classify_ev, calc_kelly_stake,
    calibrate_lambdas_to_sharp_ah, calc_dc_matrix, derive_ah_fair_from_matrix
)
from pipeline import validate_macau_response
from record_tracker import update_track_record
from team_names import (
    TEAM_NAMES, get_team_cantonese, list_missing_cantonese, normalize_team_name,
    teams_match, strip_accents
)

class TestOddsLabV4(unittest.TestCase):

    def test_margin_removal_multiplicative(self):
        # Bet365 typical 1X2 market: 2.00, 3.40, 3.80
        odds = [2.00, 3.40, 3.80]
        probs, margin = remove_margin_multiplicative(odds)
        self.assertAlmostEqual(sum(probs), 1.0, places=7)
        self.assertGreater(margin, 0.0)
        self.assertLess(margin, 0.15)
        # Probabilities should be strictly proportional to inverse odds
        self.assertAlmostEqual(probs[0] / probs[1], (1.0/odds[0]) / (1.0/odds[1]), places=5)

    def test_margin_removal_power(self):
        odds = [1.80, 2.05]
        probs, margin = remove_margin_power(odds)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertGreater(probs[0], probs[1])

    def test_poisson_lambda_solver_and_conversion(self):
        # Over 2.5 = 50% implies around 2.67 expected goals
        mu = solve_poisson_lambda_for_ou25(0.50)
        self.assertAlmostEqual(mu, 2.674, delta=0.01)

        # Convert to 2.25 line (quarter line: split 2.0 and 2.5)
        res = convert_ou_line_probs(0.50, 2.25)
        self.assertEqual(res["target_line"], 2.25)
        self.assertEqual(res["sub1"], 2.0)
        self.assertEqual(res["sub2"], 2.5)
        self.assertIn("p1", res)
        self.assertIn("p2", res)

    def test_ah_quarter_line_settlement(self):
        # Line -0.75 @ odds 2.00 (half on -0.5, half on -1.0)
        # Home wins by 2: both -0.5 and -1.0 win -> Full Win (payoff 2.0)
        self.assertEqual(settle_ah_single(2, -0.75, 2.0), 2.0)
        # Home wins by 1: -0.5 wins, -1.0 pushes -> Half Win (0.5 * 2.0 + 0.5 * 1.0 = 1.5)
        self.assertEqual(settle_ah_single(1, -0.75, 2.0), 1.5)
        # Draw (0): both lose -> Full Loss (0.0)
        self.assertEqual(settle_ah_single(0, -0.75, 2.0), 0.0)

        # Line +0.75 @ odds 2.00 (away perspective of -0.75)
        # Home wins by 1 (diff -1 for away): +0.5 loses, +1.0 pushes -> Half Loss (0.5)
        self.assertEqual(settle_ah_single(-1, 0.75, 2.0), 0.5)

        # Line -0.25 @ odds 2.00 (half on 0, half on -0.5)
        # Home wins by 1: both win -> Full Win (2.0)
        self.assertEqual(settle_ah_single(1, -0.25, 2.0), 2.0)
        # Draw (0): 0 pushes, -0.5 loses -> Half Loss (0.5 * 1.0 + 0.5 * 0.0 = 0.5)
        self.assertEqual(settle_ah_single(0, -0.25, 2.0), 0.5)
        # Loss (-1): both lose -> Full Loss (0.0)
        self.assertEqual(settle_ah_single(-1, -0.25, 2.0), 0.0)

    def test_ou_quarter_line_settlement(self):
        # Over 2.25 @ 2.00 (split 2.0 and 2.5)
        # 3 goals: both win -> Full Win (2.0)
        self.assertEqual(settle_ou_single(3, 2.25, 2.0, is_over=True), 2.0)
        # 2 goals: 2.0 pushes (0.5 return), 2.5 loses (0.0) -> Half Loss (0.5)
        self.assertEqual(settle_ou_single(2, 2.25, 2.0, is_over=True), 0.5)
        # 1 goal: both lose -> Full Loss (0.0)
        self.assertEqual(settle_ou_single(1, 2.25, 2.0, is_over=True), 0.0)

        # Under 2.25 @ 2.00
        # 2 goals: 2.0 pushes (0.5 * 1.0), 2.5 wins (0.5 * 2.0) -> Half Win (1.5)
        self.assertEqual(settle_ou_single(2, 2.25, 2.0, is_over=False), 1.5)

    def test_ev_classification_labels(self):
        # Values strictly following the specification:
        # > +3% -> 值博
        # -3% .. +3% -> 一般
        # < -3% -> 唔值博
        pos = classify_ev(0.045)
        self.assertEqual(pos["label"], "值博")
        self.assertEqual(pos["tier"], "value")

        neutral_high = classify_ev(0.025)
        self.assertEqual(neutral_high["label"], "一般")
        self.assertEqual(neutral_high["tier"], "neutral")

        neutral_low = classify_ev(-0.025)
        self.assertEqual(neutral_low["label"], "一般")
        self.assertEqual(neutral_low["tier"], "neutral")

        neg = classify_ev(-0.05)
        self.assertEqual(neg["label"], "唔值博")
        self.assertEqual(neg["tier"], "avoid")

    def test_team_name_normalization_and_matching(self):
        # Strips accents and aliases
        self.assertEqual(strip_accents("Alavés"), "Alaves")
        self.assertEqual(strip_accents("Málaga"), "Malaga")
        self.assertEqual(strip_accents("Bayern München"), "Bayern Munchen")

        # Cross-source matching
        self.assertTrue(teams_match("Brighton Hove Albion", "Brighton and Hove Albion"))
        self.assertTrue(teams_match("Newcastle Utd", "Newcastle United"))
        self.assertTrue(teams_match("Paris St Germain", "Paris Saint Germain"))
        self.assertTrue(teams_match("AS Monaco", "Monaco"))
        self.assertTrue(teams_match("Bayer 04 Leverkusen", "Bayer Leverkusen"))
        self.assertTrue(teams_match("Athletic Bilbao", "Ath Bilbao"))
        self.assertTrue(teams_match("Atalanta BC", "Atalanta"))

        # Macau Slot often uses full German club forms with year suffixes.
        # Regression 2026-10-29: "SC Paderborn 07" fell through to ASCII and
        # failed v4.test.js CJK assertion on scheduled Update Data runs.
        self.assertEqual(normalize_team_name("SC Paderborn 07"), "paderborn")
        self.assertEqual(get_team_cantonese("SC Paderborn 07"), "柏達邦")
        self.assertEqual(get_team_cantonese("SC Paderborn"), "柏達邦")
        self.assertTrue(teams_match("SC Paderborn 07", "Paderborn"))
        self.assertTrue(bool(__import__("re").search(r"[\u4e00-\u9fff]", get_team_cantonese("SC Paderborn 07"))))

    def test_paris_fc_cantonese_mapped(self):
        """Paris FC (Ligue 1 promotee) must resolve to 巴黎FC with CJK."""
        self.assertEqual(get_team_cantonese("Paris FC"), "巴黎FC")
        self.assertEqual(get_team_cantonese("Paris Football Club"), "巴黎FC")
        self.assertTrue(bool(__import__("re").search(r"[\u4e00-\u9fff]", get_team_cantonese("Paris FC"))))
        self.assertNotIn("Paris FC", list_missing_cantonese(["Paris FC", "Paris Football Club", "Arsenal"]))

    def test_get_team_cantonese_fallback_and_collect_missing(self):
        """Unmapped teams fall back to English and can be collected for warnings."""
        unknown = "Totally Fake United FC 2099"
        self.assertEqual(get_team_cantonese(unknown), unknown)
        bucket = []
        self.assertEqual(get_team_cantonese(unknown, collect_missing=bucket), unknown)
        self.assertEqual(bucket, [unknown])
        # Mapped names must not be collected
        get_team_cantonese("Arsenal", collect_missing=bucket)
        get_team_cantonese("Paris FC", collect_missing=bucket)
        self.assertEqual(bucket, [unknown])
        # Second collect of same unknown does not duplicate
        get_team_cantonese(unknown, collect_missing=bucket)
        self.assertEqual(bucket, [unknown])

    def test_list_missing_cantonese_helper(self):
        missing = list_missing_cantonese([
            "Arsenal",
            "Totally Fake United FC 2099",
            "Paris FC",
            "Another Fake Club",
            "Totally Fake United FC 2099",  # dedupe
            "",
            None,
        ])
        self.assertEqual(missing, ["Another Fake Club", "Totally Fake United FC 2099"])

    def test_schema_and_health_file(self):

        health_path = os.path.join(os.path.dirname(__file__), "..", "data", "health.json")
        self.assertTrue(os.path.exists(health_path), "data/health.json must exist")
        with open(health_path, "r", encoding="utf-8") as f:
            h = json.load(f)
        self.assertIn("status", h)
        self.assertIn("checked_at", h)
        self.assertIn("sources", h)
        self.assertIn("macau_slot", h["sources"])
        self.assertIn("the_odds_api", h["sources"])

    # --- P0 / P1 Targeted Test Cases ---

    def test_p0_1_track_record_kickoff_date_gating(self):
        """
        P0-1: A bet dated 2026-10-28 must stay 'pending' when the only CSV match
        is from a previous season (e.g. 2025-11-24).
        """
        from record_tracker import update_track_record
        from datetime import datetime, timezone
        test_file = "/tmp/test_track_record_p0_1.json"
        if os.path.exists(test_file):
            os.remove(test_file)

        value_bets = [{
            "id": "2026-10-28_test_Torino_Como_AH_A_0.75",
            "league": "I1",
            "league_name": "意甲",
            "home": "Torino",
            "away": "Como",
            "kickoff": "2026-10-28 03:45:00",
            "market": "AH",
            "market_name": "讓球",
            "selection": "A",
            "selection_name": "下盤 (科木)",
            "line": 0.75,
            "macau_odds": 1.97,
            "fair_odds": 1.90,
            "ev": 0.0368
        }]

        # Finished match from last season (2025-11-24)
        finished_matches = [{
            "date": "2025-11-24",
            "home": "Torino",
            "away": "Como",
            "fthg": 1,
            "ftag": 5
        }]

        # Current time before kickoff (e.g. 2026-10-25)
        current_time = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)
        rec = update_track_record(value_bets, finished_matches=finished_matches, filepath=test_file, current_time_utc=current_time)

        bet = rec["bets"][0]
        self.assertEqual(bet["status"], "pending", "Upcoming match must stay pending, not settle with last season score")
        self.assertIsNone(bet["score"])
        self.assertIsNone(bet["profit_1u"])
        self.assertEqual(rec["summary"]["settled_bets"], 0)

        # Now simulate actual matchday finish on 2026-10-28
        actual_finished = [{
            "date": "2026-10-28",
            "home": "Torino",
            "away": "Como",
            "fthg": 0,
            "ftag": 1
        }]
        after_kickoff_time = datetime(2026, 10, 28, 6, 0, 0, tzinfo=timezone.utc)
        rec2 = update_track_record([], finished_matches=actual_finished, filepath=test_file, current_time_utc=after_kickoff_time)
        bet2 = rec2["bets"][0]
        self.assertEqual(bet2["status"], "settled")
        self.assertEqual(bet2["score"], "0-1")
        # Line 0.75 away is -0.75; 1-0 win is a half-win on a quarter handicap
        self.assertEqual(bet2["outcome"], "half_win")

    def test_p0_2_away_asian_handicap_settlement(self):
        """
        P0-2: Away Asian Handicap selection ('A') must be settled from the away perspective.
        Home wins 2-0, home line is -0.5:
        Home bet wins (payoff odds). Away bet loses (payoff 0.0), NOT 2.0.
        """
        from record_tracker import update_track_record
        from datetime import datetime, timezone
        test_file = "/tmp/test_track_record_p0_2.json"
        if os.path.exists(test_file):
            os.remove(test_file)

        value_bets = [
            {
                "id": "2026-10-25_test_Liverpool_Brighton_AH_A",
                "league": "E0",
                "home": "Liverpool",
                "away": "Brighton",
                "kickoff": "2026-10-25 20:00:00",
                "market": "AH",
                "selection": "A",
                "line": -0.5,  # Liverpool -0.5, Brighton +0.5
                "macau_odds": 2.00,
                "fair_odds": 1.90,
                "ev": 0.05
            },
            {
                "id": "2026-10-25_test_Liverpool_Brighton_AH_H",
                "league": "E0",
                "home": "Liverpool",
                "away": "Brighton",
                "kickoff": "2026-10-25 20:00:00",
                "market": "AH",
                "selection": "H",
                "line": -0.5,
                "macau_odds": 1.90,
                "fair_odds": 1.80,
                "ev": 0.05
            }
        ]

        # Liverpool 2 - 0 Brighton
        finished = [{
            "date": "2026-10-25",
            "home": "Liverpool",
            "away": "Brighton",
            "fthg": 2,
            "ftag": 0
        }]

        post_match_time = datetime(2026, 10, 25, 23, 0, 0, tzinfo=timezone.utc)
        rec = update_track_record(value_bets, finished_matches=finished, filepath=test_file, current_time_utc=post_match_time)

        away_bet = next(b for b in rec["bets"] if b["selection"] == "A")
        home_bet = next(b for b in rec["bets"] if b["selection"] == "H")

        self.assertEqual(away_bet["outcome"], "loss", "Brighton +0.5 vs Liverpool 2-0 must be a loss")
        self.assertEqual(away_bet["profit_1u"], -1.0)

        self.assertEqual(home_bet["outcome"], "win", "Liverpool -0.5 vs Brighton 2-0 must be a win")
        self.assertAlmostEqual(home_bet["profit_1u"], 0.90)

    def test_p0_3_ah_consensus_blend_prevents_unrealistic_ev(self):
        """
        P0-3: Balanced 1.90/1.90 Macau AH with balanced consensus expectations
        must classify as 一般, not 值博.
        """
        from math_engine import (
            calc_dc_matrix, fit_lambdas_from_consensus,
            derive_ah_fair_from_matrix, blend_ah_distributions, classify_ev
        )
        # Evenly matched consensus: 37% H, 28% D, 35% A, Over 2.5 = 50%
        cons_lh, cons_la = fit_lambdas_from_consensus(0.37, 0.28, 0.35, 0.50)
        cons_mat = calc_dc_matrix(cons_lh, cons_la)
        cons_ah = derive_ah_fair_from_matrix(cons_mat, 0.0)

        # Dixon-Coles model has mild home preference
        dc_mat = calc_dc_matrix(1.4, 1.2)
        dc_ah = derive_ah_fair_from_matrix(dc_mat, 0.0)

        blended_ah = blend_ah_distributions(cons_ah, dc_ah, weight_consensus=0.7)

        # Macau balanced line 1.90 / 1.90 at line 0
        m_h = 1.90
        m_a = 1.90
        er_h = blended_ah["p_win"] * m_h + blended_ah["p_push"] * 1.0 + blended_ah["p_loss"] * 0.0
        ev_h = er_h - 1.0

        badge = classify_ev(ev_h)
        self.assertNotEqual(badge["label"], "值博", "Balanced 1.90/1.90 must not classify as 值博")
        self.assertIn(badge["label"], ["一般", "唔值博"])

    def test_p1_8_clv_freeze_and_calculation(self):
        """
        P1-8: Two-step update test: First detect @ 2.10, later close @ 2.00 -> CLV +5.0%.
        Average CLV must be averaged over settled bets only.
        """
        from record_tracker import update_track_record
        from datetime import datetime, timezone
        test_file = "/tmp/test_track_record_p1_8.json"
        if os.path.exists(test_file):
            os.remove(test_file)

        t1 = datetime(2026, 10, 25, 10, 0, 0, tzinfo=timezone.utc)
        bet_step1 = [{
            "id": "2026-10-25_test_clv_bet",
            "league": "E0",
            "home": "Arsenal",
            "away": "Chelsea",
            "kickoff": "2026-10-25 23:00:00",
            "market": "1X2",
            "selection": "H",
            "macau_odds": 2.10,
            "fair_odds": 1.95,
            "ev": 0.0769
        }]
        rec1 = update_track_record(bet_step1, filepath=test_file, current_time_utc=t1)
        self.assertEqual(rec1["bets"][0]["clv"], 0.0)
        self.assertEqual(rec1["summary"]["avg_clv"], 0.0)

        # Step 2: Odds drop to 2.00 before kickoff
        t2 = datetime(2026, 10, 25, 14, 0, 0, tzinfo=timezone.utc)
        bet_step2 = [{
            "id": "2026-10-25_test_clv_bet",
            "macau_odds": 2.00,
            "fair_odds": 1.95,
            "ev": 0.025
        }]
        rec2 = update_track_record(bet_step2, filepath=test_file, current_time_utc=t2)
        # 2.10 / 2.00 - 1 = +0.05 (+5%)
        self.assertAlmostEqual(rec2["bets"][0]["clv"], 0.05, places=3)

        # Now settle the bet after kickoff
        t3 = datetime(2026, 10, 25, 21, 0, 0, tzinfo=timezone.utc)
        finished = [{
            "date": "2026-10-25",
            "home": "Arsenal",
            "away": "Chelsea",
            "fthg": 2,
            "ftag": 1
        }]
        rec3 = update_track_record([], finished_matches=finished, filepath=test_file, current_time_utc=t3)
        self.assertEqual(rec3["summary"]["settled_bets"], 1)
        self.assertAlmostEqual(rec3["summary"]["avg_clv"], 5.0, places=1)

    def test_p1_9_strict_team_matching_no_collisions(self):
        """
        P1-9: Teams match must NOT treat Man City = Man United or Real Madrid = Real Sociedad.
        """
        self.assertFalse(teams_match("Man City", "Man United"))
        self.assertFalse(teams_match("Manchester City", "Manchester United"))
        self.assertFalse(teams_match("Manchester City", "Manchester Utd"))
        self.assertFalse(teams_match("Real Madrid", "Real Sociedad"))
        self.assertFalse(teams_match("Real Madrid", "Sociedad"))

        # Valid aliases must continue to match
        self.assertTrue(teams_match("Brighton", "Brighton Hove Albion"))
        self.assertTrue(teams_match("Athletic Bilbao", "Athletic Club"))
        self.assertTrue(teams_match("PSG", "Paris Saint-Germain"))

    def test_r3_pinnacle_anchor_rule(self):
        """
        R3 Sanity: If Macau odds <= Pinnacle odds, or EV vs Pinnacle is <= 0,
        the selection must never be badged as 值博.
        """
        macau_odds = 4.70
        pinnacle_odds = 5.07  # Pinnacle offers better odds on Hull than Macau
        # Direct rule: if macau_odds <= pinnacle_odds, it is an anti-value trap
        self.assertTrue(macau_odds <= pinnacle_odds)
        # classify_ev of clamped or non-anchored is 一般 / 數據不足
        badge_non_anchored = {
            "label": "一般",
            "tier": "neutral",
            "badgeClass": "badge-val-neutral",
            "ev_pct": -7.3
        }
        self.assertNotEqual(badge_non_anchored["label"], "值博")

    def test_r3_clamp_loophole_eliminated_marks_insufficient(self):
        """
        R3 Sanity: If raw EV > 15% (e.g. 18.9%) or is_clamped=True,
        classify_ev must mark as 數據不足, NOT 值博.
        """
        badge_high = classify_ev(0.189)
        self.assertEqual(badge_high["label"], "數據不足")
        self.assertEqual(badge_high["tier"], "insufficient")
        self.assertNotEqual(badge_high["label"], "值博")

        badge_clamped = classify_ev(0.05, is_clamped=True)
        self.assertEqual(badge_clamped["label"], "數據不足")
        self.assertEqual(badge_clamped["tier"], "insufficient")

    def test_r3_sharp_ah_interpolation_to_quarter_line(self):
        """
        R3 Sanity: Sharp line at 0.0 interpolates to Macau quarter lines (+0.25 / -0.25)
        via score matrix calibration without requiring exact line match.
        """
        mu = 2.6
        p_sharp_h = 0.54  # Sharp book has slight home bias on line 0
        sharp_lh, sharp_la = calibrate_lambdas_to_sharp_ah(mu, 0.0, p_sharp_h)
        self.assertGreater(sharp_lh, sharp_la)
        self.assertAlmostEqual(sharp_lh + sharp_la, mu, places=4)

        matrix = calc_dc_matrix(sharp_lh, sharp_la, rho=-0.05)
        quarter_ah = derive_ah_fair_from_matrix(matrix, -0.25)
        self.assertIn("p_win", quarter_ah)
        self.assertIn("p_hw", quarter_ah)
        self.assertGreater(quarter_ah["fair_h"], 1.0)
        self.assertGreater(quarter_ah["fair_a"], 1.0)

    def test_r3_truncated_macau_response_fails_validation(self):
        """
        R3 Health/Last-good: If Macau Slot returns empty list or events without odds markets,
        validation must fail so previous good data.js fixtures are preserved.
        """
        # Case 1: unpriced events
        unpriced = {
            "events": [{
                "ev_id": "999",
                "start_time": "2026-10-26 20:00",
                "league_code": "E0",
                "home_en": "Arsenal",
                "away_en": "Chelsea"
                # Missing odds_1x2 and main_ah!
            }]
        }
        valid, msg = validate_macau_response(unpriced)
        self.assertFalse(valid)
        self.assertIn("contain no odds markets", msg)

        # Case 2: priced events
        priced = {
            "events": [{
                "ev_id": "999",
                "start_time": "2026-10-26 20:00",
                "league_code": "E0",
                "home_en": "Arsenal",
                "away_en": "Chelsea",
                "odds_1x2": {"h": 1.9, "d": 3.4, "a": 3.8, "margin": 0.08}
            }]
        }
        valid2, msg2 = validate_macau_response(priced)
        self.assertTrue(valid2)

    def test_r3_track_record_half_win_string_alignment(self):
        """
        R3 Fix: summary must correctly count both half_win and half_won strings.
        """
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            test_path = tf.name

        initial_state = {
            "version": 4,
            "bets": [
                {
                    "id": "test_bet_1",
                    "match_key": "E0_TeamA_TeamB",
                    "league": "E0",
                    "league_name": "英超",
                    "home": "TeamA",
                    "away": "TeamB",
                    "date": "2026-10-24",
                    "kickoff": "2026-10-24 19:00:00",
                    "closing_frozen": True,
                    "market": "AH",
                    "selection": "H",
                    "line": -0.75,
                    "macau_odds": 2.00,
                    "initial_macau_odds": 2.00,
                    "closing_macau_odds": 2.00,
                    "clv": 0.02,
                    "status": "pending",
                    "score": None,
                    "outcome": None,
                    "profit_1u": None
                }
            ],
            "summary": {
                "total_bets": 1, "settled_bets": 0, "pending_bets": 1,
                "won": 0, "half_won": 0, "push": 0, "half_lost": 0, "lost": 0,
                "profit_units": 0.0, "roi_pct": 0.0, "avg_clv": 0.0
            }
        }
        with open(test_path, "w") as f:
            json.dump(initial_state, f)

        # Settle with 1-0 score (Home wins by 1 on line -0.75 -> half win!)
        finished = [{
            "date": "2026-10-24",
            "home": "TeamA",
            "away": "TeamB",
            "fthg": 1,
            "ftag": 0
        }]
        res = update_track_record([], finished_matches=finished, filepath=test_path,
                                  current_time_utc=datetime(2026, 10, 25, tzinfo=timezone.utc))
        self.assertEqual(res["bets"][0]["outcome"], "half_win")
        self.assertEqual(res["summary"]["half_won"], 1)
        self.assertEqual(res["summary"]["total_profit"], 0.5)

        if os.path.exists(test_path):
            os.remove(test_path)


if __name__ == "__main__":
    unittest.main()
