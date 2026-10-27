const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const data = require('../data/data.js');
const model = require('../js/model.js');

test('v4 Schema: ODDS_DATA contains rich macauFixtures, trackRecord, sourcesStatus, health', () => {
  assert.ok(data, 'data must be defined');
  assert.ok(Array.isArray(data.macauFixtures), 'macauFixtures must be an array');
  assert.ok(data.macauFixtures.length > 0, 'macauFixtures must contain upcoming matches');

  assert.ok(data.trackRecord, 'trackRecord must exist');
  assert.ok(data.trackRecord.summary, 'trackRecord.summary must exist');
  assert.ok(typeof data.trackRecord.summary.total_bets === 'number');
  assert.ok(typeof data.trackRecord.summary.roi === 'number');
  assert.ok(typeof data.trackRecord.summary.avg_clv === 'number');

  assert.ok(data.sourcesStatus, 'sourcesStatus must exist');
  assert.ok(data.sourcesStatus.macau_slot, 'sourcesStatus.macau_slot must exist');
  assert.ok(data.sourcesStatus.the_odds_api, 'sourcesStatus.the_odds_api must exist');
  assert.ok(typeof data.sourcesStatus.the_odds_api.requests_remaining === 'number');

  assert.ok(data.health, 'health must exist');
  assert.ok(data.health.status === 'healthy' || data.health.status === 'warning');
});

test('v4 Macau Fixtures: structure, classification badges, and genuine Cantonese team names', () => {
  const cjkRegex = /[\u4e00-\u9fff]/;

  for (const fix of data.macauFixtures) {
    assert.ok(fix.id, 'id required');
    assert.ok(fix.home && fix.away, 'home and away required');
    // P1-4: Must contain genuine CJK characters and no literal '?'
    assert.ok(cjkRegex.test(fix.homeZh), `homeZh '${fix.homeZh}' for ${fix.home} must contain CJK`);
    assert.ok(cjkRegex.test(fix.awayZh), `awayZh '${fix.awayZh}' for ${fix.away} must contain CJK`);
    assert.ok(!fix.homeZh.includes('?'), `homeZh '${fix.homeZh}' must not contain '?'`);
    assert.ok(!fix.awayZh.includes('?'), `awayZh '${fix.awayZh}' must not contain '?'`);

    assert.ok(fix.dayLabel, 'dayLabel required');
    assert.ok(fix.macau, 'macau odds object required');
    assert.ok(fix.consensus, 'consensus object required');

    if (fix.bestOpportunity) {
      assert.ok(['值博', '一般', '唔值博'].includes(fix.bestOpportunity.badge.label));
      assert.ok(['value', 'neutral', 'avoid'].includes(fix.bestOpportunity.badge.tier));
    }
  }
});

test('v4 Snapshots: compact snapshots have distinct match keys and valid numeric odds (P1-5)', () => {
  const fs = require('node:fs');
  const glob = fs.readdirSync(path.join(__dirname, '../data/snapshots'))
    .filter(f => f.startsWith('snapshot_') && f.endsWith('.json'))
    .sort();
  assert.ok(glob.length > 0, 'At least one snapshot file must exist');

  const latestFile = path.join(__dirname, '../data/snapshots', glob[glob.length - 1]);
  const content = JSON.parse(fs.readFileSync(latestFile, 'utf8'));

  assert.ok(content.matches, 'matches dictionary required in snapshot');
  const matchKeys = Object.keys(content.matches);
  // Live fixture count varies by day (quiet days can have <10); only require a non-empty snapshot.
  assert.ok(matchKeys.length >= 1, `Expected at least 1 match key, got ${matchKeys.length}`);
  assert.ok(!matchKeys.includes('None_None_None'), "matchKey must not be 'None_None_None'");

  const firstRec = content.matches[matchKeys[0]];
  assert.ok(firstRec.time, 'snapshot record must have timestamp');
  if (firstRec.macau_1x2) {
    assert.ok(typeof firstRec.macau_1x2.h === 'number', 'Macau 1X2 H must be numeric');
    assert.ok(typeof firstRec.macau_1x2.d === 'number', 'Macau 1X2 D must be numeric');
    assert.ok(typeof firstRec.macau_1x2.a === 'number', 'Macau 1X2 A must be numeric');
  }
});

test('v4 Track Record: upcoming fixtures remain pending without historical cross-season contamination (P0-1)', () => {
  const tr = data.trackRecord;
  assert.ok(tr && tr.bets, 'trackRecord.bets must be present');

  for (const bet of tr.bets) {
    const kickoff = new Date(bet.kickoff.replace(' ', 'T') + '+08:00');
    const now = new Date();
    if (kickoff.getTime() > now.getTime()) {
      assert.equal(bet.status, 'pending', `Bet ${bet.id} with future kickoff (${bet.kickoff}) must not be settled`);
      assert.equal(bet.score, null, `Pending bet must not have historical score`);
      assert.equal(bet.profit_1u, null, `Pending bet must not have profit`);
    }
  }
});

test('v4 Math: AH Quarter-line split settlement and exact payoffs', () => {
  const settle = model.settleAH;

  // -0.75 line (split -0.5 and -1.0) @ 2.00
  assert.equal(settle(2, -0.75, 2.0), 2.0); // full win
  assert.equal(settle(1, -0.75, 2.0), 1.5); // half win (half win @ 2.0 + half push @ 1.0 = 1.5)
  assert.equal(settle(0, -0.75, 2.0), 0.0); // full loss

  // +0.75 line (split +0.5 and +1.0) @ 2.00
  assert.equal(settle(-1, 0.75, 2.0), 0.5); // half loss (half push @ 1.0 + half loss @ 0.0 = 0.5)

  // -0.25 line (split 0 and -0.5) @ 2.00
  assert.equal(settle(1, -0.25, 2.0), 2.0); // full win
  assert.equal(settle(0, -0.25, 2.0), 0.5); // half loss (half push @ 1.0 + half loss @ 0.0 = 0.5)
  assert.equal(settle(-1, -0.25, 2.0), 0.0); // full loss

  // +0.25 line (split 0 and +0.5) @ 2.00
  assert.equal(settle(0, 0.25, 2.0), 1.5); // half win (half push @ 1.0 + half win @ 2.0 = 1.5)
});

test('v4 Math: De-vigging multiplicative normalization', () => {
  const probs = model.removeMargin([2.0, 3.4, 3.8], 'proportional');
  const sum = probs.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1.0) < 1e-9, `Sum was ${sum}`);
  assert.ok(probs[0] > probs[1] && probs[1] > probs[2]);
});

test('v4 Value Sanity: No 1X2 market with Macau odds <= Pinnacle odds is badged as 值博', () => {
  const fixtures = data.macauFixtures || [];
  for (const fx of fixtures) {
    const pinBook = (fx.booksList || []).find(b => b.key === 'pinnacle');
    const o1x2 = fx.macau?.odds_1x2;
    if (pinBook && pinBook['1x2'] && o1x2) {
      const sel = o1x2.best_sel;
      const macauOdds = o1x2[sel.toLowerCase()];
      const pinOdds = pinBook['1x2'][sel.toLowerCase()];
      if (macauOdds <= pinOdds) {
        assert.notEqual(o1x2.badge?.label, '值博',
          `Match ${fx.home} vs ${fx.away} sel ${sel}: Macau ${macauOdds} <= Pin ${pinOdds} must not be 值博`);
      }
    }
  }
});

test('v4 Value Sanity: Hull City 1X2 @ 4.70 (Pinnacle 5.07) is NOT badged as 值博', () => {
  const fixtures = data.macauFixtures || [];
  const hullFx = fixtures.find(f => f.home.includes('Hull') || f.away.includes('Hull'));
  if (hullFx && hullFx.macau?.odds_1x2) {
    const o1x2 = hullFx.macau.odds_1x2;
    assert.notEqual(o1x2.badge?.label, '值博', 'Hull City 1X2 must not be badged as 值博');
    assert.equal(o1x2.badge?.label, '一般', 'Hull City 1X2 should be labeled 一般');
  }
});

