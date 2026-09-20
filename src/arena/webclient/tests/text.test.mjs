import assert from "node:assert/strict";
import test from "node:test";

import {
  deathLines,
  legalRange,
  resultText,
  scoreText,
  squanderedTimeLine,
  standingText,
  tallyText,
} from "../src/render/text.ts";

test("result wording follows the terminal interface, with commas", () => {
  assert.equal(resultText("check_success"), "CHECK SUCCESS");
  assert.equal(resultText("check_fail_survived"), "CHECK FAILED, died, revived");
  assert.equal(resultText("overflow_died"), "ST OVERFLOW, died permanently");
});

test("the squandered time is shown as a number, not a calculation", () => {
  assert.equal(
    squanderedTimeLine({ st_gained: 10, checker: "Alice" }),
    "Squandered time 10s into Alice's ST",
  );
});

test("the death block reports the dose alone, not the chance or the roll", () => {
  assert.deepEqual(deathLines({ death_duration: 100 }), ["Injected dose 100s"]);
  assert.deepEqual(deathLines({ death_duration: 300 }), ["Injected dose 300s"]);
  assert.deepEqual(deathLines({ death_duration: 0 }), []);
});

test("legal seconds are summarised as a range only when contiguous", () => {
  assert.equal(legalRange([1, 2, 3, 4]), "1–4");
  assert.equal(legalRange([1, 3]), "1, 3");
  assert.equal(legalRange([]), "none");
});

test("the series tally reads from the human's side", () => {
  assert.equal(tallyText({ human_wins: 3, hal_wins: 1, no_winner: 0, stopped: 0 }), "3–1");
  assert.equal(tallyText({ human_wins: 0, hal_wins: 2, no_winner: 1, stopped: 1 }), "0–2 · 2 undecided");
});

test("the leaderboard tells the player where their best win stands", () => {
  assert.equal(scoreText(183), "183");
  assert.equal(scoreText(182.55), "182.6");
  assert.equal(
    standingText({ your_rank: null, your_name: "Baku", your_score: null }),
    "You must surpass the leader to hold a place.",
  );
  assert.equal(
    standingText({ your_rank: null, your_name: null, your_score: 183 }),
    "Your best win has 183 seconds of life left. Enter a name to post it.",
  );
  assert.equal(
    standingText({ your_rank: 4, your_name: "Baku", your_score: 183 }),
    "You hold rank 4 with 183 seconds of life left.",
  );
});
