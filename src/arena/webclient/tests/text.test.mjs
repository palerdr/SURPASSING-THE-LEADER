import assert from "node:assert/strict";
import test from "node:test";

import {
  deathLines,
  legalRange,
  resultText,
  squanderedTimeLine,
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
    "squandered time 10s into Alice's ST",
  );
});

test("the death block reports the dose alone, not the chance or the roll", () => {
  assert.deepEqual(deathLines({ death_duration: 100 }), ["injected dose 100s"]);
  assert.deepEqual(deathLines({ death_duration: 300 }), ["injected dose 300s"]);
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
