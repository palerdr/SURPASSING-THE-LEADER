import assert from "node:assert/strict";
import test from "node:test";

import { gainsSince } from "../src/render/gains.ts";

const player = (character, cylinder_seconds, ttd_seconds) => ({
  name: character, character, role: "checker", cylinder_seconds, ttd_seconds, deaths: 0, is_human: false,
});

test("a successful check grows the checker's ST and nothing else", () => {
  const before = [player("hal", 0, 0), player("baku", 40, 0)];
  assert.deepEqual(gainsSince(before, player("baku", 50, 0)), { cylinder: 10, ttd: 0 });
  assert.deepEqual(gainsSince(before, player("hal", 0, 0)), { cylinder: 0, ttd: 0 });
});

test("a failed check empties the vial and grows TTD; only the growth is red", () => {
  const before = [player("baku", 40, 0)];
  assert.deepEqual(gainsSince(before, player("baku", 0, 100)), { cylinder: 0, ttd: 100 });
});

test("an unknown player or an unchanged bar has no red", () => {
  assert.deepEqual(gainsSince([], player("baku", 5, 5)), { cylinder: 0, ttd: 0 });
  assert.deepEqual(gainsSince([player("baku", 5, 5)], player("baku", 5, 5)), { cylinder: 0, ttd: 0 });
});
