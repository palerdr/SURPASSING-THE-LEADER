import assert from "node:assert/strict";
import test from "node:test";

import { humanWon, playerForRole, roleTitle } from "../src/render/identity.ts";

const players = [
  {
    name: "Looks Like Hal",
    character: "baku",
    role: "checker",
    cylinder_seconds: 0,
    ttd_seconds: 0,
    deaths: 0,
    is_human: true,
  },
  {
    name: "Alice",
    character: "hal",
    role: "dropper",
    cylinder_seconds: 0,
    ttd_seconds: 0,
    deaths: 0,
    is_human: false,
  },
];

test("role and character identity ignore display labels", () => {
  const dropper = playerForRole({ players }, "dropper");
  const checker = playerForRole({ players }, "checker");

  assert.equal(dropper.name, "Alice");
  assert.equal(dropper.character, "hal");
  assert.equal(roleTitle(dropper), "Dropper");
  assert.equal(checker.name, "Looks Like Hal");
  assert.equal(checker.character, "baku");
  assert.equal(roleTitle(checker), "Checker");
});

test("winner presentation uses the server-owned seat flag", () => {
  assert.equal(humanWon({ winner_is_human: false }), false);
  assert.equal(humanWon({ winner_is_human: true }), true);
  assert.equal(humanWon({ winner_is_human: null }), false);
});

test("a malformed role mapping fails instead of guessing from names", () => {
  assert.throws(
    () => playerForRole({ players: [players[0], { ...players[1], role: "checker" }] }, "dropper"),
    /exactly one dropper/,
  );
});
