import assert from "node:assert/strict";
import test from "node:test";

import { paceDelay } from "../src/pace.ts";

test("a request outside the idle window leaves at once", () => {
  for (const since of [0, 300, 349, 1000, 1500, Infinity]) assert.equal(paceDelay(since), 0);
});

test("a request inside the idle window waits until the window closes", () => {
  assert.equal(paceDelay(350), 650);
  assert.equal(paceDelay(430), 570);
  assert.equal(paceDelay(700), 300);
  assert.equal(paceDelay(999), 1);
});
