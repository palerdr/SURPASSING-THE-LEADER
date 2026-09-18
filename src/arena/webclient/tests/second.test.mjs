import assert from "node:assert/strict";
import test from "node:test";

import { secondOnClock } from "../src/second.ts";

const NORMAL = Array.from({ length: 60 }, (_, i) => i + 1);
const LEAP_DROPPER = Array.from({ length: 61 }, (_, i) => i + 1);

test("the count opens on 1 and names the second now passing", () => {
  assert.equal(secondOnClock(0, NORMAL), 1);
  assert.equal(secondOnClock(1, NORMAL), 2);
  assert.equal(secondOnClock(36, NORMAL), 37);
  assert.equal(secondOnClock(59, NORMAL), 60);
});

test("the count holds at the last legal second through the gong", () => {
  assert.equal(secondOnClock(60, NORMAL), 60);
  assert.equal(secondOnClock(61, NORMAL), 60);
});

test("Baku dropping in the leap window reads 61 for the extra second", () => {
  assert.equal(secondOnClock(59, LEAP_DROPPER), 60);
  assert.equal(secondOnClock(60, LEAP_DROPPER), 61);
  assert.equal(secondOnClock(61, LEAP_DROPPER), 61);
});

test("the Checker stays capped at 60 through a 61-second turn", () => {
  assert.equal(secondOnClock(60, NORMAL), 60);
});

test("fractional beats are whole seconds heard, and the floor is the first legal second", () => {
  assert.equal(secondOnClock(4.9, NORMAL), 5);
  assert.equal(secondOnClock(-3, NORMAL), 1);
  assert.equal(secondOnClock(0, [5, 6, 7]), 5);
  assert.equal(secondOnClock(0, []), 1);
});
