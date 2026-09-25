import assert from "node:assert/strict";
import test from "node:test";
import { keepServerWarm } from "../src/turn-warmup.ts";

const flush = () => new Promise((resolve) => setImmediate(resolve));

test("the result screen stays warm while you read, until Continue cancels it", async (t) => {
  t.mock.timers.enable({ apis: ["setInterval"] });
  let reads = 0;
  const stop = keepServerWarm(async () => { reads++; });
  for (let step = 0; step < 12; step++) {
    t.mock.timers.tick(15000);
    await flush();
  }
  assert.equal(reads, 12);
  stop();
  t.mock.timers.tick(60000);
  assert.equal(reads, 12);
});

for (const seconds of [60, 61]) {
  test(`${seconds}-second turn warms before the gong and then stops`, async (t) => {
    t.mock.timers.enable({ apis: ["setInterval"] });
    let reads = 0;
    keepServerWarm(async () => { reads++; }, seconds);
    t.mock.timers.tick(14999);
    assert.equal(reads, 0);
    for (let index = 1; index <= 3; index++) {
      t.mock.timers.tick(index === 1 ? 1 : 15000);
      await flush();
      assert.equal(reads, index);
    }
    t.mock.timers.tick(120000);
    assert.equal(reads, 3);
  });
}

test("an early commit or page exit cancels remaining warm-up reads", async (t) => {
  t.mock.timers.enable({ apis: ["setInterval"] });
  let reads = 0;
  const stop = keepServerWarm(async () => { reads++; }, 60);
  t.mock.timers.tick(15000);
  await flush();
  stop();
  t.mock.timers.tick(120000);
  assert.equal(reads, 1);
});

test("a slow warm-up never overlaps another read or delays cancellation", async (t) => {
  t.mock.timers.enable({ apis: ["setInterval"] });
  let finish;
  let reads = 0;
  const stop = keepServerWarm(() => {
    reads++;
    return new Promise((resolve) => { finish = resolve; });
  }, 61);
  t.mock.timers.tick(30000);
  assert.equal(reads, 1);
  stop();
  finish();
  await flush();
  t.mock.timers.tick(120000);
  assert.equal(reads, 1);
});

test("a failed warm-up permits the next scheduled read", async (t) => {
  t.mock.timers.enable({ apis: ["setInterval"] });
  let reads = 0;
  keepServerWarm(async () => {
    if (++reads === 1) throw new Error("offline");
  }, 60);
  t.mock.timers.tick(15000);
  await flush();
  t.mock.timers.tick(15000);
  await flush();
  assert.equal(reads, 2);
});
