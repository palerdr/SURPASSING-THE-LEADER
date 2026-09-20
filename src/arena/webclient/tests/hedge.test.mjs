import assert from "node:assert/strict";
import test from "node:test";

import { hedged } from "../src/hedge.ts";

const after = (ms, value, fail = false) => (signal) =>
  new Promise((resolve, reject) => {
    const timer = setTimeout(() => (fail ? reject(value) : resolve(value)), ms);
    signal.addEventListener("abort", () => clearTimeout(timer));
  });

test("a prompt answer sends one copy", async () => {
  let sent = 0;
  const answer = await hedged((signal) => { sent += 1; return after(5, "ok")(signal); }, 50);
  assert.equal(answer, "ok");
  await new Promise((resolve) => setTimeout(resolve, 80));
  assert.equal(sent, 1);
});

test("a stalled first copy loses to the second copy", async () => {
  const plans = [after(400, "slow"), after(5, "fast")];
  const started = Date.now();
  const answer = await hedged((signal) => plans.shift()(signal), 30);
  assert.equal(answer, "fast");
  assert.ok(Date.now() - started < 200);
});

test("a refused second copy waits for the first copy's answer", async () => {
  const plans = [after(120, "committed"), after(5, { status: 409 }, true)];
  assert.equal(await hedged((signal) => plans.shift()(signal), 30), "committed");
});

test("a prompt failure is final and sends no second copy", async () => {
  let sent = 0;
  const send = (signal) => { sent += 1; return after(5, { status: 422 }, true)(signal); };
  await assert.rejects(hedged(send, 40), (error) => error.status === 422);
  await new Promise((resolve) => setTimeout(resolve, 70));
  assert.equal(sent, 1);
});

test("two failures report the server's refusal over a dropped connection", async () => {
  const plans = [after(80, new TypeError("network"), true), after(5, { status: 409 }, true)];
  await assert.rejects(
    hedged((signal) => plans.shift()(signal), 20, (error) => error.status !== undefined),
    (error) => error.status === 409,
  );
});
