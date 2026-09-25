import assert from "node:assert/strict";
import test from "node:test";

async function setup(t, { delayed = false, fail = false } = {}) {
  let now = 1000;
  const nodes = [];
  let release;
  const loaded = delayed ? new Promise((resolve) => { release = resolve; }) : Promise.resolve();
  let failed = fail;
  let ctx;
  t.mock.method(performance, "now", () => now);
  t.mock.method(globalThis, "fetch", async (url) => {
    await loaded;
    if (failed) return { ok: false, status: 503 };
    return { ok: true, arrayBuffer: async () => url };
  });
  const previous = globalThis.AudioContext;
  globalThis.AudioContext = class {
    state = "running";
    outputLatency = 0.15;
    destination = {};
    constructor() { ctx = this; }
    get currentTime() { return now / 1000 + 0.15; }
    getOutputTimestamp() { return { contextTime: now / 1000, performanceTime: now }; }
    async resume() { this.state = "running"; }
    async decodeAudioData(name) { return { name }; }
    createBufferSource() {
      const node = {
        connect() {}, disconnect() {},
        start(at) { this.at = at; }, stop() { this.stopped = true; },
      };
      nodes.push(node);
      return node;
    }
  };
  t.after(() => { globalThis.AudioContext = previous; });
  const clock = await import(`../src/audio/tick.ts?case=${t.name}`);
  const flush = () => new Promise((resolve) => setImmediate(resolve));
  clock.unlockTicking();
  await flush();
  return {
    clock, nodes, ctx, flush,
    advance(seconds) { now += seconds * 1000; },
    async load() { failed = false; release?.(); await flush(); },
  };
}

for (const length of [60, 61]) {
  test(`${length}-second turn has one closing gong with no overlapping tick`, async (t) => {
    const { clock, nodes, advance } = await setup(t);
    clock.scheduleTurn(length);
    assert.equal(nodes.length, length);
    assert.equal(nodes.at(-1).buffer.name, "/audio/gong.wav");
    assert.equal(nodes.at(-1).at, 1 + length);
    assert.equal(nodes.at(-2).at, length);
    advance(length - 0.01);
    assert.ok(clock.turnSeconds() < length);
    advance(0.01);
    assert.equal(clock.turnSeconds(), length);
    clock.cancelTurn(true);
    assert.ok(nodes.slice(0, -1).every((node) => node.stopped));
    assert.ok(!nodes.at(-1).stopped);
  });
}

test("late sample loading joins the current second without restarting or bursting", async (t) => {
  const { clock, nodes, advance, load } = await setup(t, { delayed: true });
  clock.scheduleTurn(61);
  advance(2.4);
  assert.equal(clock.turnSeconds(), 2.4);
  await load();
  assert.equal(nodes[0].at, 4);
  assert.equal(nodes.at(-1).at, 62);
  assert.equal(clock.turnSeconds(), 2.4);
});

test("a failed audio load can retry on the next gesture", async (t) => {
  const { clock, nodes, advance, load, flush } = await setup(t, { fail: true });
  clock.scheduleTurn(60);
  advance(3);
  await load();
  clock.unlockTicking();
  await flush();
  assert.equal(nodes[0].at, 5);
  assert.equal(nodes.at(-1).at, 61);
});

test("committing before the deadline cancels the gong even within the last 50 ms", async (t) => {
  const { clock, nodes, advance } = await setup(t);
  clock.scheduleTurn(60);
  advance(59.98);
  clock.cancelTurn();
  assert.ok(nodes.every((node) => node.stopped));
  assert.equal(clock.turnSeconds(), null);
});

test("a cancelled turn cannot start sound after its fetch completes", async (t) => {
  const { clock, nodes, load } = await setup(t, { delayed: true });
  clock.scheduleTurn(60);
  clock.cancelTurn();
  await load();
  assert.equal(nodes.length, 0);
});

test("resume replaces frozen sources and retains the elapsed turn", async (t) => {
  const { clock, nodes, advance, ctx, flush } = await setup(t);
  clock.scheduleTurn(61);
  advance(10);
  assert.equal(clock.turnSeconds(), 10);
  ctx.state = "suspended";
  ctx.onstatechange();
  assert.ok(nodes.every((node) => node.stopped));
  advance(5);
  assert.equal(clock.turnSeconds(), 15);
  const count = nodes.length;
  clock.unlockTicking();
  await flush();
  assert.equal(clock.turnSeconds(), 15);
  assert.equal(nodes[count].at, 17);
  assert.equal(nodes.at(-1).at, 62);
});
