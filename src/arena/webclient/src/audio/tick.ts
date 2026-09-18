// We schedule the sampled clock on the audio timeline and use its output
// timestamp to align the dial with the sound at your speakers.
const SOURCES = {
  tick: "/audio/tick.wav",
  gong: "/audio/gong.wav",
} as const;
type Sample = keyof typeof SOURCES;

let context: AudioContext | null = null;
let samples: Partial<Record<Sample, AudioBuffer>> = {};
let loading: Promise<void> | null = null;
let turn: { length: number; wallStart: number; audioStart: number | null; elapsed: number } | null = null;
let beats: AudioBufferSourceNode[] = [];
let gong: AudioBufferSourceNode | null = null;

function audibleTime(ctx: AudioContext): number {
  const stamp = ctx.getOutputTimestamp?.();
  if (stamp?.performanceTime !== undefined && stamp.performanceTime > 0 && stamp.contextTime !== undefined) {
    return stamp.contextTime + (performance.now() - stamp.performanceTime) / 1000;
  }
  return ctx.currentTime - (ctx.outputLatency || ctx.baseLatency || 0);
}

function stop(node: AudioBufferSourceNode | null): void {
  try { node?.stop(); } catch { /* We may have reached the end of the sample. */ }
  node?.disconnect();
}

function stopSources(keepGong = false): void {
  for (const node of beats) stop(node);
  beats = [];
  if (!keepGong) {
    stop(gong);
    gong = null;
  }
}

function play(ctx: AudioContext, name: Sample, at: number): AudioBufferSourceNode {
  const node = ctx.createBufferSource();
  node.buffer = samples[name]!;
  node.connect(ctx.destination);
  node.onended = () => node.disconnect();
  node.start(at);
  return node;
}

/** Attach loaded audio to the current turn without restarting its countdown. */
function attach(): void {
  if (!turn || turn.audioStart !== null || !context || context.state !== "running") return;
  if (!samples.tick || !samples.gong) return;
  const elapsed = turnSeconds()!;
  if (elapsed >= turn.length) return;
  const start = audibleTime(context) - elapsed;
  turn.audioStart = start;
  // We skip missed beats when loading or resuming takes time.
  const first = Math.max(1, Math.ceil(context.currentTime + 0.01 - start));
  for (let n = first; n < turn.length; n += 1) {
    beats.push(play(context, "tick", start + n));
  }
  if (start + turn.length > context.currentTime) {
    gong = play(context, "gong", start + turn.length);
  }
}

/** Call from a gesture; failed loads and suspended contexts can retry. */
export function unlockTicking(): void {
  try {
    if (!context) {
      context = new AudioContext({ latencyHint: "interactive" });
      context.onstatechange = () => {
        if (context?.state !== "running" && turn) {
          turn.wallStart = performance.now() - turn.elapsed * 1000;
          turn.audioStart = null;
          stopSources();
        }
        attach();
      };
    }
    void context.resume().then(attach).catch(() => {});
    if (!loading && (!samples.tick || !samples.gong)) {
      const ctx = context;
      loading = Promise.all(
        (Object.keys(SOURCES) as Sample[]).map(async (name) => {
          const response = await fetch(SOURCES[name]);
          if (!response.ok) throw new Error(`${name} ${response.status}`);
          return [name, await ctx.decodeAudioData(await response.arrayBuffer())] as const;
        }),
      ).then((loaded) => {
        samples = Object.fromEntries(loaded);
        attach();
      }).catch(() => {
        samples = {};
      }).finally(() => { loading = null; });
    }
  } catch {
    context = null;
  }
}

/** Stop a committed turn. A timeout can leave the closing gong to decay. */
export function cancelTurn(keepGong = false): void {
  stopSources(keepGong);
  turn = null;
}

export function scheduleTurn(length: number, elapsed = 0): void {
  cancelTurn();
  turn = { length, wallStart: performance.now() - elapsed * 1000, audioStart: null, elapsed };
  attach();
}

/** Read audible elapsed time, with a silent clock before audio is available. */
export function turnSeconds(): number | null {
  if (!turn) return null;
  const elapsed = context?.state === "running" && turn.audioStart !== null
    ? audibleTime(context) - turn.audioStart
    : (performance.now() - turn.wallStart) / 1000;
  turn.elapsed = Math.max(turn.elapsed, elapsed, 0);
  return turn.elapsed;
}
