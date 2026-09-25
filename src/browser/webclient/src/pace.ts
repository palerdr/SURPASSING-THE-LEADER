/**
 * Vercel's idle window, measured on the hosted game by sweeping the gap
 * between one answer and the next request. A request sent 0.55 s to 0.80 s
 * after the last answer reached a new server process in 25 of 86 tries, and
 * that process took five to seven seconds to boot. Gaps of 0.3 s, 0.5 s, and
 * 0.85 s to 1.5 s reached a new process in 0 of 108 tries. One "Next game"
 * request left 0.43 s after two parallel reads and met a new process, so the
 * guard opens at 0.35 s.
 */
export const IDLE_WINDOW_MS: readonly [number, number] = [350, 1000];

/** How long a request must wait so that it leaves after the idle window. */
export function paceDelay(sinceLastAnswerMs: number, window = IDLE_WINDOW_MS): number {
  const [opens, closes] = window;
  return sinceLastAnswerMs >= opens && sinceLastAnswerMs < closes
    ? closes - sinceLastAnswerMs
    : 0;
}
