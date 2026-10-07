/**
 * The second the count names once `beats` whole seconds of the turn have
 * sounded. The count reads the second now passing: 1 when the clock opens,
 * k + 1 after beat k, held at the last legal second until the gong. Commit
 * sends this number, so what you read is what you play.
 *
 * `legal` comes from the server; only the engine knows the Checker stays
 * capped at 60 through a 61-second leap turn.
 */
export function secondOnClock(beats: number, legal: readonly number[]): number {
  const first = legal[0] ?? 1;
  const last = legal[legal.length - 1] ?? first;
  const passing = Math.floor(Math.max(0, beats)) + 1;
  return Math.min(last, Math.max(first, passing));
}

/**
 * The second a commit plays. An empty field plays `onClock`, the second the
 * count names. A typed second plays at once, without the wait for the clock.
 * The result is null when the field holds anything other than a second in
 * the server's `legal` list.
 */
export function committedSecond(
  typed: string,
  onClock: number,
  legal: readonly number[],
): number | null {
  const text = typed.trim();
  if (text === "") return onClock;
  if (!/^\d+$/.test(text)) return null;
  const second = Number(text);
  return legal.includes(second) ? second : null;
}
