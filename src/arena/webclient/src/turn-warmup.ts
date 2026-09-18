/** Keep the server ready while you play, without submitting an action. */
export function keepServerWarm(warm: () => Promise<unknown>, seconds = Infinity): () => void {
  let pending = false;
  let elapsed = 0;
  const timer = setInterval(async () => {
    elapsed += 15;
    if (elapsed >= seconds - 5) {
      clearInterval(timer);
      return;
    }
    if (pending) return;
    pending = true;
    try {
      await warm();
    } catch {
      // The next scheduled request can retry; this must not block your move.
    } finally {
      pending = false;
    }
  }, 15000);
  return () => clearInterval(timer);
}
