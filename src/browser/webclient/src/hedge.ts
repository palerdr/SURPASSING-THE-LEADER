/**
 * Send a request a second time when the first has no answer after `delayMs`.
 *
 * The host starts a new server process about once a minute and gives it a
 * live request, which then waits six seconds for the process to boot. A second
 * copy of the request reaches a process that already runs. The server commits
 * one copy of a move and refuses the other, and both copies compute the same
 * reveal, so the first success is the answer.
 */
export function hedged<T>(
  send: (signal: AbortSignal) => Promise<T>,
  delayMs: number,
  refusal: (error: unknown) => boolean = () => false,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const controllers: AbortController[] = [];
    const errors: unknown[] = [];
    let pending = 0;
    let done = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const launch = (): void => {
      const controller = new AbortController();
      controllers.push(controller);
      pending += 1;
      send(controller.signal).then(
        (value) => {
          if (done) return;
          done = true;
          if (timer !== null) clearTimeout(timer);
          for (const other of controllers) if (other !== controller) other.abort();
          resolve(value);
        },
        (error) => {
          pending -= 1;
          errors.push(error);
          if (done || pending > 0) return;
          // An early failure is the answer; no second copy follows it.
          done = true;
          if (timer !== null) clearTimeout(timer);
          // The server's refusal says more than a dropped connection does.
          reject(errors.find(refusal) ?? errors[0]);
        },
      );
    };

    launch();
    timer = setTimeout(() => {
      timer = null;
      if (!done) launch();
    }, delayMs);
  });
}
