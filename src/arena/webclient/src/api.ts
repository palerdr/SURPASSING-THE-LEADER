import { hedged } from "./hedge";
import { paceDelay } from "./pace";
import type { Leaderboard, NewGameOptions, Rules, Snapshot, Transcript } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

/** A request with no answer after this long is sent a second time. */
const HEDGE_MS = 900;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // The wait comes first, so the second copy's clock starts at the real send.
  await paced();
  return hedged(
    (signal) => send<T>(path, { ...init, signal }),
    HEDGE_MS,
    (error) => error instanceof ApiError,
  );
}

/** When the hosted server last answered. A local server never sets this. */
let lastAnswerAt = Number.NEGATIVE_INFINITY;

/** Hold a request that would leave inside the host's idle window. */
async function paced(): Promise<void> {
  const wait = paceDelay(performance.now() - lastAnswerAt);
  if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
}

async function send<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  // Only the hosted server sends this header, so local play is never paced.
  if (response.headers.has("x-stl-diagnosis")) lastAnswerAt = performance.now();
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      // A non-JSON error body is still an error; the status carries the meaning.
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

const post = <T>(path: string, body: unknown): Promise<T> =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const getRules = (): Promise<Rules> => request<Rules>("/api/rules");

/** A public read wakes the policy server without loading or changing your game. */
export const warmServer = (): Promise<Rules> =>
  paced().then(() => send<Rules>("/api/rules", { cache: "no-store", credentials: "omit" }));

export const readSession = (): Promise<Snapshot> => request<Snapshot>("/api/session");

/** A page load abandons the previous game and opens a fresh series. */
export const restartSession = (sequence: number): Promise<Snapshot> =>
  post<Snapshot>("/api/session/restart", { sequence });

/** The series so far. Only resolved half-rounds appear, so nothing is hidden here. */
export const getTranscript = (): Promise<Transcript> => request<Transcript>("/api/transcript");

/** The hosted server keeps a leaderboard; a local server answers 404. */
export const getLeaderboard = (): Promise<Leaderboard> => request<Leaderboard>("/api/leaderboard");

/** Post the name a winning score appears under. The reply is the updated board. */
export const postLeaderboardName = (name: string): Promise<Leaderboard> =>
  post<Leaderboard>("/api/leaderboard/name", { name });

export const newSession = (sequence: number, options: NewGameOptions = {}): Promise<Snapshot> =>
  post<Snapshot>("/api/session", { sequence, ...options });

export const begin = (sequence: number): Promise<Snapshot> =>
  post<Snapshot>("/api/session/begin", { sequence });

/** Commit the human's second. Hal only decides once this request arrives. */
export const act = (sequence: number, second: number): Promise<Snapshot> =>
  post<Snapshot>("/api/session/action", { sequence, second });

export const acknowledge = (sequence: number): Promise<Snapshot> =>
  post<Snapshot>("/api/session/ack", { sequence });
