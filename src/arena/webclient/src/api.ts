import type { NewGameOptions, Rules, Snapshot, Transcript } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
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
  request<Rules>("/api/rules", { cache: "no-store", credentials: "omit" });

export const readSession = (): Promise<Snapshot> => request<Snapshot>("/api/session");

/** A page load abandons the previous game and opens a fresh series. */
export const restartSession = (sequence: number): Promise<Snapshot> =>
  post<Snapshot>("/api/session/restart", { sequence });

/** The series so far. Only resolved half-rounds appear, so nothing is hidden here. */
export const getTranscript = (): Promise<Transcript> => request<Transcript>("/api/transcript");

export const newSession = (sequence: number, options: NewGameOptions = {}): Promise<Snapshot> =>
  post<Snapshot>("/api/session", { sequence, ...options });

export const begin = (sequence: number): Promise<Snapshot> =>
  post<Snapshot>("/api/session/begin", { sequence });

/** Commit the human's second. Hal only decides once this request arrives. */
export const act = (sequence: number, second: number): Promise<Snapshot> =>
  post<Snapshot>("/api/session/action", { sequence, second });

export const acknowledge = (sequence: number): Promise<Snapshot> =>
  post<Snapshot>("/api/session/ack", { sequence });
