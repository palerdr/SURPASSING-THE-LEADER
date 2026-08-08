import type { Rules, Snapshot } from "./types";

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
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
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

export const readSession = (): Promise<Snapshot> => request<Snapshot>("/api/session");

export const newSession = (): Promise<Snapshot> => post<Snapshot>("/api/session", {});

export const begin = (sequence: number): Promise<Snapshot> =>
  post<Snapshot>("/api/session/begin", { sequence });

/** Commit the human's second. Hal only decides once this request arrives. */
export const act = (sequence: number, second: number): Promise<Snapshot> =>
  post<Snapshot>("/api/session/action", { sequence, second });

export const acknowledge = (sequence: number): Promise<Snapshot> =>
  post<Snapshot>("/api/session/ack", { sequence });
