/**
 * The dataset and run the user is currently working on.
 *
 * Persisted to localStorage so a refresh mid-demo does not lose the run — the
 * single most annoying way to break a live walkthrough. Every read is guarded
 * because storage throws in private windows.
 */

import { useCallback, useEffect, useState } from "react";
import type { CostConfig } from "./types";

const KEY = "kairos.session";

export type Session = {
  datasetId: string | null;
  datasetName: string | null;
  runId: string | null;
  costConfig: CostConfig | null;
};

const EMPTY: Session = { datasetId: null, datasetName: null, runId: null, costConfig: null };

function read(): Session {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? { ...EMPTY, ...JSON.parse(raw) } : EMPTY;
  } catch {
    return EMPTY;
  }
}

function write(session: Session): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(session));
  } catch {
    /* private window: the app still works, it just forgets on refresh */
  }
}

const listeners = new Set<(s: Session) => void>();
let current = read();

export function useSession(): [Session, (patch: Partial<Session>) => void] {
  const [session, setSession] = useState(current);

  useEffect(() => {
    listeners.add(setSession);
    return () => {
      listeners.delete(setSession);
    };
  }, []);

  const update = useCallback((patch: Partial<Session>) => {
    current = { ...current, ...patch };
    write(current);
    listeners.forEach((fn) => fn(current));
  }, []);

  return [session, update];
}
