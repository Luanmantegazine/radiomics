import { useCallback, useEffect, useRef, useState } from "react";

import { errorMessage } from "../api/client";

export interface ApiState<T> {
  data: T | undefined;
  error: string | undefined;
  loading: boolean;
  reload: () => void;
}

/** Fetch on mount / when ``key`` changes; optionally poll every ``intervalMs``. */
export function useApi<T>(
  fetcher: () => Promise<T>,
  key: unknown[],
  intervalMs?: number,
): ApiState<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<string>();
  const [loading, setLoading] = useState(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const load = useCallback(() => {
    let cancelled = false;
    fetcherRef
      .current()
      .then((value) => {
        if (!cancelled) {
          setData(value);
          setError(undefined);
        }
      })
      .catch((reason) => !cancelled && setError(errorMessage(reason)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, key);

  useEffect(() => {
    setLoading(true);
    const cancel = load();
    if (!intervalMs) return cancel;
    const timer = setInterval(load, intervalMs);
    return () => {
      cancel();
      clearInterval(timer);
    };
  }, [load, intervalMs]);

  return { data, error, loading, reload: load };
}
