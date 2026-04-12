'use client';

import { useEffect, useState } from 'react';
import styles from './Sidebar.module.css';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

interface DriftStatus {
  history: number[];
  consecutive_alerts: number;
  buffer_size: number;
  reindex_triggered: boolean;
}

export function Sidebar() {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [drift, setDrift] = useState<DriftStatus | null>(null);

  useEffect(() => {
    async function poll() {
      try {
        const r = await fetch(`${API}/healthz`);
        setHealthy(r.ok);
      } catch {
        setHealthy(false);
      }
      try {
        const r = await fetch(`${API}/drift`);
        if (r.ok) setDrift(await r.json());
      } catch {
        // drift unavailable
      }
    }
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);

  const lastScore =
    drift && drift.history.length > 0
      ? drift.history[drift.history.length - 1].toFixed(4)
      : '—';

  return (
    <aside className={styles.sidebar}>
      <div className={styles.header}>RAG_DRIFT</div>

      <div className={styles.section}>
        <div className={styles.label}>SYSTEM</div>
        <div className={styles.row}>
          <span
            className={styles.dot}
            style={{ background: healthy === null ? 'var(--dim)' : healthy ? 'var(--green)' : 'var(--red)' }}
          />
          <span>API {healthy === null ? 'checking' : healthy ? 'healthy' : 'down'}</span>
        </div>
      </div>

      <div className={styles.section}>
        <div className={styles.label}>DRIFT</div>
        <div className={styles.row}>
          <span className={styles.key}>score</span>
          <span className={styles.val}>{lastScore}</span>
        </div>
        <div className={styles.row}>
          <span className={styles.key}>buffer</span>
          <span className={styles.val}>{drift?.buffer_size ?? '—'}</span>
        </div>
        <div className={styles.row}>
          <span className={styles.key}>alerts</span>
          <span
            className={styles.val}
            style={{ color: (drift?.consecutive_alerts ?? 0) > 0 ? 'var(--red)' : 'inherit' }}
          >
            {drift?.consecutive_alerts ?? '—'}
          </span>
        </div>
        <div className={styles.row}>
          <span className={styles.key}>reindex</span>
          <span
            className={styles.val}
            style={{ color: drift?.reindex_triggered ? 'var(--amber)' : 'inherit' }}
          >
            {drift?.reindex_triggered ? 'YES' : 'no'}
          </span>
        </div>
      </div>

      <div className={styles.footer}>
        <span className={styles.dim}>polls every 5s</span>
      </div>
    </aside>
  );
}
