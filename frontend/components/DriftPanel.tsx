'use client';

import { useEffect, useState } from 'react';
import styles from './DriftPanel.module.css';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

interface DriftStatus {
  history: number[];
  consecutive_alerts: number;
  buffer_size: number;
  reindex_triggered: boolean;
}

const CHART_W = 60;
const CHART_H = 10;

function asciiChart(values: number[]): string {
  if (values.length === 0) return '(no data)';

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  // Sample to CHART_W points
  const sampled: number[] =
    values.length <= CHART_W
      ? values
      : Array.from({ length: CHART_W }, (_, i) =>
          values[Math.round((i / (CHART_W - 1)) * (values.length - 1))],
        );

  // Build grid CHART_H rows x sampled.length cols
  const grid: string[][] = Array.from({ length: CHART_H }, () =>
    Array(sampled.length).fill(' '),
  );

  const plotted: number[] = sampled.map((v) =>
    Math.round(((v - min) / range) * (CHART_H - 1)),
  );

  // Connect consecutive points with pipes
  for (let x = 0; x < sampled.length; x++) {
    const row = CHART_H - 1 - plotted[x];
    grid[row][x] = '●';

    if (x > 0) {
      const prevRow = CHART_H - 1 - plotted[x - 1];
      const lo = Math.min(row, prevRow);
      const hi = Math.max(row, prevRow);
      for (let r = lo + 1; r < hi; r++) {
        grid[r][x] = '|';
      }
    }
  }

  const lines: string[] = [];

  // Y-axis labels on the right
  for (let r = 0; r < CHART_H; r++) {
    const label =
      r === 0
        ? (max).toFixed(3)
        : r === CHART_H - 1
        ? (min).toFixed(3)
        : '';
    lines.push(grid[r].join('') + (label ? ' ' + label : ''));
  }

  lines.push('-'.repeat(sampled.length));
  return lines.join('\n');
}

export function DriftPanel() {
  const [drift, setDrift] = useState<DriftStatus | null>(null);
  const [lastReindex, setLastReindex] = useState<string>('');
  const [reindexing, setReindexing] = useState(false);
  const [reindexError, setReindexError] = useState('');

  async function poll() {
    try {
      const r = await fetch(`${API}/drift`);
      if (r.ok) setDrift(await r.json());
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);

  async function triggerReindex() {
    setReindexing(true);
    setReindexError('');
    try {
      const res = await fetch(`${API}/reindex`, { method: 'POST' });
      if (!res.ok) throw new Error(`${res.status}`);
      setLastReindex(new Date().toISOString());
    } catch (e) {
      setReindexError(String(e));
    } finally {
      setReindexing(false);
    }
  }

  async function resetDrift() {
    try {
      await fetch(`${API}/drift/reset`, { method: 'POST' });
      poll();
    } catch {
      // ignore
    }
  }

  const chart = drift ? asciiChart(drift.history) : '(loading...)';

  return (
    <div className={styles.panel}>
      <div className={styles.row}>
        <div className={styles.stat}>
          <span className={styles.statLabel}>consecutive_alerts</span>
          <span
            className={styles.statVal}
            style={{ color: (drift?.consecutive_alerts ?? 0) > 0 ? 'var(--red)' : 'var(--green)' }}
          >
            {drift?.consecutive_alerts ?? '—'}
          </span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>buffer_size</span>
          <span className={styles.statVal}>{drift?.buffer_size ?? '—'}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>reindex_triggered</span>
          <span
            className={styles.statVal}
            style={{ color: drift?.reindex_triggered ? 'var(--amber)' : 'inherit' }}
          >
            {drift?.reindex_triggered ? 'YES' : 'no'}
          </span>
        </div>
        {lastReindex && (
          <div className={styles.stat}>
            <span className={styles.statLabel}>last_reindex</span>
            <span className={styles.statVal}>{lastReindex.replace('T', ' ').slice(0, 19)}</span>
          </div>
        )}
      </div>

      <div className={styles.chartBox}>
        <div className={styles.chartLabel}>// drift score history ({drift?.history.length ?? 0} windows)</div>
        <pre className={styles.chart}>{chart}</pre>
      </div>

      <div className={styles.actions}>
        <button onClick={triggerReindex} disabled={reindexing}>
          {reindexing ? 'reindexing...' : '> reindex'}
        </button>
        <button onClick={resetDrift}>{'> reset drift'}</button>
      </div>

      {reindexError && <div className={styles.error}>{reindexError}</div>}
    </div>
  );
}
