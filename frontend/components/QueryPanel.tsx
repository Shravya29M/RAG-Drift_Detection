'use client';

import { useState, useRef, KeyboardEvent } from 'react';
import styles from './QueryPanel.module.css';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

interface Chunk {
  text: string;
  source: string;
  chunk_index: number;
}

interface QueryResponse {
  answer: string;
  chunks: Chunk[];
  scores: number[];
  latency_ms: number;
}

function SourceChunk({ chunk, score, idx }: { chunk: Chunk; score: number; idx: number }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={styles.chunk}>
      <button className={styles.chunkHeader} onClick={() => setOpen((o) => !o)}>
        <span className={styles.chunkToggle}>{open ? '▼' : '▶'}</span>
        <span className={styles.chunkSource}>{chunk.source}</span>
        <span className={styles.chunkMeta}>
          chunk#{chunk.chunk_index} &nbsp;|&nbsp; score={score.toFixed(4)}
        </span>
      </button>
      {open && <pre className={styles.chunkText}>{chunk.text}</pre>}
    </div>
  );
}

export function QueryPanel() {
  const [input, setInput] = useState('');
  const [answer, setAnswer] = useState('');
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  async function submit() {
    const q = input.trim();
    if (!q || loading) return;
    setLoading(true);
    setError('');
    setAnswer('');
    setResponse(null);

    try {
      const url = `${API}/query`;
      const body = JSON.stringify({ query: q, k: 5 });
      console.log('[query] url:', url);
      console.log('[query] body:', body);
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      });
      if (!res.ok) {
        const text = await res.text();
        setError(`${res.status}: ${text}`);
        return;
      }
      const data: QueryResponse = await res.json();
      // Simulate streaming by revealing the answer character by character
      setResponse(data);
      let i = 0;
      const stream = setInterval(() => {
        i += 4;
        setAnswer(data.answer.slice(0, i));
        if (i >= data.answer.length) {
          setAnswer(data.answer);
          clearInterval(stream);
        }
      }, 16);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function onKey(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') submit();
  }

  return (
    <div className={styles.panel}>
      <div className={styles.promptRow}>
        <span className={styles.prompt}>&gt;&nbsp;</span>
        <input
          ref={inputRef}
          className={styles.input}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKey}
          placeholder="enter query..."
          autoFocus
          disabled={loading}
        />
        <button className={styles.run} onClick={submit} disabled={loading}>
          {loading ? '...' : 'RUN'}
        </button>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      {answer && (
        <div className={styles.answer}>
          <div className={styles.answerLabel}>// answer</div>
          <pre className={styles.answerText}>
            {answer}
            {loading && <span className={styles.cursor}>█</span>}
          </pre>
        </div>
      )}

      {response && (
        <div className={styles.meta}>
          latency: {response.latency_ms.toFixed(1)}ms &nbsp;|&nbsp; chunks: {response.chunks.length}
        </div>
      )}

      {response && response.chunks.length > 0 && (
        <div className={styles.sources}>
          <div className={styles.sourcesLabel}>// sources</div>
          {response.chunks.map((c, i) => (
            <SourceChunk key={i} chunk={c} score={response.scores[i] ?? 0} idx={i} />
          ))}
        </div>
      )}
    </div>
  );
}
