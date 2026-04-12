'use client';

import { useState, useCallback, useEffect, DragEvent } from 'react';
import styles from './IngestPanel.module.css';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

interface JobStatus {
  job_id: string;
  status: 'PENDING' | 'RUNNING' | 'DONE' | 'ERROR';
  created_at: string;
  completed_at?: string;
  error?: string;
}

interface IngestJob {
  job_id: string;
  filename: string;
  status: JobStatus | null;
  polling: boolean;
}

export function IngestPanel() {
  const [dragging, setDragging] = useState(false);
  const [jobs, setJobs] = useState<IngestJob[]>([]);

  async function ingestFile(file: File) {
    const fd = new FormData();
    fd.append('files', file);
    fd.append('urls', '[]');

    const res = await fetch(`${API}/ingest`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    return data.job_id as string;
  }

  async function handleFiles(files: FileList | File[]) {
    const arr = Array.from(files);
    for (const file of arr) {
      const placeholder: IngestJob = {
        job_id: '',
        filename: file.name,
        status: null,
        polling: true,
      };
      setJobs((prev) => [placeholder, ...prev]);

      try {
        const job_id = await ingestFile(file);
        setJobs((prev) =>
          prev.map((j) =>
            j.filename === file.name && j.job_id === ''
              ? { ...j, job_id, polling: true }
              : j,
          ),
        );
      } catch (e) {
        setJobs((prev) =>
          prev.map((j) =>
            j.filename === file.name && j.job_id === ''
              ? { ...j, polling: false, status: { job_id: '', status: 'ERROR', created_at: '', error: String(e) } }
              : j,
          ),
        );
      }
    }
  }

  // Poll active jobs
  useEffect(() => {
    const id = setInterval(async () => {
      const active = jobs.filter((j) => j.job_id && j.polling);
      if (!active.length) return;

      await Promise.all(
        active.map(async (j) => {
          try {
            const res = await fetch(`${API}/jobs/${j.job_id}`);
            if (!res.ok) return;
            const data: JobStatus = await res.json();
            const done = data.status === 'DONE' || data.status === 'ERROR';
            setJobs((prev) =>
              prev.map((p) =>
                p.job_id === j.job_id ? { ...p, status: data, polling: !done } : p,
              ),
            );
          } catch {
            // ignore
          }
        }),
      );
    }, 2000);
    return () => clearInterval(id);
  }, [jobs]);

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragging(false);
      if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
    },
    [],
  );

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(true);
  };

  const onDragLeave = () => setDragging(false);

  function statusColor(s: string) {
    if (s === 'DONE') return 'var(--green)';
    if (s === 'ERROR') return 'var(--red)';
    if (s === 'RUNNING') return 'var(--amber)';
    return 'var(--text-dim)';
  }

  return (
    <div className={styles.panel}>
      <div
        className={`${styles.dropzone} ${dragging ? styles.dragging : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => {
          const inp = document.createElement('input');
          inp.type = 'file';
          inp.multiple = true;
          inp.onchange = () => inp.files && handleFiles(inp.files);
          inp.click();
        }}
      >
        <div className={styles.dzInner}>
          <span className={styles.dzIcon}>{dragging ? '[+]' : '[ ]'}</span>
          <span className={styles.dzText}>
            {dragging ? 'drop to ingest' : 'drag files here or click to browse'}
          </span>
          <span className={styles.dzSub}>pdf · txt · md · docx · html</span>
        </div>
      </div>

      {jobs.length > 0 && (
        <div className={styles.jobs}>
          <div className={styles.jobsLabel}>// jobs</div>
          {jobs.map((j, i) => (
            <div key={i} className={styles.job}>
              <span className={styles.jobFile}>{j.filename}</span>
              {j.job_id && (
                <span className={styles.jobId}>{j.job_id.slice(0, 8)}</span>
              )}
              <span className={styles.jobStatus} style={{ color: j.status ? statusColor(j.status.status) : 'var(--text-dim)' }}>
                {j.status ? j.status.status : 'QUEUING'}
                {j.polling && <span className={styles.blink}>█</span>}
              </span>
              {j.status?.error && (
                <span className={styles.jobError}>{j.status.error}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
