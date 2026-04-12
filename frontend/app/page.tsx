'use client';

import { useState } from 'react';
import { QueryPanel } from '@/components/QueryPanel';
import { IngestPanel } from '@/components/IngestPanel';
import { DriftPanel } from '@/components/DriftPanel';
import styles from './page.module.css';

type Tab = 'query' | 'ingest' | 'drift';

export default function Home() {
  const [tab, setTab] = useState<Tab>('query');

  return (
    <div className={styles.page}>
      <div className={styles.tabs}>
        {(['query', 'ingest', 'drift'] as Tab[]).map((t) => (
          <button
            key={t}
            className={`${styles.tab} ${tab === t ? styles.active : ''}`}
            onClick={() => setTab(t)}
          >
            {t === 'query' && '> query'}
            {t === 'ingest' && '> ingest'}
            {t === 'drift' && '> drift'}
          </button>
        ))}
      </div>
      <div className={styles.panel}>
        {tab === 'query' && <QueryPanel />}
        {tab === 'ingest' && <IngestPanel />}
        {tab === 'drift' && <DriftPanel />}
      </div>
    </div>
  );
}
