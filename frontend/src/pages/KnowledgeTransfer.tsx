import { useState, useMemo, useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Search, RefreshCw, SlidersHorizontal, X, Eye, Cpu,
  Layers, Activity, AlertTriangle, Database, ArrowRightLeft,
  BarChart3, TrendingUp, Grid3X3, PieChart as PieChartIcon,
  ArrowRight, Clock, CheckCircle2, XCircle, Target,
  GitBranch, Network,
} from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, LineChart, Line, PieChart,
  Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ScatterChart, Scatter, AreaChart, Area,
} from 'recharts';

import {
  fetchKnowledgeTransfers,
  fetchKnowledgeTransferStatistics,
} from '../api/knowledgeTransfer';
import { Card } from '../components/ui/Card';
import { StatusBadge } from '../components/ui/StatusBadge';
import type { KnowledgeTransferResponse, KnowledgeTransferStatistics } from '../types';

const PAGE_SIZES = [5, 10, 20, 50];
const AUTO_REFRESH_MS = 5000;

function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}

const MODALITY_LABELS: Record<string, string> = {
  image: 'Image',
  text: 'Text',
  audio: 'Audio',
  sensor: 'Sensor',
  visual: 'Image',
  acoustic: 'Audio',
  linguistic: 'Text',
  multimodal: 'Multimodal',
};

const MODALITY_COLORS: Record<string, string> = {
  image: 'var(--color-primary)',
  text: 'var(--color-secondary)',
  audio: '#a78bfa',
  sensor: '#f87171',
  visual: 'var(--color-primary)',
  acoustic: 'var(--color-secondary)',
  linguistic: '#a78bfa',
  multimodal: '#f87171',
};

const CHART_COLORS = [
  'var(--color-primary)',
  'var(--color-secondary)',
  '#a78bfa',
  '#f87171',
  '#34d399',
  '#fbbf24',
];

const TRANSFER_STRATEGY_LABELS: Record<string, string> = {
  direct: 'Direct Mapping',
  sequential: 'Sequential',
  graph_based: 'Graph-Based',
  prototype_to_prototype: 'Prototype → Prototype',
  feature_to_prototype: 'Feature → Prototype',
  cross_modal: 'Cross-Modal',
};

const STATUS_STYLES: Record<string, string> = {
  completed: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  running: 'bg-primary-container/20 text-primary border-primary/30',
  failed: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
  pending: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
};

function getModalityLabel(m: string): string {
  return MODALITY_LABELS[m] || m;
}

function getTransferStrategyLabel(s: string): string {
  return TRANSFER_STRATEGY_LABELS[s] || s;
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

type SortField = 'transfer_id' | 'source_client' | 'target_client' | 'source_modality' | 'target_modality' | 'transfer_strategy' | 'similarity_score' | 'confidence' | 'transfer_loss' | 'transfer_status' | 'communication_round' | 'execution_time' | 'created_at';

/* ====================================================================== */
/*  Skeleton                                                               */
/* ====================================================================== */

function SkeletonRow() {
  return (
    <tr className="animate-pulse border-b border-outline-variant/30">
      {Array.from({ length: 8 }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <div className="h-4 rounded bg-surface-container-high" style={{ width: `${50 + Math.random() * 40}%` }} />
        </td>
      ))}
    </tr>
  );
}

function SkeletonCard() {
  return (
    <Card className="p-5 flex flex-col gap-3">
      <div className="h-3 w-24 rounded-full bg-surface-container-high animate-pulse" />
      <div className="h-5 w-full rounded-lg bg-surface-container-high animate-pulse" />
    </Card>
  );
}

/* ====================================================================== */
/*  Empty & Error states                                                   */
/* ====================================================================== */

function EmptyState({ onRefresh }: { onRefresh: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <div className="w-16 h-16 rounded-2xl bg-surface-container-high flex items-center justify-center">
        <ArrowRightLeft size={32} className="text-outline" />
      </div>
      <h3 className="text-lg font-display font-bold text-on-surface">No transfers found</h3>
      <p className="text-sm text-outline max-w-sm text-center">
        No knowledge transfers match your search criteria. Try adjusting your filters.
      </p>
      <button
        onClick={onRefresh}
        className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity cursor-pointer"
      >
        <RefreshCw size={16} />
        Refresh
      </button>
    </div>
  );
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <div className="w-14 h-14 rounded-2xl bg-rose-500/10 flex items-center justify-center">
        <AlertTriangle size={28} className="text-rose-400" />
      </div>
      <h3 className="text-lg font-display font-bold text-on-surface">Failed to load transfers</h3>
      <p className="text-sm text-outline max-w-md text-center">{message}</p>
      <button
        onClick={onRetry}
        className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity cursor-pointer"
      >
        <RefreshCw size={16} />
        Try Again
      </button>
    </div>
  );
}

/* ====================================================================== */
/*  Detail Panel                                                           */
/* ====================================================================== */

function DetailPanel({
  transfer: t,
  onClose,
}: {
  transfer: KnowledgeTransferResponse;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/40 backdrop-blur-sm">
      <Card className="w-full max-w-lg h-full overflow-y-auto rounded-none rounded-l-2xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center">
              <ArrowRightLeft size={20} className="text-primary" />
            </div>
            <div>
              <h2 className="text-lg font-display font-bold text-on-surface">Transfer Details</h2>
              <p className="text-[10px] font-mono text-outline uppercase">{t.transfer_id}</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-surface-container-high transition-colors cursor-pointer">
            <X size={20} className="text-outline" />
          </button>
        </div>

        {/* Transfer Metadata */}
        <div className="space-y-4 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Transfer Metadata</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Source Client</p>
              <p className="text-sm font-semibold text-on-surface">{t.source_client}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Target Client</p>
              <p className="text-sm font-semibold text-on-surface">{t.target_client}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Source Modality</p>
              <p className="text-sm font-semibold text-on-surface">{getModalityLabel(t.source_modality)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Target Modality</p>
              <p className="text-sm font-semibold text-on-surface">{getModalityLabel(t.target_modality)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Transfer Strategy</p>
              <p className="text-sm font-semibold text-on-surface">{getTransferStrategyLabel(t.transfer_strategy)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Transfer Status</p>
              <StatusBadge status={t.transfer_status} />
            </div>
          </div>
        </div>

        {/* Alignment Information */}
        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Alignment Information</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Cross-Modal Mapping</p>
              <p className="text-sm font-semibold text-on-surface">{t.cross_modal_mapping || '—'}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Alignment Method</p>
              <p className="text-sm font-semibold text-on-surface">{t.alignment_method || '—'}</p>
            </div>
          </div>
        </div>

        {/* Similarity Metrics */}
        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Similarity Metrics</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Similarity Score</p>
              <p className="text-lg font-display font-bold text-primary">{(t.similarity_score * 100).toFixed(1)}%</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Confidence</p>
              <p className="text-lg font-display font-bold text-secondary">{(t.confidence * 100).toFixed(1)}%</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Transfer Loss</p>
              <p className="text-lg font-display font-bold text-amber-400">{t.transfer_loss.toFixed(4)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Execution Time</p>
              <p className="text-lg font-display font-bold text-on-surface">{t.execution_time.toFixed(1)}s</p>
            </div>
          </div>
        </div>

        {/* Associated Prototypes */}
        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Associated Prototypes</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Source Prototype</p>
              <p className="text-sm font-semibold text-on-surface">{t.source_prototype}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Target Prototype</p>
              <p className="text-sm font-semibold text-on-surface">{t.target_prototype}</p>
            </div>
          </div>
        </div>

        {/* History */}
        <div className="space-y-3">
          <h3 className="text-sm font-bold text-on-surface">Knowledge Transfer History</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Communication Round</p>
              <p className="text-sm font-semibold text-on-surface">Round #{t.communication_round}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Created Time</p>
              <p className="text-sm font-semibold text-on-surface">{formatTimestamp(t.created_at)}</p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

/* ====================================================================== */
/*  Cross-Modal Transfer Graph                                             */
/* ====================================================================== */

function CrossModalTransferGraph({ transfers }: { transfers: KnowledgeTransferResponse[] }) {
  const data = useMemo(() => {
    const pairs: Record<string, { source: string; target: string; count: number }> = {};
    transfers.forEach((t) => {
      const key = `${t.source_modality}→${t.target_modality}`;
      if (!pairs[key]) {
        pairs[key] = { source: getModalityLabel(t.source_modality), target: getModalityLabel(t.target_modality), count: 0 };
      }
      pairs[key].count += 1;
    });
    return Object.values(pairs).sort((a, b) => b.count - a.count);
  }, [transfers]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No transfer data available</p>;

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={data} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis type="number" tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} allowDecimals={false} />
        <YAxis type="category" dataKey="source" tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} width={90} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
          formatter={(value: number) => [`${value} transfers`]}
        />
        <Bar dataKey="count" fill="var(--color-primary)" radius={[0, 4, 4, 0]} name="Transfers">
          {data.map((_, i) => (
            <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ====================================================================== */
/*  Similarity Heatmap                                                     */
/* ====================================================================== */

function SimilarityHeatmap({ transfers }: { transfers: KnowledgeTransferResponse[] }) {
  const data = useMemo(() => {
    const pairs: Record<string, { source: string; target: string; avgSimilarity: number; count: number }> = {};
    transfers.forEach((t) => {
      const key = `${t.source_modality}→${t.target_modality}`;
      if (!pairs[key]) {
        pairs[key] = { source: t.source_modality, target: t.target_modality, avgSimilarity: 0, count: 0 };
      }
      pairs[key].avgSimilarity += t.similarity_score;
      pairs[key].count += 1;
    });
    return Object.entries(pairs).map(([k, v]) => ({
      pair: k,
      avgSimilarity: parseFloat((v.avgSimilarity / v.count).toFixed(3)),
      count: v.count,
    }));
  }, [transfers]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No similarity data available</p>;

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="pair" tick={{ fill: 'var(--color-outline)', fontSize: 9 }} axisLine={false} tickLine={false} angle={-20} textAnchor="end" height={60} />
        <YAxis domain={[0, 1]} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
          formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Avg Similarity']}
        />
        <Bar dataKey="avgSimilarity" radius={[4, 4, 0, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ====================================================================== */
/*  Transfer Timeline                                                      */
/* ====================================================================== */

function TransferTimeline({ transfers }: { transfers: KnowledgeTransferResponse[] }) {
  const data = useMemo(() => {
    const grouped: Record<number, { round: number; avgSimilarity: number; count: number }> = {};
    transfers.forEach((t) => {
      const r = t.communication_round;
      if (!grouped[r]) {
        grouped[r] = { round: r, avgSimilarity: 0, count: 0 };
      }
      grouped[r].avgSimilarity += t.similarity_score;
      grouped[r].count += 1;
    });
    return Object.values(grouped)
      .map((g) => ({
        round: `R${g.round}`,
        avgSimilarity: parseFloat((g.avgSimilarity / g.count * 100).toFixed(1)),
        count: g.count,
      }))
      .sort((a, b) => parseInt(a.round.slice(1)) - parseInt(b.round.slice(1)));
  }, [transfers]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No timeline data available</p>;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="round" tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis domain={[0, 100]} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${v}%`} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
          formatter={(value: number, name: string) => [
            name === 'avgSimilarity' ? `${value}%` : value,
            name === 'avgSimilarity' ? 'Avg Similarity' : 'Transfers',
          ]}
        />
        <Line type="monotone" dataKey="avgSimilarity" stroke="var(--color-primary)" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} name="avgSimilarity" />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ====================================================================== */
/*  Transfer Success Distribution                                          */
/* ====================================================================== */

function TransferSuccessDistribution({ transfers }: { transfers: KnowledgeTransferResponse[] }) {
  const data = useMemo(() => {
    const counts: Record<string, number> = {};
    transfers.forEach((t) => {
      const status = t.transfer_status || 'unknown';
      counts[status] = (counts[status] || 0) + 1;
    });
    return Object.entries(counts).map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value }));
  }, [transfers]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No status data available</p>;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
          {data.map((entry) => {
            const idx = entry.name === 'Completed' ? 0 : entry.name === 'Failed' ? 3 : entry.name === 'Running' ? 1 : 5;
            return <Cell key={entry.name} fill={CHART_COLORS[idx % CHART_COLORS.length]} />;
          })}
        </Pie>
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}

/* ====================================================================== */
/*  Transfer Loss Curve                                                    */
/* ====================================================================== */

function TransferLossCurve({ transfers }: { transfers: KnowledgeTransferResponse[] }) {
  const data = useMemo(() => {
    return [...transfers]
      .sort((a, b) => a.communication_round - b.communication_round)
      .map((t, i) => ({
        index: i,
        round: `R${t.communication_round}`,
        loss: parseFloat(t.transfer_loss.toFixed(4)),
        id: t.transfer_id.slice(-6),
      }));
  }, [transfers]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No loss data available</p>;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="round" tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
          formatter={(value: number) => [value.toFixed(4), 'Transfer Loss']}
        />
        <Area type="monotone" dataKey="loss" stroke="var(--color-secondary)" fill="var(--color-secondary)" fillOpacity={0.15} strokeWidth={2} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* ====================================================================== */
/*  Modality Interaction Matrix                                            */
/* ====================================================================== */

function ModalityInteractionMatrix({ transfers }: { transfers: KnowledgeTransferResponse[] }) {
  const matrix = useMemo(() => {
    const modalities = Array.from(new Set(transfers.flatMap((t) => [t.source_modality, t.target_modality]))).sort();
    const pairs: Record<string, { count: number; totalSimilarity: number }> = {};
    transfers.forEach((t) => {
      const key = `${t.source_modality}→${t.target_modality}`;
      if (!pairs[key]) pairs[key] = { count: 0, totalSimilarity: 0 };
      pairs[key].count += 1;
      pairs[key].totalSimilarity += t.similarity_score;
    });
    const rows = modalities.map((src) => {
      const row: Record<string, string | number> = { source: src };
      modalities.forEach((tgt) => {
        const key = `${src}→${tgt}`;
        const p = pairs[key];
        row[tgt] = p ? parseFloat((p.totalSimilarity / p.count).toFixed(3)) : 0;
      });
      return row;
    });
    return { modalities, rows };
  }, [transfers]);

  if (matrix.modalities.length === 0) return <p className="text-xs text-outline text-center py-4">No interaction data available</p>;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr>
            <th className="p-1.5 text-outline text-[10px] font-bold uppercase" />
            {matrix.modalities.map((m) => (
              <th key={m} className="p-1.5 text-outline text-[10px] font-mono font-bold text-center">{getModalityLabel(m)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.rows.map((row) => (
            <tr key={row.source as string}>
              <td className="p-1.5 text-outline text-[10px] font-mono font-bold">{getModalityLabel(row.source as string)}</td>
              {matrix.modalities.map((mod) => {
                const val = row[mod] as number;
                const intensity = Math.round(Math.min(val, 1) * 100);
                return (
                  <td
                    key={mod}
                    className="p-1.5 text-center font-mono font-bold"
                    style={{
                      backgroundColor: `color-mix(in srgb, var(--color-primary) ${intensity}%, var(--color-surface-container))`,
                      color: intensity > 50 ? 'white' : 'var(--color-on-surface)',
                    }}
                  >
                    {val.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ====================================================================== */
/*  Main Knowledge Transfer Page                                           */
/* ====================================================================== */

export const KnowledgeTransfer = () => {
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [strategyFilter, setStrategyFilter] = useState('');
  const [sourceClientFilter, setSourceClientFilter] = useState('');
  const [targetClientFilter, setTargetClientFilter] = useState('');
  const [sourceModalityFilter, setSourceModalityFilter] = useState('');
  const [targetModalityFilter, setTargetModalityFilter] = useState('');
  const [sortField, setSortField] = useState<SortField>('created_at');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(10);
  const [selectedTransfer, setSelectedTransfer] = useState<KnowledgeTransferResponse | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const {
    data: listResponse,
    isLoading,
    isError,
    error,
    refetch,
    isRefetching,
    dataUpdatedAt,
  } = useQuery({
    queryKey: ['knowledge-transfers'],
    queryFn: fetchKnowledgeTransfers,
    refetchInterval: AUTO_REFRESH_MS,
    retry: 2,
    staleTime: 3000,
  });

  const {
    data: statsResponse,
    isLoading: statsLoading,
  } = useQuery({
    queryKey: ['knowledge-transfer-statistics'],
    queryFn: fetchKnowledgeTransferStatistics,
    refetchInterval: AUTO_REFRESH_MS,
    retry: 2,
    staleTime: 3000,
  });

  const [lastRefresh, setLastRefresh] = useState<string>('');
  useEffect(() => {
    if (dataUpdatedAt) {
      setLastRefresh(new Date(dataUpdatedAt).toLocaleTimeString());
    }
  }, [dataUpdatedAt]);

  const allTransfers: KnowledgeTransferResponse[] = listResponse?.data ?? [];
  const statistics: KnowledgeTransferStatistics | null = statsResponse?.data ?? null;

  const uniqueStatuses = useMemo(() => {
    return Array.from(new Set(allTransfers.map((t) => t.transfer_status))).sort();
  }, [allTransfers]);

  const uniqueStrategies = useMemo(() => {
    return Array.from(new Set(allTransfers.map((t) => t.transfer_strategy))).sort();
  }, [allTransfers]);

  const uniqueSourceClients = useMemo(() => {
    return Array.from(new Set(allTransfers.map((t) => t.source_client))).sort();
  }, [allTransfers]);

  const uniqueTargetClients = useMemo(() => {
    return Array.from(new Set(allTransfers.map((t) => t.target_client))).sort();
  }, [allTransfers]);

  const uniqueSourceModalities = useMemo(() => {
    return Array.from(new Set(allTransfers.map((t) => t.source_modality))).sort();
  }, [allTransfers]);

  const uniqueTargetModalities = useMemo(() => {
    return Array.from(new Set(allTransfers.map((t) => t.target_modality))).sort();
  }, [allTransfers]);

  const filtered = useMemo(() => {
    let list = allTransfers;

    if (search) {
      const q = search.toLowerCase();
      list = list.filter(
        (t) =>
          t.transfer_id.toLowerCase().includes(q) ||
          t.source_client.toLowerCase().includes(q) ||
          t.target_client.toLowerCase().includes(q) ||
          t.source_modality.toLowerCase().includes(q) ||
          t.target_modality.toLowerCase().includes(q) ||
          t.transfer_strategy.toLowerCase().includes(q) ||
          t.transfer_status.toLowerCase().includes(q),
      );
    }

    if (statusFilter) list = list.filter((t) => t.transfer_status === statusFilter);
    if (strategyFilter) list = list.filter((t) => t.transfer_strategy === strategyFilter);
    if (sourceClientFilter) list = list.filter((t) => t.source_client === sourceClientFilter);
    if (targetClientFilter) list = list.filter((t) => t.target_client === targetClientFilter);
    if (sourceModalityFilter) list = list.filter((t) => t.source_modality === sourceModalityFilter);
    if (targetModalityFilter) list = list.filter((t) => t.target_modality === targetModalityFilter);

    list.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case 'transfer_id': cmp = a.transfer_id.localeCompare(b.transfer_id); break;
        case 'source_client': cmp = a.source_client.localeCompare(b.source_client); break;
        case 'target_client': cmp = a.target_client.localeCompare(b.target_client); break;
        case 'source_modality': cmp = a.source_modality.localeCompare(b.source_modality); break;
        case 'target_modality': cmp = a.target_modality.localeCompare(b.target_modality); break;
        case 'transfer_strategy': cmp = a.transfer_strategy.localeCompare(b.transfer_strategy); break;
        case 'similarity_score': cmp = a.similarity_score - b.similarity_score; break;
        case 'confidence': cmp = a.confidence - b.confidence; break;
        case 'transfer_loss': cmp = a.transfer_loss - b.transfer_loss; break;
        case 'transfer_status': cmp = a.transfer_status.localeCompare(b.transfer_status); break;
        case 'communication_round': cmp = a.communication_round - b.communication_round; break;
        case 'execution_time': cmp = a.execution_time - b.execution_time; break;
        case 'created_at': cmp = a.created_at.localeCompare(b.created_at); break;
        default: cmp = 0;
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return list;
  }, [allTransfers, search, statusFilter, strategyFilter, sourceClientFilter, targetClientFilter, sourceModalityFilter, targetModalityFilter, sortField, sortDir]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(page, totalPages - 1);
  const paged = filtered.slice(safePage * pageSize, (safePage + 1) * pageSize);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDir('asc');
    }
  };

  const handleRefresh = () => {
    refetch();
  };

  function SortIcon({ field }: { field: SortField }) {
    if (sortField !== field) return <span className="text-outline/30 ml-1">&#8597;</span>;
    return <span className="text-primary ml-1">{sortDir === 'asc' ? '&#8593;' : '&#8595;'}</span>;
  }

  const hasFilters = search || statusFilter || strategyFilter || sourceClientFilter || targetClientFilter || sourceModalityFilter || targetModalityFilter;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">Knowledge Transfer</h1>
          <p className="text-sm text-outline">Cross-modal knowledge transfer across clients and modalities</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[10px] text-outline">
            {lastRefresh ? `Last updated: ${lastRefresh}` : ''}
          </span>
          <button
            onClick={handleRefresh}
            disabled={isRefetching}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-container-high text-on-surface text-sm font-semibold hover:bg-surface-container-highest transition-colors disabled:opacity-50 cursor-pointer"
          >
            <RefreshCw size={16} className={cn(isRefetching && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>

      {/* Search & Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-outline" />
          <input
            ref={searchInputRef}
            type="text"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0); }}
            placeholder="Search by ID, client, modality, strategy..."
            className="w-full pl-9 pr-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm placeholder:text-outline/50 focus:outline-none focus:border-primary transition-colors"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Statuses</option>
          {uniqueStatuses.map((s) => (
            <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>
          ))}
        </select>
        <select
          value={strategyFilter}
          onChange={(e) => { setStrategyFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Strategies</option>
          {uniqueStrategies.map((s) => (
            <option key={s} value={s}>{getTransferStrategyLabel(s)}</option>
          ))}
        </select>
        <select
          value={sourceClientFilter}
          onChange={(e) => { setSourceClientFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Source Clients</option>
          {uniqueSourceClients.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <select
          value={targetClientFilter}
          onChange={(e) => { setTargetClientFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Target Clients</option>
          {uniqueTargetClients.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <select
          value={sourceModalityFilter}
          onChange={(e) => { setSourceModalityFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Source Modalities</option>
          {uniqueSourceModalities.map((m) => (
            <option key={m} value={m}>{getModalityLabel(m)}</option>
          ))}
        </select>
        <select
          value={targetModalityFilter}
          onChange={(e) => { setTargetModalityFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Target Modalities</option>
          {uniqueTargetModalities.map((m) => (
            <option key={m} value={m}>{getModalityLabel(m)}</option>
          ))}
        </select>
        <div className="flex items-center gap-2 text-xs text-outline">
          <SlidersHorizontal size={14} />
          <span>{filtered.length} transfer{filtered.length !== 1 ? 's' : ''}</span>
        </div>
        {hasFilters && (
          <button
            onClick={() => { setSearch(''); setStatusFilter(''); setStrategyFilter(''); setSourceClientFilter(''); setTargetClientFilter(''); setSourceModalityFilter(''); setTargetModalityFilter(''); setPage(0); }}
            className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs text-outline hover:text-on-surface hover:bg-surface-container-high transition-colors cursor-pointer"
          >
            <X size={12} />
            Clear
          </button>
        )}
      </div>

      {/* Loading skeleton */}
      {isLoading && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)}
          </div>
          <Card className="overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="border-b border-outline-variant/30 text-left text-[10px] uppercase tracking-widest font-bold text-outline">
                  {['Transfer ID', 'Source', 'Target', 'Source Mod', 'Target Mod', 'Strategy', 'Similarity', 'Status'].map((h) => (
                    <th key={h} className="px-4 py-3">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)}
              </tbody>
            </table>
          </Card>
        </>
      )}

      {/* Error state */}
      {!isLoading && isError && (
        <ErrorState
          message={error instanceof Error ? error.message : 'An unexpected error occurred.'}
          onRetry={() => refetch()}
        />
      )}

      {/* Empty state */}
      {!isLoading && !isError && filtered.length === 0 && (
        <EmptyState onRefresh={handleRefresh} />
      )}

      {/* Data */}
      {!isLoading && !isError && filtered.length > 0 && (
        <>
          {/* Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                  <ArrowRightLeft size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Total Transfers</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (statistics?.total_transfers ?? allTransfers.length)}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-emerald-400">
                  <CheckCircle2 size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Successful</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (statistics?.successful_transfers ?? allTransfers.filter((t) => t.transfer_status === 'completed').length)}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-rose-400">
                  <XCircle size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Failed</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (statistics?.failed_transfers ?? allTransfers.filter((t) => t.transfer_status === 'failed').length)}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-amber-400">
                  <Activity size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Avg Similarity</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (
                      statistics
                        ? `${(statistics.average_similarity * 100).toFixed(1)}%`
                        : allTransfers.length > 0
                          ? `${(allTransfers.reduce((s, t) => s + t.similarity_score, 0) / allTransfers.length * 100).toFixed(1)}%`
                          : '—'
                    )}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                  <Target size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Avg Confidence</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (
                      statistics
                        ? `${(statistics.average_confidence * 100).toFixed(1)}%`
                        : allTransfers.length > 0
                          ? `${(allTransfers.reduce((s, t) => s + t.confidence, 0) / allTransfers.length * 100).toFixed(1)}%`
                          : '—'
                    )}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                  <BarChart3 size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Avg Loss</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (
                      statistics
                        ? statistics.average_transfer_loss.toFixed(4)
                        : allTransfers.length > 0
                          ? (allTransfers.reduce((s, t) => s + t.transfer_loss, 0) / allTransfers.length).toFixed(4)
                          : '—'
                    )}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-amber-400">
                  <Clock size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Avg Exec Time</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (
                      statistics
                        ? `${statistics.average_execution_time.toFixed(1)}s`
                        : allTransfers.length > 0
                          ? `${(allTransfers.reduce((s, t) => s + t.execution_time, 0) / allTransfers.length).toFixed(1)}s`
                          : '—'
                    )}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-emerald-400">
                  <GitBranch size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Comm Efficiency</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (
                      statistics
                        ? `${(statistics.communication_efficiency * 100).toFixed(1)}%`
                        : '—'
                    )}
                  </p>
                </div>
              </div>
            </Card>
          </div>

          {/* Data table */}
          <Card className="overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-outline-variant/30 text-left text-[10px] uppercase tracking-widest font-bold text-outline">
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('transfer_id')}>
                      Transfer ID <SortIcon field="transfer_id" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('source_client')}>
                      Source Client <SortIcon field="source_client" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('target_client')}>
                      Target Client <SortIcon field="target_client" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('source_modality')}>
                      Source Mod <SortIcon field="source_modality" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('target_modality')}>
                      Target Mod <SortIcon field="target_modality" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('transfer_strategy')}>
                      Strategy <SortIcon field="transfer_strategy" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('similarity_score')}>
                      Similarity <SortIcon field="similarity_score" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('confidence')}>
                      Confidence <SortIcon field="confidence" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('transfer_status')}>
                      Status <SortIcon field="transfer_status" />
                    </th>
                    <th className="px-4 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {paged.map((transfer) => (
                    <tr
                      key={transfer.transfer_id}
                      className="border-b border-outline-variant/20 hover:bg-surface-container-low transition-colors"
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-lg bg-surface-container-high flex items-center justify-center">
                            <ArrowRightLeft size={16} className="text-primary" />
                          </div>
                          <div>
                            <p className="text-sm font-semibold text-on-surface">{transfer.transfer_id}</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-on-surface-variant">{transfer.source_client}</td>
                      <td className="px-4 py-3 text-sm text-on-surface-variant">{transfer.target_client}</td>
                      <td className="px-4 py-3">
                        <StatusBadge status={transfer.source_modality} />
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={transfer.target_modality} />
                      </td>
                      <td className="px-4 py-3 text-sm text-on-surface-variant">{getTransferStrategyLabel(transfer.transfer_strategy)}</td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 max-w-[50px] h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary rounded-full transition-all"
                              style={{ width: `${transfer.similarity_score * 100}%` }}
                            />
                          </div>
                          <span className="text-sm font-mono font-semibold text-primary">{(transfer.similarity_score * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm font-mono text-on-surface">{(transfer.confidence * 100).toFixed(0)}%</td>
                      <td className="px-4 py-3">
                        <StatusBadge status={transfer.transfer_status} />
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => setSelectedTransfer(transfer)}
                          className="p-1.5 rounded-lg hover:bg-surface-container-high text-outline hover:text-on-surface transition-colors cursor-pointer"
                          title="View details"
                        >
                          <Eye size={16} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between px-4 py-3 border-t border-outline-variant/30">
              <div className="flex items-center gap-2 text-xs text-outline">
                <span>Rows per page:</span>
                <select
                  value={pageSize}
                  onChange={(e) => { setPageSize(Number(e.target.value)); setPage(0); }}
                  className="bg-transparent border border-outline-variant rounded px-2 py-1 text-on-surface text-xs focus:outline-none"
                >
                  {PAGE_SIZES.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2 text-xs text-outline">
                <span>
                  {safePage * pageSize + 1}&ndash;{Math.min((safePage + 1) * pageSize, filtered.length)} of {filtered.length}
                </span>
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={safePage === 0}
                  className="p-1 rounded hover:bg-surface-container-high disabled:opacity-30 transition-colors cursor-pointer"
                >
                  &#9664;
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={safePage >= totalPages - 1}
                  className="p-1 rounded hover:bg-surface-container-high disabled:opacity-30 transition-colors cursor-pointer"
                >
                  &#9654;
                </button>
              </div>
            </div>
          </Card>

          {/* Visualizations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                  <Network size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Cross-Modal Transfer Graph</h3>
                  <p className="text-xs text-outline">Transfer count by modality pair</p>
                </div>
              </div>
              <CrossModalTransferGraph transfers={allTransfers} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                  <Grid3X3 size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Similarity Heatmap</h3>
                  <p className="text-xs text-outline">Average similarity by modality pair</p>
                </div>
              </div>
              <SimilarityHeatmap transfers={allTransfers} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-amber-400">
                  <TrendingUp size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Transfer Timeline</h3>
                  <p className="text-xs text-outline">Average similarity per communication round</p>
                </div>
              </div>
              <TransferTimeline transfers={allTransfers} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                  <PieChartIcon size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Transfer Success Distribution</h3>
                  <p className="text-xs text-outline">Transfers by status</p>
                </div>
              </div>
              <TransferSuccessDistribution transfers={allTransfers} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                  <BarChart3 size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Transfer Loss Curve</h3>
                  <p className="text-xs text-outline">Transfer loss across transfers</p>
                </div>
              </div>
              <TransferLossCurve transfers={allTransfers} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                  <Layers size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Modality Interaction Matrix</h3>
                  <p className="text-xs text-outline">Average similarity by source-target modality</p>
                </div>
              </div>
              <ModalityInteractionMatrix transfers={allTransfers} />
            </Card>
          </div>
        </>
      )}

      {/* Detail panel */}
      {selectedTransfer && (
        <DetailPanel
          transfer={selectedTransfer}
          onClose={() => setSelectedTransfer(null)}
        />
      )}
    </div>
  );
};
