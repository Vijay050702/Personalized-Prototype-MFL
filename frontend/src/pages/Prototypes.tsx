import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Search, RefreshCw, SlidersHorizontal, X, Eye, Cpu,
  Layers, Activity, AlertTriangle, Database, Hash,
  BarChart3, TrendingUp, Grid3X3, PieChart as PieChartIcon,
} from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, LineChart, Line, PieChart,
  Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts';

import { fetchPrototypes } from '../api/prototypes';
import { Card } from '../components/ui/Card';
import { StatusBadge } from '../components/ui/StatusBadge';
import type { PrototypeResponse } from '../types';

const PAGE_SIZES = [5, 10, 20, 50];
const AUTO_REFRESH_MS = 5000;

function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}

const MODALITY_LABELS: Record<string, string> = {
  visual: 'Image',
  acoustic: 'Audio',
  linguistic: 'Text',
  multimodal: 'Multimodal',
};

const MODALITY_COLORS: Record<string, string> = {
  visual: 'var(--color-primary)',
  acoustic: 'var(--color-secondary)',
  linguistic: 'var(--color-tertiary, #a78bfa)',
  multimodal: 'var(--color-error, #f87171)',
};

const CHART_COLORS = [
  'var(--color-primary)',
  'var(--color-secondary)',
  '#a78bfa',
  '#f87171',
  '#34d399',
  '#fbbf24',
];

function formatTimestamp(iso: string): string {
  try {
    const date = new Date(iso);
    return date.toLocaleTimeString();
  } catch {
    return iso;
  }
}

function getModalityLabel(m: string): string {
  return MODALITY_LABELS[m] || m;
}

type SortField = 'id' | 'modality' | 'dimension' | 'cluster_id' | 'quality_score' | 'client_id' | 'created_round';

/* ====================================================================== */
/*  Skeleton                                                               */
/* ====================================================================== */

function SkeletonRow() {
  return (
    <tr className="animate-pulse border-b border-outline-variant/30">
      {Array.from({ length: 7 }).map((_, i) => (
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
        <Database size={32} className="text-outline" />
      </div>
      <h3 className="text-lg font-display font-bold text-on-surface">No prototypes found</h3>
      <p className="text-sm text-outline max-w-sm text-center">
        No prototypes match your search criteria. Try adjusting your filters.
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
      <h3 className="text-lg font-display font-bold text-on-surface">Failed to load prototypes</h3>
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
  prototype: p,
  onClose,
}: {
  prototype: PrototypeResponse;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/40 backdrop-blur-sm">
      <Card className="w-full max-w-lg h-full overflow-y-auto rounded-none rounded-l-2xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center">
              <Database size={20} className="text-primary" />
            </div>
            <div>
              <h2 className="text-lg font-display font-bold text-on-surface">Prototype Details</h2>
              <p className="text-[10px] font-mono text-outline uppercase">{p.id}</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-surface-container-high transition-colors cursor-pointer">
            <X size={20} className="text-outline" />
          </button>
        </div>

        <div className="space-y-4 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Metadata</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Modality</p>
              <p className="text-sm font-semibold text-on-surface">{getModalityLabel(p.modality)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Class ID</p>
              <p className="text-sm font-semibold text-on-surface">Class #{p.cluster_id}</p>
            </div>
          </div>
        </div>

        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Prototype Statistics</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Quality Score</p>
              <p className="text-lg font-display font-bold text-primary">{(p.quality_score * 100).toFixed(1)}%</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Embedding Dim</p>
              <p className="text-lg font-display font-bold text-on-surface">{p.dimension}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Support Samples</p>
              <p className="text-lg font-display font-bold text-on-surface">—</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Similarity Score</p>
              <p className="text-lg font-display font-bold text-on-surface">—</p>
            </div>
          </div>
        </div>

        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-bold text-on-surface">History</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Created Round</p>
              <p className="text-sm font-semibold text-on-surface">Round #{p.created_round}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Created Time</p>
              <p className="text-sm font-semibold text-on-surface">—</p>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-bold text-on-surface">Associations</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Client</p>
              <p className="text-sm font-semibold text-on-surface">{p.client_id}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Communication Round</p>
              <p className="text-sm font-semibold text-on-surface">Round #{p.created_round}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Dataset</p>
              <p className="text-sm font-semibold text-outline">—</p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

/* ====================================================================== */
/*  Similarity Matrix                                                      */
/* ====================================================================== */

function SimilarityMatrix({ prototypes }: { prototypes: PrototypeResponse[] }) {
  const matrix = useMemo(() => {
    const n = prototypes.length;
    const rows: { id: string; [key: string]: number | string }[] = [];
    for (let i = 0; i < n; i++) {
      const row: { id: string; [key: string]: number | string } = { id: prototypes[i].id.slice(-6) };
      for (let j = 0; j < n; j++) {
        const a = prototypes[i];
        const b = prototypes[j];
        let score = 0;
        if (a.cluster_id === b.cluster_id) score += 0.5;
        if (a.modality === b.modality) score += 0.3;
        score += 0.2 * (1 - Math.abs(a.quality_score - b.quality_score));
        row[prototypes[j].id.slice(-6)] = parseFloat(score.toFixed(3));
      }
      rows.push(row);
    }
    return rows;
  }, [prototypes]);

  const keys = prototypes.map((p) => p.id.slice(-6));

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr>
            <th className="p-1.5 text-outline text-[10px] font-bold uppercase" />
            {keys.map((k) => (
              <th key={k} className="p-1.5 text-outline text-[10px] font-mono font-bold text-center">{k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={row.id}>
              <td className="p-1.5 text-outline text-[10px] font-mono font-bold">{row.id}</td>
              {keys.map((k) => {
                const val = row[k] as number;
                const intensity = Math.round(val * 100);
                return (
                  <td
                    key={k}
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
/*  Visualization Sections                                                 */
/* ====================================================================== */

function ClassDistributionChart({ prototypes }: { prototypes: PrototypeResponse[] }) {
  const data = useMemo(() => {
    const counts: Record<number, number> = {};
    prototypes.forEach((p) => { counts[p.cluster_id] = (counts[p.cluster_id] || 0) + 1; });
    return Object.entries(counts)
      .map(([cls, count]) => ({ class: `Class ${cls}`, count }))
      .sort((a, b) => a.class.localeCompare(b.class));
  }, [prototypes]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No data available</p>;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="class" tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} allowDecimals={false} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
        />
        <Bar dataKey="count" fill="var(--color-primary)" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function ModalityDistributionChart({ prototypes }: { prototypes: PrototypeResponse[] }) {
  const data = useMemo(() => {
    const counts: Record<string, number> = {};
    prototypes.forEach((p) => {
      const label = getModalityLabel(p.modality);
      counts[label] = (counts[label] || 0) + 1;
    });
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [prototypes]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No data available</p>;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
          {data.map((_, i) => (
            <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}

function ConfidenceChart({ prototypes }: { prototypes: PrototypeResponse[] }) {
  const data = useMemo(() => {
    return [...prototypes]
      .sort((a, b) => a.created_round - b.created_round)
      .map((p) => ({
        round: `R${p.created_round}`,
        quality: parseFloat((p.quality_score * 100).toFixed(1)),
        id: p.id.slice(-6),
      }));
  }, [prototypes]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No data available</p>;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="round" tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis domain={[0, 100]} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${v}%`} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
          formatter={(value: number) => [`${value}%`, 'Quality Score']}
        />
        <Line type="monotone" dataKey="quality" stroke="var(--color-primary)" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function DimensionChart({ prototypes }: { prototypes: PrototypeResponse[] }) {
  const data = useMemo(() => {
    const counts: Record<number, number> = {};
    prototypes.forEach((p) => { counts[p.dimension] = (counts[p.dimension] || 0) + 1; });
    return Object.entries(counts)
      .map(([dim, count]) => ({ dimension: `${dim}d`, count }))
      .sort((a, b) => parseInt(a.dimension) - parseInt(b.dimension));
  }, [prototypes]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No data available</p>;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="dimension" tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} allowDecimals={false} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
        />
        <Bar dataKey="count" fill="var(--color-secondary)" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function TimelineChart({ prototypes }: { prototypes: PrototypeResponse[] }) {
  const data = useMemo(() => {
    const grouped: Record<number, { round: number; avgQuality: number; count: number }> = {};
    prototypes.forEach((p) => {
      if (!grouped[p.created_round]) {
        grouped[p.created_round] = { round: p.created_round, avgQuality: 0, count: 0 };
      }
      grouped[p.created_round].avgQuality += p.quality_score;
      grouped[p.created_round].count += 1;
    });
    return Object.values(grouped)
      .map((g) => ({
        round: `R${g.round}`,
        avgQuality: parseFloat(((g.avgQuality / g.count) * 100).toFixed(1)),
        count: g.count,
      }))
      .sort((a, b) => parseInt(a.round.slice(1)) - parseInt(b.round.slice(1)));
  }, [prototypes]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No data available</p>;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="round" tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis domain={[0, 100]} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${v}%`} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
          formatter={(value: number, name: string) => [
            name === 'avgQuality' ? `${value}%` : value,
            name === 'avgQuality' ? 'Avg Quality' : 'Prototypes',
          ]}
        />
        <Line type="monotone" dataKey="avgQuality" stroke="var(--color-primary)" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} name="avgQuality" />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ====================================================================== */
/*  Main Prototypes Page                                                   */
/* ====================================================================== */

export const Prototypes = () => {
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState('');
  const [clientFilter, setClientFilter] = useState('');
  const [modalityFilter, setModalityFilter] = useState('');
  const [sortField, setSortField] = useState<SortField>('id');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(10);
  const [selectedPrototype, setSelectedPrototype] = useState<PrototypeResponse | null>(null);
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
    queryKey: ['prototypes'],
    queryFn: fetchPrototypes,
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

  const allPrototypes: PrototypeResponse[] = listResponse?.data ?? [];

  const uniqueTypes = useMemo(() => {
    const set = new Set(allPrototypes.map((p) => p.modality));
    return Array.from(set).sort();
  }, [allPrototypes]);

  const uniqueClients = useMemo(() => {
    const set = new Set(allPrototypes.map((p) => p.client_id));
    return Array.from(set).sort();
  }, [allPrototypes]);

  const filtered = useMemo(() => {
    let list = allPrototypes;

    if (search) {
      const q = search.toLowerCase();
      list = list.filter(
        (p) =>
          p.id.toLowerCase().includes(q) ||
          p.client_id.toLowerCase().includes(q) ||
          p.modality.toLowerCase().includes(q) ||
          `class ${p.cluster_id}`.includes(q),
      );
    }

    if (typeFilter) {
      list = list.filter((p) => p.modality === typeFilter);
    }

    if (clientFilter) {
      list = list.filter((p) => p.client_id === clientFilter);
    }

    if (modalityFilter) {
      list = list.filter((p) => p.modality === modalityFilter);
    }

    list.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case 'id': cmp = a.id.localeCompare(b.id); break;
        case 'modality': cmp = a.modality.localeCompare(b.modality); break;
        case 'dimension': cmp = a.dimension - b.dimension; break;
        case 'cluster_id': cmp = a.cluster_id - b.cluster_id; break;
        case 'quality_score': cmp = a.quality_score - b.quality_score; break;
        case 'client_id': cmp = a.client_id.localeCompare(b.client_id); break;
        case 'created_round': cmp = a.created_round - b.created_round; break;
        default: cmp = 0;
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return list;
  }, [allPrototypes, search, typeFilter, clientFilter, modalityFilter, sortField, sortDir]);

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

  const hasFilters = search || typeFilter || clientFilter || modalityFilter;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">Prototype Repository</h1>
          <p className="text-sm text-outline">Browse and analyze learned prototypes across clients and modalities</p>
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
            placeholder="Search by ID, client, modality, class..."
            className="w-full pl-9 pr-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm placeholder:text-outline/50 focus:outline-none focus:border-primary transition-colors"
          />
        </div>
        <select
          value={typeFilter}
          onChange={(e) => { setTypeFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Types</option>
          {uniqueTypes.map((t) => (
            <option key={t} value={t}>{getModalityLabel(t)}</option>
          ))}
        </select>
        <select
          value={clientFilter}
          onChange={(e) => { setClientFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Clients</option>
          {uniqueClients.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <select
          value={modalityFilter}
          onChange={(e) => { setModalityFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Modalities</option>
          {uniqueTypes.map((t) => (
            <option key={t} value={t}>{getModalityLabel(t)}</option>
          ))}
        </select>
        <div className="flex items-center gap-2 text-xs text-outline">
          <SlidersHorizontal size={14} />
          <span>{filtered.length} prototype{filtered.length !== 1 ? 's' : ''}</span>
        </div>
        {hasFilters && (
          <button
            onClick={() => { setSearch(''); setTypeFilter(''); setClientFilter(''); setModalityFilter(''); setPage(0); }}
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
                  {['ID', 'Modality', 'Dimension', 'Class', 'Quality', 'Client', 'Round'].map((h) => (
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

      {/* Data table */}
      {!isLoading && !isError && filtered.length > 0 && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                  <Database size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Total Prototypes</p>
                  <p className="text-xl font-display font-bold text-on-surface">{allPrototypes.length}</p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                  <Layers size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Modalities</p>
                  <p className="text-xl font-display font-bold text-on-surface">{uniqueTypes.length}</p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-tertiary">
                  <Hash size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Classes</p>
                  <p className="text-xl font-display font-bold text-on-surface">{new Set(allPrototypes.map((p) => p.cluster_id)).size}</p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-amber-400">
                  <Activity size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Avg Quality</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {allPrototypes.length > 0
                      ? `${(allPrototypes.reduce((s, p) => s + p.quality_score, 0) / allPrototypes.length * 100).toFixed(1)}%`
                      : '—'}
                  </p>
                </div>
              </div>
            </Card>
          </div>

          <Card className="overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-outline-variant/30 text-left text-[10px] uppercase tracking-widest font-bold text-outline">
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('id')}>
                      Prototype ID <SortIcon field="id" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('modality')}>
                      Modality <SortIcon field="modality" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('dimension')}>
                      Dimension <SortIcon field="dimension" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('cluster_id')}>
                      Class <SortIcon field="cluster_id" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('quality_score')}>
                      Quality Score <SortIcon field="quality_score" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('client_id')}>
                      Client <SortIcon field="client_id" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('created_round')}>
                      Round <SortIcon field="created_round" />
                    </th>
                    <th className="px-4 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {paged.map((prototype) => (
                    <tr
                      key={prototype.id}
                      className="border-b border-outline-variant/20 hover:bg-surface-container-low transition-colors"
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-lg bg-surface-container-high flex items-center justify-center">
                            <Database size={16} className="text-primary" />
                          </div>
                          <div>
                            <p className="text-sm font-semibold text-on-surface">{prototype.id}</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={prototype.modality} />
                      </td>
                      <td className="px-4 py-3 text-sm font-mono text-on-surface">{prototype.dimension}</td>
                      <td className="px-4 py-3 text-sm font-mono text-on-surface">#{prototype.cluster_id}</td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 max-w-[60px] h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary rounded-full transition-all"
                              style={{ width: `${prototype.quality_score * 100}%` }}
                            />
                          </div>
                          <span className="text-sm font-mono font-semibold text-primary">{(prototype.quality_score * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-on-surface-variant">{prototype.client_id}</td>
                      <td className="px-4 py-3 text-sm font-mono text-on-surface">#{prototype.created_round}</td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => setSelectedPrototype(prototype)}
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
                  <Grid3X3 size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Similarity Matrix</h3>
                  <p className="text-xs text-outline">Pairwise prototype similarity (metadata-based)</p>
                </div>
              </div>
              <SimilarityMatrix prototypes={allPrototypes} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                  <BarChart3 size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Quality Score Distribution</h3>
                  <p className="text-xs text-outline">Prototype quality scores across rounds</p>
                </div>
              </div>
              <ConfidenceChart prototypes={allPrototypes} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-amber-400">
                  <TrendingUp size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Prototype Evolution Timeline</h3>
                  <p className="text-xs text-outline">Average quality score per communication round</p>
                </div>
              </div>
              <TimelineChart prototypes={allPrototypes} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                  <PieChartIcon size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Modality Distribution</h3>
                  <p className="text-xs text-outline">Prototypes grouped by modality</p>
                </div>
              </div>
              <ModalityDistributionChart prototypes={allPrototypes} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                  <Layers size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Class Distribution</h3>
                  <p className="text-xs text-outline">Prototypes grouped by class ID</p>
                </div>
              </div>
              <ClassDistributionChart prototypes={allPrototypes} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-amber-400">
                  <BarChart3 size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Embedding Dimension Statistics</h3>
                  <p className="text-xs text-outline">Prototype count by embedding dimension</p>
                </div>
              </div>
              <DimensionChart prototypes={allPrototypes} />
            </Card>
          </div>
        </>
      )}

      {/* Detail panel */}
      {selectedPrototype && (
        <DetailPanel
          prototype={selectedPrototype}
          onClose={() => setSelectedPrototype(null)}
        />
      )}
    </div>
  );
};
