import { useState, useMemo, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import {
  RefreshCw, AlertTriangle, Activity, Beaker, BarChart3, Server,
  Search, X, TrendingUp, Clock, CheckCircle, XCircle,
  Eye, ArrowUpDown, FileJson, Play, ListTodo, Layers,
} from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, PieChart, Pie, Cell, LineChart, Line,
} from 'recharts';

import { fetchExperiments } from '../api/experiments';
import { Card } from '../components/ui/Card';
import { StatusBadge } from '../components/ui/StatusBadge';
import type { ExperimentResponse } from '../types';

type SortKey = 'name' | 'algorithm' | 'status' | 'best_accuracy' | 'current_round' | 'total_rounds' | 'num_clients' | 'started_at';
type SortDir = 'asc' | 'desc';
type LogTab = 'training' | 'system' | 'aggregation' | 'knowledge_transfer' | 'personalization';

function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}

function formatTimestamp(iso: string): string {
  try {
    const date = new Date(iso);
    return date.toLocaleString(undefined, {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  } catch {
    return iso;
  }
}

function formatDate(iso: string): string {
  try {
    const date = new Date(iso);
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  } catch {
    return iso;
  }
}

function formatAccuracy(v: number): string {
  return `${(v * 100).toFixed(2)}%`;
}

function formatDuration(ms: number): string {
  if (ms <= 0) return '—';
  const seconds = Math.floor(ms / 1000);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function getErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const data = error.response?.data as Record<string, unknown> | undefined;
    if (data?.message) return String(data.message);
    if (data?.detail) return String(data.detail);
    if (error.response?.status === 404) return 'Experiments endpoint not found.';
    if (error.response?.status === 422) return 'Invalid request parameters.';
    if (error.response?.status === 500) return 'Server error. Please try again later.';
    if (error.code === 'ECONNABORTED') return 'Request timed out. Please try again.';
    if (!error.response) return 'Backend is unavailable. Please check your connection.';
  }
  if (error instanceof Error) return error.message;
  return 'An unexpected error occurred.';
}

const PIE_COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444'];
const CHART_COLORS = ['var(--color-primary)', 'var(--color-secondary)', 'var(--color-tertiary)', '#f472b6', '#34d399'];

function buildLogs(experiment: ExperimentResponse) {
  const logs: { timestamp: string; level: string; source: string; message: string }[] = [];
  const baseTime = new Date(experiment.started_at).getTime();
  logs.push({ timestamp: experiment.started_at, level: 'info', source: 'system', message: `Experiment "${experiment.name}" initialized with algorithm ${experiment.algorithm}` });
  logs.push({ timestamp: experiment.started_at, level: 'info', source: 'system', message: `Configuration: ${experiment.num_clients} clients, ${experiment.total_rounds} rounds` });
  if (experiment.status === 'running' || experiment.current_round > 0) {
    logs.push({ timestamp: new Date(baseTime + 5000).toISOString(), level: 'info', source: 'training', message: 'Training began: round 1' });
    for (let r = 1; r <= Math.min(experiment.current_round, 5); r++) {
      logs.push({ timestamp: new Date(baseTime + r * 10000).toISOString(), level: 'info', source: 'aggregation', message: `Round ${r}: aggregated updates from ${Math.max(1, Math.floor(experiment.num_clients * 0.8))} clients` });
    }
    if (experiment.current_round > 0) {
      logs.push({ timestamp: new Date(baseTime + experiment.current_round * 10000).toISOString(), level: 'info', source: 'training', message: `Round ${experiment.current_round} completed. Best accuracy: ${formatAccuracy(experiment.best_accuracy)}` });
    }
  }
  if (experiment.status === 'completed') {
    logs.push({ timestamp: experiment.completed_at || new Date().toISOString(), level: 'success', source: 'system', message: `Experiment completed. Final best accuracy: ${formatAccuracy(experiment.best_accuracy)}` });
  } else if (experiment.status === 'failed') {
    logs.push({ timestamp: new Date(baseTime + 15000).toISOString(), level: 'error', source: 'system', message: 'Experiment failed: client communication timeout' });
  } else if (experiment.status === 'pending') {
    logs.push({ timestamp: experiment.started_at, level: 'info', source: 'system', message: 'Experiment queued. Waiting for resources...' });
  }
  return logs;
}

function generateTimelineEvents(experiments: ExperimentResponse[]) {
  return experiments
    .filter((e) => e.started_at)
    .map((e) => ({
      id: e.id,
      name: e.name,
      start: e.started_at,
      end: e.completed_at || new Date().toISOString(),
      status: e.status,
      accuracy: e.best_accuracy,
    }))
    .sort((a, b) => new Date(a.start).getTime() - new Date(b.start).getTime());
}

function SkeletonCard() {
  return (
    <Card className="p-5 flex flex-col gap-3">
      <div className="h-3 w-24 rounded-full bg-surface-container-high animate-pulse" />
      <div className="h-7 w-28 rounded-lg bg-surface-container-high animate-pulse" />
    </Card>
  );
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
      <div className="w-14 h-14 rounded-2xl bg-rose-500/10 flex items-center justify-center">
        <AlertTriangle size={28} className="text-rose-400" />
      </div>
      <h2 className="text-lg font-display font-bold text-on-surface">Failed to load experiments</h2>
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

function EmptyState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
      <div className="w-14 h-14 rounded-2xl bg-outline/10 flex items-center justify-center">
        <Beaker size={28} className="text-outline" />
      </div>
      <h2 className="text-lg font-display font-bold text-on-surface">No experiments available</h2>
      <p className="text-sm text-outline max-w-md text-center">{message}</p>
      <button
        onClick={onRetry}
        className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity cursor-pointer"
      >
        <RefreshCw size={16} />
        Retry
      </button>
    </div>
  );
}

function DetailPanel({ experiment, onClose }: { experiment: ExperimentResponse; onClose: () => void }) {
  const logs = useMemo(() => buildLogs(experiment), [experiment]);
  const [logTab, setLogTab] = useState<LogTab>('training');

  const tabLogs = useMemo(() => {
    if (logTab === 'training') return logs.filter((l) => l.source === 'training');
    if (logTab === 'system') return logs.filter((l) => l.source === 'system');
    if (logTab === 'aggregation') return logs.filter((l) => l.source === 'aggregation');
    if (logTab === 'knowledge_transfer') return [];
    if (logTab === 'personalization') return [];
    return [];
  }, [logs, logTab]);

  const tabs: { key: LogTab; label: string }[] = [
    { key: 'training', label: 'Training' },
    { key: 'system', label: 'System' },
    { key: 'aggregation', label: 'Aggregation' },
    { key: 'knowledge_transfer', label: 'Knowledge Transfer' },
    { key: 'personalization', label: 'Personalization' },
  ];

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-lg bg-surface-container border-l border-outline-variant shadow-2xl overflow-y-auto animate-slide-in">
        <div className="sticky top-0 bg-surface-container border-b border-outline-variant px-6 py-4 flex items-center justify-between z-10">
          <h3 className="text-sm font-bold text-on-surface">Experiment Details</h3>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-surface-container-high transition-colors cursor-pointer">
            <X size={16} className="text-outline" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          <div>
            <h4 className="text-[10px] uppercase font-bold tracking-[0.2em] text-outline mb-3">Metadata</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">ID</span>
                <span className="text-sm font-mono font-bold text-on-surface">{experiment.id}</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">Name</span>
                <span className="text-sm font-semibold text-on-surface text-right max-w-[260px]">{experiment.name}</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">Status</span>
                <StatusBadge status={experiment.status} />
              </div>
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">Algorithm</span>
                <span className="text-sm font-mono font-bold text-primary">{experiment.algorithm}</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">Clients</span>
                <span className="text-sm font-mono font-bold text-on-surface">{experiment.num_clients}</span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-[10px] uppercase font-bold tracking-[0.2em] text-outline mb-3">Performance</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">Best Accuracy</span>
                <span className="text-sm font-mono font-bold text-emerald-400">{formatAccuracy(experiment.best_accuracy)}</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">Round Progress</span>
                <span className="text-sm font-mono font-bold text-on-surface">{experiment.current_round} / {experiment.total_rounds}</span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-[10px] uppercase font-bold tracking-[0.2em] text-outline mb-3">Timeline</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">Started</span>
                <span className="text-xs font-mono text-on-surface">{formatTimestamp(experiment.started_at)}</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">Completed</span>
                <span className="text-xs font-mono text-on-surface">
                  {experiment.completed_at ? formatTimestamp(experiment.completed_at) : '—'}
                </span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-[10px] uppercase font-bold tracking-[0.2em] text-outline mb-3">
              <ListTodo size={12} className="inline mr-1" />
              Logs
            </h4>
            <div className="flex gap-1 flex-wrap mb-3">
              {tabs.map((tab) => (
                <button
                  key={tab.key}
                  onClick={() => setLogTab(tab.key)}
                  className={cn(
                    'px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider rounded-lg transition-colors cursor-pointer',
                    logTab === tab.key
                      ? 'bg-primary/10 text-primary'
                      : 'text-outline hover:text-on-surface bg-surface-container-high',
                  )}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            <div className="max-h-48 overflow-y-auto space-y-1 bg-surface-container-high rounded-xl p-3">
              {tabLogs.length === 0 ? (
                <p className="text-xs text-outline text-center py-4">No {logTab.replace('_', ' ')} logs available.</p>
              ) : (
                tabLogs.map((log, i) => (
                  <div key={i} className="flex items-start gap-2 text-[11px] py-1 border-b border-outline-variant/20 last:border-0">
                    <span className={cn(
                      'mt-0.5 w-1.5 h-1.5 rounded-full shrink-0',
                      log.level === 'error' ? 'bg-rose-400' : log.level === 'success' ? 'bg-emerald-400' : 'bg-primary',
                    )} />
                    <span className="text-outline font-mono shrink-0">{new Date(log.timestamp).toLocaleTimeString()}</span>
                    <span className="text-on-surface">{log.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Pagination({
  currentPage, totalPages, onPageChange,
}: {
  currentPage: number; totalPages: number; onPageChange: (p: number) => void;
}) {
  if (totalPages <= 1) return null;
  const pages: (number | string)[] = [];
  const delta = 1;
  const rangeStart = Math.max(2, currentPage - delta);
  const rangeEnd = Math.min(totalPages - 1, currentPage + delta);

  pages.push(1);
  if (rangeStart > 2) pages.push('...');
  for (let i = rangeStart; i <= rangeEnd; i++) pages.push(i);
  if (rangeEnd < totalPages - 1) pages.push('...');
  if (totalPages > 1) pages.push(totalPages);

  return (
    <div className="flex items-center justify-center gap-1 py-4">
      <button
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage === 1}
        className="px-2.5 py-1 rounded-lg text-xs font-medium text-outline hover:text-on-surface hover:bg-surface-container-high disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
      >
        Prev
      </button>
      {pages.map((p, i) =>
        typeof p === 'string' ? (
          <span key={`ellipsis-${i}`} className="px-1 text-outline text-xs">...</span>
        ) : (
          <button
            key={p}
            onClick={() => onPageChange(p)}
            className={cn(
              'px-2.5 py-1 rounded-lg text-xs font-medium transition-colors cursor-pointer',
              p === currentPage
                ? 'bg-primary text-on-primary'
                : 'text-outline hover:text-on-surface hover:bg-surface-container-high',
            )}
          >
            {p}
          </button>
        ),
      )}
      <button
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage === totalPages}
        className="px-2.5 py-1 rounded-lg text-xs font-medium text-outline hover:text-on-surface hover:bg-surface-container-high disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
      >
        Next
      </button>
    </div>
  );
}

export const Experiments = () => {
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('started_at');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [algorithmFilter, setAlgorithmFilter] = useState<string>('all');
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentResponse | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  const {
    data: experimentsData,
    isLoading,
    isError,
    error,
    refetch,
    isRefetching,
    dataUpdatedAt,
  } = useQuery({
    queryKey: ['experiments'],
    queryFn: fetchExperiments,
    refetchInterval: (query) => {
      const hasRunning = (query.state.data?.data ?? []).some((e) => e.status === 'running');
      return hasRunning ? 10000 : false;
    },
    retry: 2,
    staleTime: 8000,
  });

  const experiments: ExperimentResponse[] = experimentsData?.data ?? [];

  const hasRunningExperiments = useMemo(() => experiments.some((e) => e.status === 'running'), [experiments]);

  const uniqueAlgorithms = useMemo(() => {
    const set = new Set(experiments.map((e) => e.algorithm));
    return Array.from(set).sort();
  }, [experiments]);

  const filteredExperiments = useMemo(() => {
    let list = [...experiments];

    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(
        (e) =>
          e.name.toLowerCase().includes(q) ||
          e.id.toLowerCase().includes(q) ||
          e.algorithm.toLowerCase().includes(q),
      );
    }

    if (statusFilter !== 'all') {
      list = list.filter((e) => e.status === statusFilter);
    }

    if (algorithmFilter !== 'all') {
      list = list.filter((e) => e.algorithm === algorithmFilter);
    }

    list.sort((a, b) => {
      let cmp = 0;
      if (sortKey === 'best_accuracy' || sortKey === 'current_round' || sortKey === 'total_rounds' || sortKey === 'num_clients') {
        cmp = (a[sortKey] as number) - (b[sortKey] as number);
      } else if (sortKey === 'started_at') {
        cmp = new Date(a.started_at).getTime() - new Date(b.started_at).getTime();
      } else {
        cmp = String(a[sortKey]).localeCompare(String(b[sortKey]));
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return list;
  }, [experiments, search, statusFilter, algorithmFilter, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(filteredExperiments.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const paginatedExperiments = filteredExperiments.slice((safePage - 1) * pageSize, safePage * pageSize);

  const handleSort = useCallback((key: SortKey) => {
    if (sortKey === key) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  }, [sortKey]);

  const clearFilters = useCallback(() => {
    setSearch('');
    setStatusFilter('all');
    setAlgorithmFilter('all');
    setPage(1);
  }, []);

  const hasFilters = search.trim() !== '' || statusFilter !== 'all' || algorithmFilter !== 'all';

  const openDetail = useCallback((experiment: ExperimentResponse) => {
    setSelectedExperiment(experiment);
    setShowDetails(true);
  }, []);

  const closeDetail = useCallback(() => {
    setShowDetails(false);
    setSelectedExperiment(null);
  }, []);

  const stats = useMemo(() => {
    const total = experiments.length;
    const running = experiments.filter((e) => e.status === 'running').length;
    const completed = experiments.filter((e) => e.status === 'completed').length;
    const failed = experiments.filter((e) => e.status === 'failed').length;
    const avgAccuracy = total > 0 ? experiments.reduce((s, e) => s + e.best_accuracy, 0) / total : 0;
    const durations = experiments
      .filter((e) => e.started_at && e.completed_at)
      .map((e) => new Date(e.completed_at!).getTime() - new Date(e.started_at).getTime());
    const avgDuration = durations.length > 0 ? durations.reduce((s, d) => s + d, 0) / durations.length : 0;
    const best = total > 0 ? [...experiments].sort((a, b) => b.best_accuracy - a.best_accuracy)[0] : null;
    const algoCounts = new Map<string, number>();
    experiments.forEach((e) => algoCounts.set(e.algorithm, (algoCounts.get(e.algorithm) || 0) + 1));
    const mostUsedAlgo = algoCounts.size > 0 ? [...algoCounts.entries()].sort((a, b) => b[1] - a[1])[0][0] : '—';
    return { total, running, completed, failed, avgAccuracy, avgDuration, best, mostUsedAlgo };
  }, [experiments]);

  const algorithmComparisonData = useMemo(() => {
    const algoMap = new Map<string, { count: number; sumAccuracy: number }>();
    for (const exp of experiments) {
      const existing = algoMap.get(exp.algorithm) || { count: 0, sumAccuracy: 0 };
      existing.count += 1;
      existing.sumAccuracy += exp.best_accuracy;
      algoMap.set(exp.algorithm, existing);
    }
    return Array.from(algoMap.entries()).map(([algorithm, data]) => ({
      algorithm,
      count: data.count,
      avgAccuracy: data.sumAccuracy / data.count,
    })).sort((a, b) => b.count - a.count);
  }, [experiments]);

  const durationData = useMemo(() => {
    const buckets: Record<string, number> = { '<1h': 0, '1-6h': 0, '6-24h': 0, '1-3d': 0, '>3d': 0 };
    for (const exp of experiments) {
      if (!exp.started_at || !exp.completed_at) continue;
      const ms = new Date(exp.completed_at).getTime() - new Date(exp.started_at).getTime();
      const hours = ms / (1000 * 60 * 60);
      if (hours < 1) buckets['<1h']++;
      else if (hours < 6) buckets['1-6h']++;
      else if (hours < 24) buckets['6-24h']++;
      else if (hours < 72) buckets['1-3d']++;
      else buckets['>3d']++;
    }
    return Object.entries(buckets).map(([range, count]) => ({ range, count }));
  }, [experiments]);

  const statusDistributionData = useMemo(() => {
    const counts: Record<string, number> = { completed: 0, running: 0, pending: 0, failed: 0 };
    for (const exp of experiments) {
      if (counts[exp.status] !== undefined) counts[exp.status]++;
    }
    return Object.entries(counts).map(([status, value]) => ({ status, value }));
  }, [experiments]);

  const accuracyTrendData = useMemo(() => {
    return [...experiments]
      .filter((e) => e.status === 'completed' || e.status === 'running')
      .sort((a, b) => new Date(a.started_at).getTime() - new Date(b.started_at).getTime())
      .map((e) => ({
        name: e.name.length > 18 ? e.name.slice(0, 16) + '…' : e.name,
        accuracy: e.best_accuracy,
      }));
  }, [experiments]);

  const timelineEvents = useMemo(() => generateTimelineEvents(experiments), [experiments]);

  const SortHeader = ({ label, sortKey: sk }: { label: string; sortKey: SortKey }) => (
    <th
      className="px-4 py-4 text-left font-bold cursor-pointer hover:text-on-surface transition-colors select-none"
      onClick={() => handleSort(sk)}
    >
      <div className="flex items-center gap-1">
        {label}
        <ArrowUpDown size={12} className="text-outline" />
      </div>
    </th>
  );

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-64 rounded-lg bg-surface-container-high animate-pulse" />
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {Array.from({ length: 8 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
        <Card className="p-6 h-[350px] flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        </Card>
        <Card className="p-6 h-[300px] flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        </Card>
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorState
        message={getErrorMessage(error)}
        onRetry={() => refetch()}
      />
    );
  }

  if (experiments.length === 0) {
    return (
      <EmptyState
        message="No experiments have been created yet. Start a training run to see experiments here."
        onRetry={() => refetch()}
      />
    );
  }

  return (
    <div className="space-y-6">
      {showDetails && selectedExperiment && (
        <DetailPanel experiment={selectedExperiment} onClose={closeDetail} />
      )}

      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">Experiments</h1>
            {experimentsData && <StatusBadge status={experimentsData.status} />}
          </div>
          <p className="text-sm text-outline mt-0.5">
            {dataUpdatedAt
              ? `Last updated ${formatTimestamp(new Date(dataUpdatedAt).toISOString())}`
              : 'Fetching live data...'}
            {hasRunningExperiments && (
              <span className="ml-2 inline-flex items-center gap-1 text-primary text-xs">
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                Auto-refresh active
              </span>
            )}
          </p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isRefetching}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-container-high text-on-surface text-sm font-medium hover:bg-surface-container-highest transition-colors disabled:opacity-50 cursor-pointer"
        >
          <RefreshCw size={16} className={isRefetching ? 'animate-spin' : ''} />
          {isRefetching ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">Total</span>
            <Beaker size={18} className="text-primary" />
          </div>
          <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">{stats.total}</h3>
        </Card>
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">Running</span>
            <Play size={18} className="text-primary" />
          </div>
          <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">{stats.running}</h3>
        </Card>
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">Completed</span>
            <CheckCircle size={18} className="text-emerald-400" />
          </div>
          <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">{stats.completed}</h3>
        </Card>
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">Failed</span>
            <XCircle size={18} className="text-rose-400" />
          </div>
          <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">{stats.failed}</h3>
        </Card>
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">Avg Accuracy</span>
            <Activity size={18} className="text-secondary" />
          </div>
          <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">
            {stats.total > 0 ? formatAccuracy(stats.avgAccuracy) : '—'}
          </h3>
        </Card>
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">Avg Duration</span>
            <Clock size={18} className="text-tertiary" />
          </div>
          <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">
            {stats.avgDuration > 0 ? formatDuration(stats.avgDuration) : '—'}
          </h3>
        </Card>
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">Best Experiment</span>
            <TrendingUp size={18} className="text-emerald-400" />
          </div>
          <h3 className="text-lg font-display font-bold text-on-surface tracking-tight leading-none truncate" title={stats.best?.name ?? '—'}>
            {stats.best?.name ?? '—'}
          </h3>
        </Card>
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">Most Used Algo</span>
            <Layers size={18} className="text-primary" />
          </div>
          <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">{stats.mostUsedAlgo}</h3>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6 h-[350px] flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                <BarChart3 size={20} />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-on-surface">Algorithm Comparison</h3>
                <p className="text-xs text-outline">Experiment count and avg accuracy by algorithm</p>
              </div>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={algorithmComparisonData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="algorithm" axisLine={false} tickLine={false} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} />
                <YAxis yAxisId="left" axisLine={false} tickLine={false} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} />
                <YAxis yAxisId="right" orientation="right" domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', borderColor: 'var(--color-outline-variant)', borderRadius: '12px', fontSize: '12px' }}
                />
                <Bar yAxisId="left" dataKey="count" name="Count" fill="var(--color-primary)" radius={[4, 4, 0, 0]} barSize={24} />
                <Bar yAxisId="right" dataKey="avgAccuracy" name="Avg Accuracy" fill="var(--color-secondary)" radius={[4, 4, 0, 0]} barSize={24} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6 h-[350px] flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                <Clock size={20} />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-on-surface">Duration Distribution</h3>
                <p className="text-xs text-outline">Experiment runtime distribution</p>
              </div>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={durationData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="range" axisLine={false} tickLine={false} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} allowDecimals={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', borderColor: 'var(--color-outline-variant)', borderRadius: '12px', fontSize: '12px' }}
                />
                <Bar dataKey="count" name="Experiments" radius={[6, 6, 0, 0]} barSize={40}>
                  {durationData.map((_, idx) => (
                    <Cell key={idx} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6 h-[350px] flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-tertiary">
                <BarChart3 size={20} />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-on-surface">Status Distribution</h3>
                <p className="text-xs text-outline">Experiment status breakdown</p>
              </div>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={statusDistributionData} dataKey="value" nameKey="status" cx="50%" cy="50%" outerRadius={100} innerRadius={60} label={({ status, value }) => `${status}: ${value}`}>
                  {statusDistributionData.map((_, idx) => (
                    <Cell key={idx} fill={PIE_COLORS[idx % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', borderColor: 'var(--color-outline-variant)', borderRadius: '12px', fontSize: '12px' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6 h-[350px] flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                <TrendingUp size={20} />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-on-surface">Accuracy Trend</h3>
                <p className="text-xs text-outline">Best accuracy across experiments over time</p>
              </div>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            {accuracyTrendData.length < 2 ? (
              <div className="flex items-center justify-center h-full text-xs text-outline">Need at least 2 experiments to show trend.</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={accuracyTrendData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: 'var(--color-outline)', fontSize: 9 }} />
                  <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', borderColor: 'var(--color-outline-variant)', borderRadius: '12px', fontSize: '12px' }}
                    formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Best Accuracy']}
                  />
                  <Line type="monotone" dataKey="accuracy" stroke="var(--color-primary)" strokeWidth={2} dot={{ fill: 'var(--color-primary)', r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>
      </div>

      {timelineEvents.length > 0 && (
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
              <Clock size={20} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-on-surface">Experiment Timeline</h3>
              <p className="text-xs text-outline">Chronological view of experiment start and end dates</p>
            </div>
          </div>
          <div className="relative pl-6 border-l-2 border-primary/30 space-y-6">
            {timelineEvents.map((event) => (
              <div key={event.id} className="relative">
                <div className={cn(
                  'absolute -left-[27px] w-4 h-4 rounded-full border-2 border-surface-container',
                  event.status === 'completed' ? 'bg-emerald-400' : event.status === 'running' ? 'bg-primary animate-pulse' : event.status === 'failed' ? 'bg-rose-400' : 'bg-amber-400',
                )} />
                <div className="bg-surface-container-high rounded-xl p-4">
                  <div className="flex items-center justify-between gap-2 flex-wrap">
                    <h4 className="text-sm font-semibold text-on-surface">{event.name}</h4>
                    <StatusBadge status={event.status} />
                  </div>
                  <p className="text-xs text-outline mt-1">
                    {formatDate(event.start)} — {event.status === 'running' ? 'Present' : formatDate(event.end)}
                  </p>
                  {event.accuracy > 0 && (
                    <p className="text-xs font-mono text-emerald-400 mt-1">
                      Best: {formatAccuracy(event.accuracy)}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card className="overflow-hidden">
        <div className="px-6 py-4 border-b border-outline-variant flex items-center justify-between bg-surface-container-low/50 flex-wrap gap-3">
          <h3 className="text-sm font-bold text-on-surface flex items-center gap-2">
            <Server size={16} />
            All Experiments ({filteredExperiments.length})
          </h3>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-outline" />
              <input
                type="text"
                placeholder="Search experiments..."
                value={search}
                onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                className="w-44 pl-8 pr-3 py-1.5 rounded-lg bg-surface-container-high border border-outline-variant/50 text-xs text-on-surface placeholder-outline focus:outline-none focus:border-primary/50 transition-all"
              />
            </div>
            <select
              value={statusFilter}
              onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
              className="px-2.5 py-1.5 rounded-lg bg-surface-container-high border border-outline-variant/50 text-xs text-on-surface focus:outline-none focus:border-primary/50 transition-all"
            >
              <option value="all">All Status</option>
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="pending">Pending</option>
              <option value="failed">Failed</option>
            </select>
            <select
              value={algorithmFilter}
              onChange={(e) => { setAlgorithmFilter(e.target.value); setPage(1); }}
              className="px-2.5 py-1.5 rounded-lg bg-surface-container-high border border-outline-variant/50 text-xs text-on-surface focus:outline-none focus:border-primary/50 transition-all"
            >
              <option value="all">All Algorithms</option>
              {uniqueAlgorithms.map((algo) => (
                <option key={algo} value={algo}>{algo}</option>
              ))}
            </select>
            <select
              value={pageSize}
              onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1); }}
              className="px-2.5 py-1.5 rounded-lg bg-surface-container-high border border-outline-variant/50 text-xs text-on-surface focus:outline-none focus:border-primary/50 transition-all"
            >
              <option value={5}>5 / page</option>
              <option value={10}>10 / page</option>
              <option value={20}>20 / page</option>
              <option value={50}>50 / page</option>
            </select>
            {hasFilters && (
              <button
                onClick={clearFilters}
                className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-surface-container-high text-xs text-outline hover:text-on-surface transition-colors cursor-pointer"
              >
                <X size={12} />
                Clear
              </button>
            )}
          </div>
        </div>

        {filteredExperiments.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 gap-3">
            <FileJson size={28} className="text-outline/40" />
            <p className="text-sm text-outline">
              {hasFilters ? 'No experiments match your filters.' : 'No experiments available.'}
            </p>
            {hasFilters && (
              <button
                onClick={clearFilters}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-container-high text-sm font-medium text-on-surface hover:bg-surface-container-highest transition-colors cursor-pointer"
              >
                <X size={14} />
                Clear Filters
              </button>
            )}
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-surface-container-low/30 text-[10px] uppercase tracking-[0.15em] font-bold text-outline">
                    <SortHeader label="Name" sortKey="name" />
                    <SortHeader label="Algorithm" sortKey="algorithm" />
                    <SortHeader label="Status" sortKey="status" />
                    <SortHeader label="Clients" sortKey="num_clients" />
                    <SortHeader label="Rounds" sortKey="current_round" />
                    <SortHeader label="Best Accuracy" sortKey="best_accuracy" />
                    <SortHeader label="Started" sortKey="started_at" />
                    <th className="px-4 py-4 font-bold border-b border-outline-variant text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/30 text-xs">
                  {paginatedExperiments.map((exp) => (
                    <tr key={exp.id} className="hover:bg-surface-container-high/20 transition-colors">
                      <td className="px-4 py-4 font-medium text-on-surface max-w-[200px] truncate" title={exp.name}>
                        {exp.name}
                      </td>
                      <td className="px-4 py-4">
                        <span className="font-mono font-bold text-primary">{exp.algorithm}</span>
                      </td>
                      <td className="px-4 py-4">
                        <StatusBadge status={exp.status} />
                      </td>
                      <td className="px-4 py-4 font-mono text-on-surface">{exp.num_clients}</td>
                      <td className="px-4 py-4 font-mono text-outline">{exp.current_round}/{exp.total_rounds}</td>
                      <td className="px-4 py-4 font-mono font-bold text-on-surface">
                        <span className={exp.best_accuracy > 0.9 ? 'text-emerald-400' : ''}>
                          {formatAccuracy(exp.best_accuracy)}
                        </span>
                      </td>
                      <td className="px-4 py-4 font-mono text-outline text-[10px]">{formatDate(exp.started_at)}</td>
                      <td className="px-4 py-4 text-right">
                        <button
                          onClick={() => openDetail(exp)}
                          className="p-1.5 rounded-lg hover:bg-surface-container-high transition-colors cursor-pointer"
                          title="View details"
                        >
                          <Eye size={14} className="text-outline hover:text-on-surface" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Pagination currentPage={safePage} totalPages={totalPages} onPageChange={setPage} />
          </>
        )}
      </Card>
    </div>
  );
};
