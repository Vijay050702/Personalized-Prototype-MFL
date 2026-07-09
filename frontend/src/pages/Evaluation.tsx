import { useState, useMemo, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import {
  RefreshCw, AlertTriangle, Activity, Target, BarChart3, Server,
  Search, X, TrendingUp,
  Layers, Eye, ArrowUpDown, FileJson,
} from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Radar, Cell,
} from 'recharts';

import { fetchEvaluation } from '../api/evaluation';
import { fetchExperiments } from '../api/experiments';
import { Card } from '../components/ui/Card';
import { StatusBadge } from '../components/ui/StatusBadge';
import type {
  EvaluationSummary, EvaluationResponse,
  ExperimentListResponse, ExperimentResponse,
} from '../types';

type SortKey = 'name' | 'algorithm' | 'status' | 'best_accuracy' | 'current_round' | 'total_rounds' | 'num_clients';
type SortDir = 'asc' | 'desc';

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

function getErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const data = error.response?.data as Record<string, unknown> | undefined;
    if (data?.message) return String(data.message);
    if (data?.detail) return String(data.detail);
    if (error.response?.status === 404) return 'Evaluation endpoint not found.';
    if (error.response?.status === 422) return 'Invalid request parameters.';
    if (error.response?.status === 500) return 'Server error. Please try again later.';
    if (error.code === 'ECONNABORTED') return 'Request timed out. Please try again.';
    if (!error.response) return 'Backend is unavailable. Please check your connection.';
  }
  if (error instanceof Error) return error.message;
  return 'An unexpected error occurred.';
}

function getChartLabel(key: string): string {
  const map: Record<string, string> = {
    precision: 'Precision',
    recall: 'Recall',
    f1_score: 'F1 Score',
    accuracy: 'Accuracy',
    auc_roc: 'ROC-AUC',
  };
  return map[key] ?? key;
}

function formatPct(v: number): string {
  return `${(v * 100).toFixed(2)}%`;
}

function formatAccuracy(v: number): string {
  return `${(v * 100).toFixed(2)}%`;
}

function SkeletonCard() {
  return (
    <Card className="p-5 flex flex-col gap-3">
      <div className="h-3 w-24 rounded-full bg-surface-container-high animate-pulse" />
      <div className="h-7 w-28 rounded-lg bg-surface-container-high animate-pulse" />
    </Card>
  );
}

function SkeletonLine() {
  return <div className="h-4 w-full rounded-full bg-surface-container-high animate-pulse" />;
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
      <div className="w-14 h-14 rounded-2xl bg-rose-500/10 flex items-center justify-center">
        <AlertTriangle size={28} className="text-rose-400" />
      </div>
      <h2 className="text-lg font-display font-bold text-on-surface">Failed to load evaluation data</h2>
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
        <BarChart3 size={28} className="text-outline" />
      </div>
      <h2 className="text-lg font-display font-bold text-on-surface">No evaluation data available</h2>
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

function DetailPanel({
  experiment,
  onClose,
}: {
  experiment: ExperimentResponse;
  onClose: () => void;
}) {
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
        </div>
      </div>
    </div>
  );
}

const ALGORITHM_ORDER = ['FedAvg', 'FedProx', 'SCAFFOLD', 'pFedProto'];

export const Evaluation = () => {
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('best_accuracy');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [algorithmFilter, setAlgorithmFilter] = useState<string>('all');
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentResponse | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  const {
    data: evalSummary,
    isLoading: evalLoading,
    isError: evalError,
    error: evalErr,
    refetch: refetchEval,
    isRefetching: evalRefetching,
    dataUpdatedAt: evalUpdatedAt,
  } = useQuery({
    queryKey: ['evaluation'],
    queryFn: fetchEvaluation,
    refetchInterval: 10000,
    retry: 2,
    staleTime: 8000,
  });

  const {
    data: experimentsData,
    isLoading: expLoading,
    isError: expError,
    error: expErr,
    refetch: refetchExp,
    isRefetching: expRefetching,
  } = useQuery({
    queryKey: ['experiments'],
    queryFn: fetchExperiments,
    refetchInterval: 30000,
    retry: 2,
    staleTime: 25000,
  });

  const evalData: EvaluationResponse | null = evalSummary?.data ?? null;
  const experiments: ExperimentResponse[] = experimentsData?.data ?? [];

  const isLoading = evalLoading || expLoading;
  const isError = evalError || expError;
  const error = evalErr || expErr;
  const isRefetching = evalRefetching || expRefetching;
  const refetchAll = useCallback(() => {
    refetchEval();
    refetchExp();
  }, [refetchEval, refetchExp]);

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
      if (sortKey === 'best_accuracy') {
        cmp = a.best_accuracy - b.best_accuracy;
      } else if (sortKey === 'current_round') {
        cmp = a.current_round - b.current_round;
      } else if (sortKey === 'total_rounds') {
        cmp = a.total_rounds - b.total_rounds;
      } else if (sortKey === 'num_clients') {
        cmp = a.num_clients - b.num_clients;
      } else if (sortKey === 'name') {
        cmp = a.name.localeCompare(b.name);
      } else if (sortKey === 'algorithm') {
        cmp = a.algorithm.localeCompare(b.algorithm);
      } else if (sortKey === 'status') {
        cmp = a.status.localeCompare(b.status);
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return list;
  }, [experiments, search, statusFilter, algorithmFilter, sortKey, sortDir]);

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

  const barData = evalData
    ? [
        { key: 'precision', value: evalData.precision, fill: 'var(--color-primary)' },
        { key: 'recall', value: evalData.recall, fill: 'var(--color-secondary)' },
        { key: 'f1_score', value: evalData.f1_score, fill: 'var(--color-tertiary)' },
      ]
    : [];

  const radarData = evalData
    ? [
        { metric: 'Accuracy', value: evalData.accuracy, fullMark: 1 },
        { metric: 'Precision', value: evalData.precision, fullMark: 1 },
        { metric: 'Recall', value: evalData.recall, fullMark: 1 },
        { metric: 'F1 Score', value: evalData.f1_score, fullMark: 1 },
        { metric: 'ROC-AUC', value: evalData.auc_roc, fullMark: 1 },
      ]
    : [];

  const baselineData = useMemo(() => {
    if (experiments.length === 0) return [];

    const algoMap = new Map<string, { best: number; current: number }>();
    for (const exp of experiments) {
      const existing = algoMap.get(exp.algorithm);
      if (!existing || exp.best_accuracy > existing.best) {
        algoMap.set(exp.algorithm, { best: exp.best_accuracy, current: exp.current_round / Math.max(exp.total_rounds, 1) });
      }
    }

    const sorted = ALGORITHM_ORDER.filter((a) => algoMap.has(a));
    const other = Array.from(algoMap.entries())
      .filter(([a]) => !ALGORITHM_ORDER.includes(a))
      .map(([a]) => a)
      .sort();

    return [...sorted, ...other].map((algo) => {
      const entry = algoMap.get(algo)!;
      return {
        algorithm: algo,
        accuracy: entry.best,
        label: algo === 'pFedProto' ? 'pFedProto' : algo,
      };
    });
  }, [experiments]);

  const bestModel = useMemo(() => {
    if (experiments.length === 0) return null;
    return [...experiments].sort((a, b) => b.best_accuracy - a.best_accuracy)[0];
  }, [experiments]);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-64 rounded-lg bg-surface-container-high animate-pulse" />
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {Array.from({ length: 5 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="p-6 h-[350px] flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          </Card>
          <Card className="p-6 h-[350px] flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          </Card>
        </div>
        <Card className="p-6 h-[300px] flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        </Card>
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorState
        message={error instanceof Error ? error.message : 'An unexpected error occurred.'}
        onRetry={() => refetchAll()}
      />
    );
  }

  if (!evalData) {
    return (
      <EmptyState
        message="The server returned an empty response for evaluation data."
        onRetry={() => refetchAll()}
      />
    );
  }

  const accuracyPct = formatPct(evalData.accuracy);
  const precisionPct = formatPct(evalData.precision);
  const recallPct = formatPct(evalData.recall);
  const f1Pct = formatPct(evalData.f1_score);
  const aucRocPct = formatPct(evalData.auc_roc);

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

  return (
    <div className="space-y-6">
      {showDetails && selectedExperiment && (
        <DetailPanel experiment={selectedExperiment} onClose={closeDetail} />
      )}

      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">Model Evaluation</h1>
            {evalData && <StatusBadge status={evalSummary?.status ?? 'unknown'} />}
          </div>
          <p className="text-sm text-outline mt-0.5">
            {evalUpdatedAt
              ? `Last updated ${formatTimestamp(new Date(evalUpdatedAt).toISOString())}`
              : 'Fetching live data...'}
            <span className="ml-2 text-outline text-xs">10s auto-refresh</span>
          </p>
        </div>
        <button
          onClick={() => refetchAll()}
          disabled={isRefetching}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-container-high text-on-surface text-sm font-medium hover:bg-surface-container-highest transition-colors disabled:opacity-50 cursor-pointer"
        >
          <RefreshCw size={16} className={isRefetching ? 'animate-spin' : ''} />
          {isRefetching ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">Accuracy</span>
            <Target size={18} className="text-primary" />
          </div>
          <div className="flex items-end justify-between gap-2">
            <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">{accuracyPct}</h3>
          </div>
        </Card>
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">Precision</span>
            <Activity size={18} className="text-secondary" />
          </div>
          <div className="flex items-end justify-between gap-2">
            <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">{precisionPct}</h3>
          </div>
        </Card>
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">Recall</span>
            <Activity size={18} className="text-tertiary" />
          </div>
          <div className="flex items-end justify-between gap-2">
            <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">{recallPct}</h3>
          </div>
        </Card>
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">F1 Score</span>
            <BarChart3 size={18} className="text-primary" />
          </div>
          <div className="flex items-end justify-between gap-2">
            <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">{f1Pct}</h3>
          </div>
        </Card>
        <Card className="p-5 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-outline uppercase tracking-wider">ROC-AUC</span>
            <Activity size={18} className="text-secondary" />
          </div>
          <div className="flex items-end justify-between gap-2">
            <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">{aucRocPct}</h3>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <Card className="p-4 flex flex-col gap-2">
          <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Comm Round</span>
          <h3 className="text-xl font-display font-bold text-on-surface font-mono">R-{evalData.round}</h3>
        </Card>
        <Card className="p-4 flex flex-col gap-2">
          <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Samples</span>
          <h3 className="text-xl font-display font-bold text-on-surface font-mono">{evalData.samples_evaluated.toLocaleString()}</h3>
        </Card>
        <Card className="p-4 flex flex-col gap-2">
          <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Client</span>
          <h3 className="text-xl font-display font-bold text-on-surface font-mono">{evalData.client_id}</h3>
        </Card>
        <Card className="p-4 flex flex-col gap-2 col-span-1 lg:col-span-2">
          <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Evaluation Time</span>
          <h3 className="text-xl font-display font-bold text-on-surface font-mono">
            {formatTimestamp(new Date(evalUpdatedAt).toISOString())}
          </h3>
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
                <h3 className="text-sm font-semibold text-on-surface">Precision / Recall / F1</h3>
                <p className="text-xs text-outline">Classification metrics comparison</p>
              </div>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="key"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'var(--color-outline)', fontSize: 11 }}
                  tickFormatter={(v: string) => getChartLabel(v)}
                />
                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
                  domain={[0, 1]}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--color-surface-container-highest)',
                    borderColor: 'var(--color-outline-variant)',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}
                  formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, '']}
                  labelFormatter={(label: string) => getChartLabel(label)}
                />
                <Bar
                  dataKey="value"
                  radius={[6, 6, 0, 0]}
                  barSize={60}
                >
                  {barData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6 h-[350px] flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                <Activity size={20} />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-on-surface">Performance Radar</h3>
                <p className="text-xs text-outline">Multi-metric model evaluation</p>
              </div>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.08)" />
                <PolarAngleAxis
                  dataKey="metric"
                  tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
                />
                <PolarRadiusAxis
                  angle={90}
                  domain={[0, 1]}
                  tick={{ fill: 'var(--color-outline)', fontSize: 9 }}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--color-surface-container-highest)',
                    borderColor: 'var(--color-outline-variant)',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}
                  formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, '']}
                />
                <Radar
                  name="Model Performance"
                  dataKey="value"
                  stroke="var(--color-primary)"
                  fill="var(--color-primary)"
                  fillOpacity={0.2}
                  strokeWidth={2}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {baselineData.length > 0 && (
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
              <Layers size={20} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-on-surface">Baseline Comparison</h3>
              <p className="text-xs text-outline">Best accuracy across algorithms from experiment runs</p>
            </div>
          </div>

          {bestModel && (
            <div className="mb-6 p-4 bg-emerald-500/5 border border-emerald-500/20 rounded-xl flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center">
                <TrendingUp size={20} className="text-emerald-400" />
              </div>
              <div>
                <p className="text-xs text-emerald-400 font-bold uppercase tracking-wider">Best Model</p>
                <p className="text-sm font-semibold text-on-surface">
                  {bestModel.name} — <span className="text-emerald-400 font-mono">{formatAccuracy(bestModel.best_accuracy)}</span>
                </p>
                <p className="text-[11px] text-outline">{bestModel.algorithm} · {bestModel.num_clients} clients · {bestModel.total_rounds} rounds</p>
              </div>
            </div>
          )}

          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={baselineData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  type="number"
                  domain={[0, 1]}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                />
                <YAxis
                  type="category"
                  dataKey="label"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'var(--color-on-surface)', fontSize: 11, fontWeight: 600 }}
                  width={120}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--color-surface-container-highest)',
                    borderColor: 'var(--color-outline-variant)',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}
                  formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Best Accuracy']}
                />
                <Bar
                  dataKey="accuracy"
                  radius={[0, 6, 6, 0]}
                  barSize={28}
                >
                  {baselineData.map((_, idx) => {
                    const colors = ['var(--color-primary)', 'var(--color-secondary)', 'var(--color-tertiary)', '#f472b6', '#34d399', '#fbbf24'];
                    return <Cell key={idx} fill={colors[idx % colors.length]} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      <Card className="overflow-hidden">
        <div className="px-6 py-4 border-b border-outline-variant flex items-center justify-between bg-surface-container-low/50 flex-wrap gap-3">
          <h3 className="text-sm font-bold text-on-surface flex items-center gap-2">
            <Server size={16} />
            Experiment Runs
          </h3>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-outline" />
              <input
                type="text"
                placeholder="Search experiments..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-44 pl-8 pr-3 py-1.5 rounded-lg bg-surface-container-high border border-outline-variant/50 text-xs text-on-surface placeholder-outline focus:outline-none focus:border-primary/50 transition-all"
              />
            </div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
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
              onChange={(e) => setAlgorithmFilter(e.target.value)}
              className="px-2.5 py-1.5 rounded-lg bg-surface-container-high border border-outline-variant/50 text-xs text-on-surface focus:outline-none focus:border-primary/50 transition-all"
            >
              <option value="all">All Algorithms</option>
              {uniqueAlgorithms.map((algo) => (
                <option key={algo} value={algo}>{algo}</option>
              ))}
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
                  <th className="px-4 py-4 font-bold border-b border-outline-variant text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/30 text-xs">
                {filteredExperiments.map((exp) => (
                  <tr
                    key={exp.id}
                    className="hover:bg-surface-container-high/20 transition-colors"
                  >
                    <td className="px-4 py-4 font-medium text-on-surface max-w-[220px] truncate" title={exp.name}>
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
        )}
      </Card>

      {evalData && (
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
              <Activity size={20} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-on-surface">Evaluation Metadata</h3>
              <p className="text-xs text-outline">Inference statistics and evaluation configuration</p>
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-surface-container-high/50 rounded-xl p-4">
              <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Client ID</span>
              <p className="text-lg font-mono font-bold text-on-surface mt-1">{evalData.client_id}</p>
            </div>
            <div className="bg-surface-container-high/50 rounded-xl p-4">
              <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Comm Round</span>
              <p className="text-lg font-mono font-bold text-on-surface mt-1">R-{evalData.round}</p>
            </div>
            <div className="bg-surface-container-high/50 rounded-xl p-4">
              <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Samples</span>
              <p className="text-lg font-mono font-bold text-on-surface mt-1">{evalData.samples_evaluated.toLocaleString()}</p>
            </div>
            <div className="bg-surface-container-high/50 rounded-xl p-4">
              <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Avg Accuracy</span>
              <p className="text-lg font-mono font-bold text-on-surface mt-1">{accuracyPct}</p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};
