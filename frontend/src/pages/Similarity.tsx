import { useState, useMemo, useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Search, RefreshCw, SlidersHorizontal, X, Eye, Cpu,
  Layers, Activity, AlertTriangle, Database, ArrowRightLeft,
  BarChart3, TrendingUp, Grid3X3, PieChart as PieChartIcon,
  ArrowRight, Clock, CheckCircle2, XCircle, Target,
  GitBranch, Network, Users, TrendingDown, Share2,
} from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, LineChart, Line, PieChart,
  Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ScatterChart, Scatter, AreaChart, Area,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts';

import {
  fetchSimilarityAnalyses,
  fetchSimilarityStatistics,
} from '../api/similarity';
import { Card } from '../components/ui/Card';
import { StatusBadge } from '../components/ui/StatusBadge';
import type { SimilarityAnalysis, SimilarityStatistics } from '../types';

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

const CLUSTER_COLORS = [
  'var(--color-primary)',
  'var(--color-secondary)',
  '#a78bfa',
  '#f87171',
  '#34d399',
  '#fbbf24',
  '#fb923c',
  '#22d3ee',
];

const STATUS_STYLES: Record<string, string> = {
  completed: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  running: 'bg-primary-container/20 text-primary border-primary/30',
  failed: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
  pending: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
};

function getModalityLabel(m: string): string {
  return MODALITY_LABELS[m] || m;
}

function getMetricLabel(m: string): string {
  const labels: Record<string, string> = {
    cosine: 'Cosine Similarity',
    euclidean: 'Euclidean Distance',
    manhattan: 'Manhattan Distance',
    dot: 'Dot Product',
    mse: 'MSE',
    l1: 'L1 Distance',
  };
  return labels[m] || m;
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

type SortField = 'analysis_id' | 'source_client' | 'target_client' | 'source_prototype' | 'target_prototype' | 'source_modality' | 'target_modality' | 'similarity_metric' | 'cosine_similarity' | 'euclidean_distance' | 'prototype_distance' | 'transfer_confidence' | 'aggregation_round' | 'cluster_id' | 'analysis_status' | 'created_at';

/* ====================================================================== */
/*  Skeleton                                                               */
/* ====================================================================== */

function SkeletonRow() {
  return (
    <tr className="animate-pulse border-b border-outline-variant/30">
      {Array.from({ length: 10 }).map((_, i) => (
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
        <Network size={32} className="text-outline" />
      </div>
      <h3 className="text-lg font-display font-bold text-on-surface">No similarity analyses found</h3>
      <p className="text-sm text-outline max-w-sm text-center">
        No similarity data is available for the current configuration.
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
      <h3 className="text-lg font-display font-bold text-on-surface">Failed to load similarity analyses</h3>
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

function DetailPanel({ analysis: a, onClose }: { analysis: SimilarityAnalysis; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/40 backdrop-blur-sm">
      <Card className="w-full max-w-lg h-full overflow-y-auto rounded-none rounded-l-2xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center">
              <Activity size={20} className="text-primary" />
            </div>
            <div>
              <h2 className="text-lg font-display font-bold text-on-surface">Analysis Details</h2>
              <p className="text-[10px] font-mono text-outline uppercase">{a.analysis_id}</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-surface-container-high transition-colors cursor-pointer">
            <X size={20} className="text-outline" />
          </button>
        </div>

        {/* Similarity Metrics */}
        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Similarity Metrics</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Cosine Similarity</p>
              <p className="text-lg font-display font-bold text-primary">{(a.cosine_similarity * 100).toFixed(1)}%</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Euclidean Distance</p>
              <p className="text-lg font-display font-bold text-secondary">{a.euclidean_distance.toFixed(4)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Prototype Distance</p>
              <p className="text-lg font-display font-bold text-amber-400">{a.prototype_distance.toFixed(4)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Transfer Confidence</p>
              <p className="text-lg font-display font-bold text-emerald-400">{(a.transfer_confidence * 100).toFixed(1)}%</p>
            </div>
          </div>
        </div>

        {/* Distance Metrics */}
        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Distance Metrics</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Similarity Metric</p>
              <p className="text-sm font-semibold text-on-surface">{getMetricLabel(a.similarity_metric)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Pairwise Distance</p>
              <p className="text-sm font-semibold text-on-surface">{(a.euclidean_distance + a.prototype_distance / 2).toFixed(4)}</p>
            </div>
          </div>
        </div>

        {/* Prototype Statistics */}
        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Prototype Statistics</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Source Prototype</p>
              <p className="text-sm font-semibold text-on-surface">{a.source_prototype}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Target Prototype</p>
              <p className="text-sm font-semibold text-on-surface">{a.target_prototype}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Cluster ID</p>
              <p className="text-sm font-semibold text-on-surface">Cluster #{a.cluster_id}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Similarity Metric</p>
              <StatusBadge status={a.similarity_metric} />
            </div>
          </div>
        </div>

        {/* Client Information */}
        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Client Information</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Source Client</p>
              <p className="text-sm font-semibold text-on-surface">{a.source_client}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Target Client</p>
              <p className="text-sm font-semibold text-on-surface">{a.target_client}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Source Modality</p>
              <p className="text-sm font-semibold text-on-surface">{getModalityLabel(a.source_modality)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Target Modality</p>
              <p className="text-sm font-semibold text-on-surface">{getModalityLabel(a.target_modality)}</p>
            </div>
          </div>
        </div>

        {/* Transfer Relationship */}
        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Transfer Relationship</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Analysis ID</p>
              <p className="text-sm font-mono font-semibold text-on-surface">{a.analysis_id}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Status</p>
              <StatusBadge status={a.analysis_status} />
            </div>
          </div>
        </div>

        {/* Aggregation History */}
        <div className="space-y-3">
          <h3 className="text-sm font-bold text-on-surface">Aggregation History</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Aggregation Round</p>
              <p className="text-sm font-semibold text-on-surface">Round #{a.aggregation_round}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Created Time</p>
              <p className="text-sm font-semibold text-on-surface">{formatTimestamp(a.created_at)}</p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

/* ====================================================================== */
/*  Similarity Heatmap                                                     */
/* ====================================================================== */

function SimilarityHeatmap({ analyses }: { analyses: SimilarityAnalysis[] }) {
  const data = useMemo(() => {
    const pairs: Record<string, { source: string; target: string; avgSimilarity: number; count: number }> = {};
    analyses.forEach((a) => {
      const key = `${a.source_modality}→${a.target_modality}`;
      if (!pairs[key]) {
        pairs[key] = { source: a.source_modality, target: a.target_modality, avgSimilarity: 0, count: 0 };
      }
      pairs[key].avgSimilarity += a.cosine_similarity;
      pairs[key].count += 1;
    });
    return Object.entries(pairs).map(([k, v]) => ({
      pair: k,
      avgSimilarity: parseFloat((v.avgSimilarity / v.count).toFixed(3)),
      count: v.count,
    }));
  }, [analyses]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No similarity data available</p>;

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="pair" tick={{ fill: 'var(--color-outline)', fontSize: 9 }} axisLine={false} tickLine={false} angle={-20} textAnchor="end" height={60} />
        <YAxis domain={[0, 1]} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
          formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Avg Cosine Similarity']}
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
/*  Client Similarity Matrix                                               */
/* ====================================================================== */

function ClientSimilarityMatrix({ analyses }: { analyses: SimilarityAnalysis[] }) {
  const data = useMemo(() => {
    const pairs: Record<string, { source: string; target: string; avgSimilarity: number; count: number }> = {};
    analyses.forEach((a) => {
      const key = `${a.source_client}→${a.target_client}`;
      if (!pairs[key]) {
        pairs[key] = { source: a.source_client, target: a.target_client, avgSimilarity: 0, count: 0 };
      }
      pairs[key].avgSimilarity += a.cosine_similarity;
      pairs[key].count += 1;
    });
    return Object.entries(pairs)
      .map(([k, v]) => ({
        pair: k,
        avgSimilarity: parseFloat((v.avgSimilarity / v.count).toFixed(3)),
        count: v.count,
      }))
      .sort((a, b) => b.avgSimilarity - a.avgSimilarity)
      .slice(0, 10);
  }, [analyses]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No client similarity data available</p>;

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={data} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis type="number" domain={[0, 1]} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
        <YAxis type="category" dataKey="pair" tick={{ fill: 'var(--color-outline)', fontSize: 9 }} axisLine={false} tickLine={false} width={120} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
          formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Avg Cosine Similarity']}
        />
        <Bar dataKey="avgSimilarity" radius={[0, 4, 4, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ====================================================================== */
/*  Prototype Similarity Matrix                                            */
/* ====================================================================== */

function PrototypeSimilarityMatrix({ analyses }: { analyses: SimilarityAnalysis[] }) {
  const matrix = useMemo(() => {
    const prototypes = Array.from(new Set(analyses.flatMap((a) => [a.source_prototype, a.target_prototype]))).sort();
    const pairs: Record<string, { count: number; totalSimilarity: number }> = {};
    analyses.forEach((a) => {
      const key = `${a.source_prototype}→${a.target_prototype}`;
      if (!pairs[key]) pairs[key] = { count: 0, totalSimilarity: 0 };
      pairs[key].count += 1;
      pairs[key].totalSimilarity += a.cosine_similarity;
    });
    const rows = prototypes.map((src) => {
      const row: Record<string, string | number> = { source: src };
      prototypes.forEach((tgt) => {
        const key = `${src}→${tgt}`;
        const p = pairs[key];
        row[tgt] = p ? parseFloat((p.totalSimilarity / p.count).toFixed(3)) : 0;
      });
      return row;
    });
    return { prototypes, rows };
  }, [analyses]);

  if (matrix.prototypes.length === 0) return <p className="text-xs text-outline text-center py-4">No prototype similarity data available</p>;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr>
            <th className="p-1.5 text-outline text-[10px] font-bold uppercase" />
            {matrix.prototypes.map((p) => (
              <th key={p} className="p-1.5 text-outline text-[10px] font-mono font-bold text-center">{p}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.rows.map((row) => (
            <tr key={row.source as string}>
              <td className="p-1.5 text-outline text-[10px] font-mono font-bold">{row.source as string}</td>
              {matrix.prototypes.map((proto) => {
                const val = row[proto] as number;
                const intensity = Math.round(Math.min(val, 1) * 100);
                return (
                  <td
                    key={proto}
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
/*  Cluster Visualization                                                  */
/* ====================================================================== */

function ClusterVisualization({ analyses }: { analyses: SimilarityAnalysis[] }) {
  const data = useMemo(() => {
    return analyses.map((a) => ({
      cosine_similarity: a.cosine_similarity,
      euclidean_distance: a.euclidean_distance,
      cluster_id: a.cluster_id,
      analysis_id: a.analysis_id,
    }));
  }, [analyses]);

  const uniqueClusters = useMemo(() => {
    return Array.from(new Set(data.map((d) => d.cluster_id))).sort();
  }, [data]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No cluster data available</p>;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis
          dataKey="cosine_similarity"
          domain={[0, 1]}
          tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          name="Cosine Similarity"
        />
        <YAxis
          dataKey="euclidean_distance"
          tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          name="Euclidean Distance"
        />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
          formatter={(value: number, name: string) => [
            name === 'cosine_similarity' ? `${(value * 100).toFixed(1)}%` : value.toFixed(4),
            name === 'cosine_similarity' ? 'Cosine Similarity' : 'Euclidean Distance',
          ]}
        />
        <Legend />
        {uniqueClusters.map((clusterId) => {
          const clusterData = data.filter((d) => d.cluster_id === clusterId);
          return (
            <Scatter
              key={clusterId}
              name={`Cluster ${clusterId}`}
              data={clusterData}
              fill={CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length]}
            />
          );
        })}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/* ====================================================================== */
/*  Similarity Timeline                                                    */
/* ====================================================================== */

function SimilarityTimeline({ analyses }: { analyses: SimilarityAnalysis[] }) {
  const data = useMemo(() => {
    const grouped: Record<number, { round: number; similarities: number[] }> = {};
    analyses.forEach((a) => {
      const r = a.aggregation_round;
      if (!grouped[r]) {
        grouped[r] = { round: r, similarities: [] };
      }
      grouped[r].similarities.push(a.cosine_similarity);
    });
    return Object.values(grouped)
      .map((g) => {
        const avg = g.similarities.reduce((s, v) => s + v, 0) / g.similarities.length;
        const min = Math.min(...g.similarities);
        const max = Math.max(...g.similarities);
        return {
          round: `R${g.round}`,
          avgSimilarity: parseFloat((avg * 100).toFixed(1)),
          minSimilarity: parseFloat((min * 100).toFixed(1)),
          maxSimilarity: parseFloat((max * 100).toFixed(1)),
          count: g.similarities.length,
        };
      })
      .sort((a, b) => parseInt(a.round.slice(1)) - parseInt(b.round.slice(1)));
  }, [analyses]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No timeline data available</p>;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="round" tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis domain={[0, 100]} tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${v}%`} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
          formatter={(value: number, name: string) => {
            const labels: Record<string, string> = {
              avgSimilarity: 'Avg Similarity',
              minSimilarity: 'Min Similarity',
              maxSimilarity: 'Max Similarity',
            };
            return [`${value}%`, labels[name] || name];
          }}
        />
        <Legend />
        <Line type="monotone" dataKey="avgSimilarity" stroke="var(--color-primary)" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} name="avgSimilarity" />
        <Line type="monotone" dataKey="minSimilarity" stroke="#f87171" strokeWidth={1} dot={false} name="minSimilarity" />
        <Line type="monotone" dataKey="maxSimilarity" stroke="#34d399" strokeWidth={1} dot={false} name="maxSimilarity" />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ====================================================================== */
/*  Distribution Histogram                                                 */
/* ====================================================================== */

function DistributionHistogram({ analyses }: { analyses: SimilarityAnalysis[] }) {
  const data = useMemo(() => {
    const bins: Record<string, { range: string; count: number; min: number; max: number }> = {
      '0.0-0.2': { range: '0.0-0.2', count: 0, min: 0, max: 0.2 },
      '0.2-0.4': { range: '0.2-0.4', count: 0, min: 0.2, max: 0.4 },
      '0.4-0.6': { range: '0.4-0.6', count: 0, min: 0.4, max: 0.6 },
      '0.6-0.8': { range: '0.6-0.8', count: 0, min: 0.6, max: 0.8 },
      '0.8-1.0': { range: '0.8-1.0', count: 0, min: 0.8, max: 1.0 },
    };
    analyses.forEach((a) => {
      const v = a.cosine_similarity;
      if (v < 0.2) bins['0.0-0.2'].count += 1;
      else if (v < 0.4) bins['0.2-0.4'].count += 1;
      else if (v < 0.6) bins['0.4-0.6'].count += 1;
      else if (v < 0.8) bins['0.6-0.8'].count += 1;
      else bins['0.8-1.0'].count += 1;
    });
    return Object.values(bins);
  }, [analyses]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No distribution data available</p>;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="range" tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: 'var(--color-outline)', fontSize: 10 }} axisLine={false} tickLine={false} allowDecimals={false} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
          formatter={(value: number) => [value, 'Count']}
        />
        <Bar dataKey="count" fill="var(--color-primary)" radius={[4, 4, 0, 0]} name="Count" />
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ====================================================================== */
/*  Radar Comparison                                                       */
/* ====================================================================== */

function RadarComparison({ analyses }: { analyses: SimilarityAnalysis[] }) {
  const data = useMemo(() => {
    const grouped: Record<string, { cosineTotal: number; confidenceTotal: number; invEuclideanTotal: number; count: number }> = {};
    analyses.forEach((a) => {
      const mods = [a.source_modality, a.target_modality];
      mods.forEach((m) => {
        if (!grouped[m]) {
          grouped[m] = { cosineTotal: 0, confidenceTotal: 0, invEuclideanTotal: 0, count: 0 };
        }
        grouped[m].cosineTotal += a.cosine_similarity;
        grouped[m].confidenceTotal += a.transfer_confidence;
        grouped[m].invEuclideanTotal += 1 / (1 + a.euclidean_distance);
        grouped[m].count += 1;
      });
    });
    return Object.entries(grouped).map(([modality, g]) => ({
      modality: getModalityLabel(modality),
      avgCosine: parseFloat(((g.cosineTotal / g.count) * 100).toFixed(1)),
      avgConfidence: parseFloat(((g.confidenceTotal / g.count) * 100).toFixed(1)),
      avgInvDistance: parseFloat(((g.invEuclideanTotal / g.count) * 100).toFixed(1)),
    }));
  }, [analyses]);

  if (data.length === 0) return <p className="text-xs text-outline text-center py-4">No radar data available</p>;

  return (
    <ResponsiveContainer width="100%" height={250}>
      <RadarChart data={data}>
        <PolarGrid stroke="rgba(255,255,255,0.1)" />
        <PolarAngleAxis dataKey="modality" tick={{ fill: 'var(--color-outline)', fontSize: 10 }} />
        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: 'var(--color-outline)', fontSize: 9 }} tickFormatter={(v: number) => `${v}%`} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--color-surface-container-highest)', border: 'none', borderRadius: '8px', fontSize: '12px' }}
        />
        <Radar name="Cosine Similarity" dataKey="avgCosine" stroke="var(--color-primary)" fill="var(--color-primary)" fillOpacity={0.3} />
        <Radar name="Transfer Confidence" dataKey="avgConfidence" stroke="var(--color-secondary)" fill="var(--color-secondary)" fillOpacity={0.3} />
        <Radar name="Inv. Distance" dataKey="avgInvDistance" stroke="#34d399" fill="#34d399" fillOpacity={0.3} />
        <Legend />
      </RadarChart>
    </ResponsiveContainer>
  );
}

/* ====================================================================== */
/*  Main Similarity Analysis Page                                          */
/* ====================================================================== */

export const Similarity = () => {
  const [search, setSearch] = useState('');
  const [clientFilter, setClientFilter] = useState('');
  const [prototypeFilter, setPrototypeFilter] = useState('');
  const [metricFilter, setMetricFilter] = useState('');
  const [modalityFilter, setModalityFilter] = useState('');
  const [roundFilter, setRoundFilter] = useState('');
  const [clusterFilter, setClusterFilter] = useState('');
  const [similarityMin, setSimilarityMin] = useState('');
  const [similarityMax, setSimilarityMax] = useState('');
  const [sortField, setSortField] = useState<SortField>('created_at');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(10);
  const [selectedAnalysis, setSelectedAnalysis] = useState<SimilarityAnalysis | null>(null);
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
    queryKey: ['similarity-analyses'],
    queryFn: fetchSimilarityAnalyses,
    refetchInterval: AUTO_REFRESH_MS,
    retry: 2,
    staleTime: 3000,
  });

  const {
    data: statsResponse,
    isLoading: statsLoading,
  } = useQuery({
    queryKey: ['similarity-statistics'],
    queryFn: fetchSimilarityStatistics,
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

  const allAnalyses: SimilarityAnalysis[] = listResponse?.data ?? [];
  const statistics: SimilarityStatistics | null = statsResponse?.data ?? null;

  const uniqueClients = useMemo(() => {
    return Array.from(new Set(allAnalyses.flatMap((a) => [a.source_client, a.target_client]))).sort();
  }, [allAnalyses]);

  const uniquePrototypes = useMemo(() => {
    return Array.from(new Set(allAnalyses.flatMap((a) => [a.source_prototype, a.target_prototype]))).sort();
  }, [allAnalyses]);

  const uniqueMetrics = useMemo(() => {
    return Array.from(new Set(allAnalyses.map((a) => a.similarity_metric))).sort();
  }, [allAnalyses]);

  const uniqueModalities = useMemo(() => {
    return Array.from(new Set(allAnalyses.flatMap((a) => [a.source_modality, a.target_modality]))).sort();
  }, [allAnalyses]);

  const uniqueRounds = useMemo(() => {
    return Array.from(new Set(allAnalyses.map((a) => a.aggregation_round))).sort((a, b) => a - b);
  }, [allAnalyses]);

  const uniqueClusters = useMemo(() => {
    return Array.from(new Set(allAnalyses.map((a) => a.cluster_id))).sort((a, b) => a - b);
  }, [allAnalyses]);

  const filtered = useMemo(() => {
    let list = allAnalyses;

    if (search) {
      const q = search.toLowerCase();
      list = list.filter(
        (a) =>
          a.analysis_id.toLowerCase().includes(q) ||
          a.source_client.toLowerCase().includes(q) ||
          a.target_client.toLowerCase().includes(q) ||
          a.source_prototype.toLowerCase().includes(q) ||
          a.target_prototype.toLowerCase().includes(q) ||
          a.source_modality.toLowerCase().includes(q) ||
          a.target_modality.toLowerCase().includes(q) ||
          a.similarity_metric.toLowerCase().includes(q) ||
          a.analysis_status.toLowerCase().includes(q),
      );
    }

    if (clientFilter) list = list.filter((a) => a.source_client === clientFilter || a.target_client === clientFilter);
    if (prototypeFilter) list = list.filter((a) => a.source_prototype === prototypeFilter || a.target_prototype === prototypeFilter);
    if (metricFilter) list = list.filter((a) => a.similarity_metric === metricFilter);
    if (modalityFilter) list = list.filter((a) => a.source_modality === modalityFilter || a.target_modality === modalityFilter);
    if (roundFilter) list = list.filter((a) => a.aggregation_round === Number(roundFilter));
    if (clusterFilter) list = list.filter((a) => a.cluster_id === Number(clusterFilter));
    if (similarityMin) {
      const minVal = parseFloat(similarityMin);
      if (!isNaN(minVal)) list = list.filter((a) => a.cosine_similarity >= minVal);
    }
    if (similarityMax) {
      const maxVal = parseFloat(similarityMax);
      if (!isNaN(maxVal)) list = list.filter((a) => a.cosine_similarity <= maxVal);
    }

    list.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case 'analysis_id': cmp = a.analysis_id.localeCompare(b.analysis_id); break;
        case 'source_client': cmp = a.source_client.localeCompare(b.source_client); break;
        case 'target_client': cmp = a.target_client.localeCompare(b.target_client); break;
        case 'source_prototype': cmp = a.source_prototype.localeCompare(b.source_prototype); break;
        case 'target_prototype': cmp = a.target_prototype.localeCompare(b.target_prototype); break;
        case 'source_modality': cmp = a.source_modality.localeCompare(b.source_modality); break;
        case 'target_modality': cmp = a.target_modality.localeCompare(b.target_modality); break;
        case 'similarity_metric': cmp = a.similarity_metric.localeCompare(b.similarity_metric); break;
        case 'cosine_similarity': cmp = a.cosine_similarity - b.cosine_similarity; break;
        case 'euclidean_distance': cmp = a.euclidean_distance - b.euclidean_distance; break;
        case 'prototype_distance': cmp = a.prototype_distance - b.prototype_distance; break;
        case 'transfer_confidence': cmp = a.transfer_confidence - b.transfer_confidence; break;
        case 'aggregation_round': cmp = a.aggregation_round - b.aggregation_round; break;
        case 'cluster_id': cmp = a.cluster_id - b.cluster_id; break;
        case 'analysis_status': cmp = a.analysis_status.localeCompare(b.analysis_status); break;
        case 'created_at': cmp = a.created_at.localeCompare(b.created_at); break;
        default: cmp = 0;
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return list;
  }, [allAnalyses, search, clientFilter, prototypeFilter, metricFilter, modalityFilter, roundFilter, clusterFilter, similarityMin, similarityMax, sortField, sortDir]);

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

  const hasFilters = search || clientFilter || prototypeFilter || metricFilter || modalityFilter || roundFilter || clusterFilter || similarityMin || similarityMax;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">Similarity Analysis</h1>
          <p className="text-sm text-outline">Cross-client prototype similarity and transfer analysis</p>
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
            placeholder="Search by ID, client, prototype, modality..."
            className="w-full pl-9 pr-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm placeholder:text-outline/50 focus:outline-none focus:border-primary transition-colors"
          />
        </div>
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
          value={prototypeFilter}
          onChange={(e) => { setPrototypeFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Prototypes</option>
          {uniquePrototypes.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
        <select
          value={metricFilter}
          onChange={(e) => { setMetricFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Metrics</option>
          {uniqueMetrics.map((m) => (
            <option key={m} value={m}>{getMetricLabel(m)}</option>
          ))}
        </select>
        <select
          value={modalityFilter}
          onChange={(e) => { setModalityFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Modalities</option>
          {uniqueModalities.map((m) => (
            <option key={m} value={m}>{getModalityLabel(m)}</option>
          ))}
        </select>
        <select
          value={roundFilter}
          onChange={(e) => { setRoundFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Rounds</option>
          {uniqueRounds.map((r) => (
            <option key={r} value={r}>Round #{r}</option>
          ))}
        </select>
        <select
          value={clusterFilter}
          onChange={(e) => { setClusterFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Clusters</option>
          {uniqueClusters.map((c) => (
            <option key={c} value={c}>Cluster #{c}</option>
          ))}
        </select>
        <div className="flex items-center gap-2">
          <input
            type="number"
            value={similarityMin}
            onChange={(e) => { setSimilarityMin(e.target.value); setPage(0); }}
            placeholder="Min sim"
            min={0}
            max={1}
            step={0.1}
            className="w-20 px-2 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-xs placeholder:text-outline/50 focus:outline-none focus:border-primary transition-colors"
          />
          <span className="text-outline text-xs">-</span>
          <input
            type="number"
            value={similarityMax}
            onChange={(e) => { setSimilarityMax(e.target.value); setPage(0); }}
            placeholder="Max sim"
            min={0}
            max={1}
            step={0.1}
            className="w-20 px-2 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-xs placeholder:text-outline/50 focus:outline-none focus:border-primary transition-colors"
          />
        </div>
        <div className="flex items-center gap-2 text-xs text-outline">
          <SlidersHorizontal size={14} />
          <span>{filtered.length} analysis{filtered.length !== 1 ? 'es' : ''}</span>
        </div>
        {hasFilters && (
          <button
            onClick={() => { setSearch(''); setClientFilter(''); setPrototypeFilter(''); setMetricFilter(''); setModalityFilter(''); setRoundFilter(''); setClusterFilter(''); setSimilarityMin(''); setSimilarityMax(''); setPage(0); }}
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
                  {['Analysis ID', 'Source Client', 'Target Client', 'Source Mod', 'Target Mod', 'Metric', 'Cosine', 'Euclidean', 'Confidence', 'Status'].map((h) => (
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
                  <Activity size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Average Similarity</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (
                      statistics
                        ? `${(statistics.average_similarity * 100).toFixed(1)}%`
                        : allAnalyses.length > 0
                          ? `${(allAnalyses.reduce((s, a) => s + a.cosine_similarity, 0) / allAnalyses.length * 100).toFixed(1)}%`
                          : '—'
                    )}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-emerald-400">
                  <TrendingUp size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Maximum Similarity</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (
                      statistics
                        ? `${(statistics.maximum_similarity * 100).toFixed(1)}%`
                        : allAnalyses.length > 0
                          ? `${(Math.max(...allAnalyses.map((a) => a.cosine_similarity)) * 100).toFixed(1)}%`
                          : '—'
                    )}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-rose-400">
                  <TrendingDown size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Minimum Similarity</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (
                      statistics
                        ? `${(statistics.minimum_similarity * 100).toFixed(1)}%`
                        : allAnalyses.length > 0
                          ? `${(Math.min(...allAnalyses.map((a) => a.cosine_similarity)) * 100).toFixed(1)}%`
                          : '—'
                    )}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                  <GitBranch size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Average Distance</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (
                      statistics
                        ? statistics.average_distance.toFixed(4)
                        : allAnalyses.length > 0
                          ? (allAnalyses.reduce((s, a) => s + a.euclidean_distance, 0) / allAnalyses.length).toFixed(4)
                          : '—'
                    )}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                  <Layers size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Cluster Count</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (statistics?.cluster_count ?? uniqueClusters.length)}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-amber-400">
                  <Users size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Client Groups</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (statistics?.client_groups ?? uniqueClients.length)}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                  <Database size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Prototype Groups</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (statistics?.prototype_groups ?? uniquePrototypes.length)}
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-emerald-400">
                  <Clock size={20} />
                </div>
                <div>
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline">Communication Round</p>
                  <p className="text-xl font-display font-bold text-on-surface">
                    {statsLoading ? '—' : (
                      statistics
                        ? `Round #${statistics.communication_round}`
                        : allAnalyses.length > 0
                          ? `Round #${Math.max(...allAnalyses.map((a) => a.aggregation_round))}`
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
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('analysis_id')}>
                      Analysis ID <SortIcon field="analysis_id" />
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
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('similarity_metric')}>
                      Metric <SortIcon field="similarity_metric" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('cosine_similarity')}>
                      Cosine <SortIcon field="cosine_similarity" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('euclidean_distance')}>
                      Euclidean <SortIcon field="euclidean_distance" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('transfer_confidence')}>
                      Confidence <SortIcon field="transfer_confidence" />
                    </th>
                    <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('analysis_status')}>
                      Status <SortIcon field="analysis_status" />
                    </th>
                    <th className="px-4 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {paged.map((analysis) => (
                    <tr
                      key={analysis.analysis_id}
                      className="border-b border-outline-variant/20 hover:bg-surface-container-low transition-colors"
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-lg bg-surface-container-high flex items-center justify-center">
                            <Share2 size={16} className="text-primary" />
                          </div>
                          <div>
                            <p className="text-sm font-semibold text-on-surface">{analysis.analysis_id}</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-on-surface-variant">{analysis.source_client}</td>
                      <td className="px-4 py-3 text-sm text-on-surface-variant">{analysis.target_client}</td>
                      <td className="px-4 py-3">
                        <StatusBadge status={analysis.source_modality} />
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={analysis.target_modality} />
                      </td>
                      <td className="px-4 py-3 text-sm text-on-surface-variant">{getMetricLabel(analysis.similarity_metric)}</td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 max-w-[50px] h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary rounded-full transition-all"
                              style={{ width: `${analysis.cosine_similarity * 100}%` }}
                            />
                          </div>
                          <span className="text-sm font-mono font-semibold text-primary">{(analysis.cosine_similarity * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm font-mono text-on-surface">{analysis.euclidean_distance.toFixed(4)}</td>
                      <td className="px-4 py-3 text-sm font-mono text-on-surface">{(analysis.transfer_confidence * 100).toFixed(0)}%</td>
                      <td className="px-4 py-3">
                        <StatusBadge status={analysis.analysis_status} />
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => setSelectedAnalysis(analysis)}
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
                  <h3 className="text-sm font-semibold text-on-surface">Similarity Heatmap</h3>
                  <p className="text-xs text-outline">Average cosine similarity by modality pair</p>
                </div>
              </div>
              <SimilarityHeatmap analyses={allAnalyses} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                  <Users size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Client Similarity Matrix</h3>
                  <p className="text-xs text-outline">Top 10 client pairs by avg similarity</p>
                </div>
              </div>
              <ClientSimilarityMatrix analyses={allAnalyses} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-amber-400">
                  <Database size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Prototype Similarity Matrix</h3>
                  <p className="text-xs text-outline">Cosine similarity by source-target prototype</p>
                </div>
              </div>
              <PrototypeSimilarityMatrix analyses={allAnalyses} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                  <Activity size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Cluster Visualization</h3>
                  <p className="text-xs text-outline">Cosine similarity vs euclidean distance by cluster</p>
                </div>
              </div>
              <ClusterVisualization analyses={allAnalyses} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                  <TrendingUp size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Similarity Timeline</h3>
                  <p className="text-xs text-outline">Cosine similarity across aggregation rounds</p>
                </div>
              </div>
              <SimilarityTimeline analyses={allAnalyses} />
            </Card>

            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
                  <BarChart3 size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Distribution Histogram</h3>
                  <p className="text-xs text-outline">Cosine similarity value distribution</p>
                </div>
              </div>
              <DistributionHistogram analyses={allAnalyses} />
            </Card>
          </div>

          {/* Radar Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-1 gap-6">
            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                  <Activity size={20} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-on-surface">Radar Comparison</h3>
                  <p className="text-xs text-outline">Multi-metric comparison across modalities</p>
                </div>
              </div>
              <RadarComparison analyses={allAnalyses} />
            </Card>
          </div>
        </>
      )}

      {/* Detail panel */}
      {selectedAnalysis && (
        <DetailPanel
          analysis={selectedAnalysis}
          onClose={() => setSelectedAnalysis(null)}
        />
      )}
    </div>
  );
};
