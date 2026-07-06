import { useQuery } from '@tanstack/react-query';
import { RefreshCw, Activity, AlertTriangle, Server, Clock, BarChart3 } from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
} from 'recharts';

import { fetchDashboard } from '../api/dashboard';
import { Card } from '../components/ui/Card';
import { StatCard } from '../components/ui/StatCard';
import { StatusBadge } from '../components/ui/StatusBadge';
import type { DashboardResponse } from '../types';

const chartData = [
  { time: '00:00', load: 32, latency: 120 },
  { time: '04:00', load: 45, latency: 110 },
  { time: '08:00', load: 68, latency: 145 },
  { time: '12:00', load: 82, latency: 160 },
  { time: '16:00', load: 74, latency: 155 },
  { time: '20:00', load: 52, latency: 130 },
  { time: '23:59', load: 38, latency: 115 },
];

const privacyData = [
  { name: 'Cluster A', score: 98 },
  { name: 'Cluster B', score: 92 },
  { name: 'Cluster C', score: 85 },
  { name: 'Cluster D', score: 96 },
  { name: 'Cluster E', score: 89 },
];

function formatTimestamp(iso: string): string {
  try {
    const date = new Date(iso);
    return date.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return iso;
  }
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
      <h2 className="text-lg font-display font-bold text-on-surface">Failed to load dashboard</h2>
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

export const Dashboard = () => {
  const {
    data: summary,
    isLoading,
    isError,
    error,
    refetch,
    isRefetching,
    dataUpdatedAt,
  } = useQuery({
    queryKey: ['dashboard'],
    queryFn: fetchDashboard,
    refetchInterval: 5000,
    refetchIntervalInBackground: false,
    retry: 2,
    staleTime: 4000,
  });

  const d: DashboardResponse | null = summary?.data ?? null;

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2 p-6 h-[400px] flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          </Card>
          <Card className="p-6 h-[400px] flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          </Card>
        </div>
      </div>
    );
  }

  if (isError) {
    const msg: string =
      error instanceof Error ? error.message : 'An unexpected error occurred.';
    return <ErrorState message={msg} onRetry={() => refetch()} />;
  }

  if (!d) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
        <div className="w-14 h-14 rounded-2xl bg-outline/10 flex items-center justify-center">
          <BarChart3 size={28} className="text-outline" />
        </div>
        <h2 className="text-lg font-display font-bold text-on-surface">No dashboard data available</h2>
        <p className="text-sm text-outline">The server returned an empty response.</p>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity cursor-pointer"
        >
          <RefreshCw size={16} />
          Retry
        </button>
      </div>
    );
  }

  const accuracyPct: string = `${(d.global_accuracy * 100).toFixed(1)}%`;
  const lossVal: string = d.global_loss.toFixed(4);
  const clientsStr: string = `${d.active_clients} / ${d.total_clients}`;
  const roundStr: string = `${d.current_round} / ${d.total_rounds}`;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-display font-bold text-on-surface">Overview</h2>
          <p className="text-xs text-outline mt-0.5">
            {dataUpdatedAt
              ? `Last updated ${formatTimestamp(new Date(dataUpdatedAt).toISOString())}`
              : 'Fetching live data...'}
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

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Active Clients" value={clientsStr} icon={<Server size={18} />} />
        <StatCard label="Current Round" value={roundStr} icon={<Activity size={18} />} />
        <StatCard label="Global Accuracy" value={accuracyPct} icon={<BarChart3 size={18} />} />
        <StatCard label="Training Loss" value={lossVal} icon={<BarChart3 size={18} />} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2 p-6 h-[400px] flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-sm font-semibold text-on-surface">System Performance</h3>
              <p className="text-xs text-outline">Real-time aggregate load and latency metrics</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-primary" />
                <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Load</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-secondary" />
                <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Latency</span>
              </div>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorLoad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--color-primary)" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="var(--color-primary)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorLatency" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--color-secondary)" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="var(--color-secondary)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="time"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
                  dy={10}
                />
                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--color-surface-container-highest)',
                    borderColor: 'var(--color-outline-variant)',
                    borderRadius: '12px',
                    fontSize: '12px',
                    color: 'var(--color-on-surface)',
                  }}
                  itemStyle={{ color: 'var(--color-on-surface)' }}
                />
                <Area
                  type="monotone"
                  dataKey="load"
                  stroke="var(--color-primary)"
                  fillOpacity={1}
                  fill="url(#colorLoad)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="latency"
                  stroke="var(--color-secondary)"
                  fillOpacity={1}
                  fill="url(#colorLatency)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-sm font-semibold text-on-surface">System Status</h3>
          </div>
          <div className="space-y-5">
            <div className="flex items-center justify-between">
              <span className="text-xs text-outline">Training Status</span>
              <StatusBadge status={d.training_status} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-outline">Experiments Running</span>
              <span className="text-sm font-semibold text-on-surface font-mono">{d.experiments_running}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-outline">Active Clients</span>
              <span className="text-sm font-semibold text-on-surface font-mono">
                {d.active_clients}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-outline">Total Clients</span>
              <span className="text-sm font-semibold text-on-surface font-mono">{d.total_clients}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-outline">Uptime</span>
              <span className="text-sm font-semibold text-on-surface font-mono">
                {d.uptime_hours.toFixed(1)}h
              </span>
            </div>
            <div className="border-t border-outline-variant/50 pt-4 mt-4">
              <div className="flex items-center gap-2 text-[10px] text-outline uppercase tracking-wider font-bold mb-2">
                <Clock size={12} />
                Last Updated
              </div>
              <p className="text-xs text-on-surface font-mono">
                {formatTimestamp(d.last_updated)}
              </p>
            </div>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
              <BarChart3 size={20} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-on-surface">Privacy Score Distribution</h3>
              <p className="text-xs text-outline">Anonymization effectiveness across clusters</p>
            </div>
          </div>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={privacyData}>
                <XAxis dataKey="name" hide />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--color-surface-container-highest)',
                    borderColor: 'var(--color-outline-variant)',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="score" fill="var(--color-primary)" radius={[4, 4, 0, 0]} barSize={32} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6 relative overflow-hidden group">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
              <Clock size={20} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-on-surface">Round Progress</h3>
              <p className="text-xs text-outline">Training round {d.current_round} of {d.total_rounds}</p>
            </div>
          </div>
          <div className="flex items-baseline gap-2 mb-4">
            <span className="text-4xl font-display font-bold text-on-surface">
              {d.total_rounds > 0
                ? `${((d.current_round / d.total_rounds) * 100).toFixed(1)}%`
                : '—'}
            </span>
            <span className="text-xs text-emerald-400 font-medium">
              {d.active_clients} / {d.total_clients} clients active
            </span>
          </div>
          <div className="w-full h-2 rounded-full bg-surface-container-high overflow-hidden">
            <div
              className="h-full rounded-full bg-secondary transition-all duration-700"
              style={{
                width:
                  d.total_rounds > 0
                    ? `${(d.current_round / d.total_rounds) * 100}%`
                    : '0%',
              }}
            />
          </div>
          <div className="flex gap-1 h-8 items-end mt-6">
            {Array.from({ length: 40 }).map((_, i) => (
              <div
                key={i}
                className="flex-1 rounded-sm transition-all duration-500 hover:scale-y-110 cursor-help bg-secondary/40"
                style={{ height: `${Math.random() * 60 + 40}%` }}
                title={`Cycle ${i + 1}`}
              />
            ))}
          </div>
          <p className="text-[10px] text-outline mt-4 uppercase tracking-widest font-bold">
            Past 40 Training Cycles
          </p>
        </Card>
      </div>
    </div>
  );
};
