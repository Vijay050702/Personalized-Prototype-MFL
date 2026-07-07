import { useState, useEffect, useRef, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import {
  RefreshCw, AlertTriangle, Activity, Target, Zap, Play, Pause,
  Square, Save, Settings, RotateCcw, Clock, BarChart3, Server,
  Brain, Users, CheckCircle, XCircle, FileJson, ChevronDown,
  ChevronUp, X,
} from 'lucide-react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from 'recharts';

import {
  fetchTrainingStatus, fetchTrainingConfig,
  startTraining, pauseTraining, resumeTraining,
  stopTraining, saveCheckpoint, updateTrainingConfig,
} from '../api/training';
import { Card } from '../components/ui/Card';
import { StatCard } from '../components/ui/StatCard';
import { StatusBadge } from '../components/ui/StatusBadge';
import type {
  TrainingStatusData, TrainingConfigData,
} from '../types';

function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}

function formatDuration(seconds: number): string {
  if (!seconds || seconds <= 0) return '—';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
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
    if (error.response?.status === 409) return 'Training is already in progress.';
    if (error.response?.status === 422) return 'Invalid configuration. Please check your inputs.';
    if (error.response?.status === 404) return 'Training endpoint not found. The backend may be incomplete.';
    if (error.code === 'ECONNABORTED') return 'Request timed out. Please try again.';
    if (!error.response) return 'Backend is unavailable. Please check your connection.';
  }
  if (error instanceof Error) return error.message;
  return 'An unexpected error occurred.';
}

function getTrainingPhase(status: string | undefined): string {
  if (!status) return 'idle';
  const s = status.toLowerCase();
  if (['running', 'paused', 'completed', 'failed', 'idle', 'initialized'].includes(s)) return s;
  if (s === 'resumed') return 'running';
  return 'idle';
}

function isRunning(status: string | undefined): boolean {
  return getTrainingPhase(status) === 'running';
}

function isPaused(status: string | undefined): boolean {
  return getTrainingPhase(status) === 'paused';
}

function isIdle(status: string | undefined): boolean {
  const phase = getTrainingPhase(status);
  return phase === 'idle' || phase === 'initialized' || phase === 'completed' || phase === 'failed';
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
      <h2 className="text-lg font-display font-bold text-on-surface">Failed to load training data</h2>
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
      <h2 className="text-lg font-display font-bold text-on-surface">No training data available</h2>
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

function Notification({ message, type, onClose }: { message: string; type: 'success' | 'error'; onClose: () => void }) {
  useEffect(() => {
    const timer = setTimeout(onClose, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className={cn(
      'fixed bottom-6 right-6 z-50 flex items-center gap-3 px-5 py-3 rounded-xl shadow-2xl text-sm font-medium transition-all',
      type === 'success' ? 'bg-emerald-500/90 text-white' : 'bg-rose-500/90 text-white',
    )}>
      {type === 'success' ? <CheckCircle size={18} /> : <XCircle size={18} />}
      <span>{message}</span>
      <button onClick={onClose} className="ml-2 hover:opacity-70 cursor-pointer">
        <X size={16} />
      </button>
    </div>
  );
}

const OPTIMIZER_OPTIONS = ['adam', 'sgd', 'adamw', 'fedprox'] as const;
const SCHEDULER_OPTIONS = ['step_lr', 'cosine_annealing', 'reduce_on_plateau', 'none'] as const;
const STRATEGY_OPTIONS = ['FedAvg', 'FedProx', 'FedProto'] as const;

export const Training = () => {
  const queryClient = useQueryClient();
  const [notification, setNotification] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const [showConfig, setShowConfig] = useState(false);
  const [configForm, setConfigForm] = useState<TrainingConfigData | null>(null);
  const lastRoundRef = useRef<number>(-1);

  const [convergenceData, setConvergenceData] = useState<Array<{ round: number; loss: number; accuracy: number }>>([]);

  const {
    data: statusSummary,
    isLoading: statusLoading,
    isError: statusError,
    error: statusErr,
    refetch: refetchStatus,
    isRefetching: statusRefetching,
    dataUpdatedAt,
  } = useQuery({
    queryKey: ['training-status'],
    queryFn: fetchTrainingStatus,
    refetchInterval: (query) => query.state.data?.data?.status === 'running' ? 2000 : false,
    retry: 2,
    staleTime: 4000,
  });

  const status: TrainingStatusData | null = statusSummary?.data ?? null;
  const phase = getTrainingPhase(status?.status);

  useEffect(() => {
    if (!status) return;
    const phase = getTrainingPhase(status.status);
    const isIdleZero = (phase === 'idle' || phase === 'initialized') && status.current_round === 0;
    if (isIdleZero) return;
    if (convergenceData.length === 0 || status.current_round !== lastRoundRef.current) {
      lastRoundRef.current = status.current_round;
      setConvergenceData(prev => {
        if (prev.length > 0 && prev[prev.length - 1].round === status.current_round) return prev;
        return [...prev, { round: status.current_round, loss: status.current_loss, accuracy: status.current_accuracy }];
      });
    }
  }, [status, convergenceData.length]);

  const {
    data: configSummary,
    isLoading: configLoading,
    error: configErr,
    refetch: refetchConfig,
  } = useQuery({
    queryKey: ['training-config'],
    queryFn: fetchTrainingConfig,
    retry: false,
    staleTime: 30000,
    enabled: showConfig,
  });

  useEffect(() => {
    if (configSummary?.data && !configForm) {
      setConfigForm(configSummary.data);
    }
  }, [configSummary, configForm]);

  const showNotification = useCallback((message: string, type: 'success' | 'error') => {
    setNotification({ message, type });
  }, []);

  const dismissNotification = useCallback(() => {
    setNotification(null);
  }, []);

  const onMutError = useCallback((err: Error) => {
    showNotification(getErrorMessage(err), 'error');
  }, [showNotification]);

  const onMutSuccess = useCallback((message: string) => {
    queryClient.invalidateQueries({ queryKey: ['training-status'] });
    showNotification(message, 'success');
  }, [queryClient, showNotification]);

  const startMut = useMutation({
    mutationFn: startTraining,
    onSuccess: () => onMutSuccess('Training started successfully'),
    onError: onMutError,
  });

  const pauseMut = useMutation({
    mutationFn: pauseTraining,
    onSuccess: () => onMutSuccess('Training paused'),
    onError: onMutError,
  });

  const resumeMut = useMutation({
    mutationFn: resumeTraining,
    onSuccess: () => onMutSuccess('Training resumed'),
    onError: onMutError,
  });

  const stopMut = useMutation({
    mutationFn: stopTraining,
    onSuccess: () => onMutSuccess('Training stopped'),
    onError: onMutError,
  });

  const checkpointMut = useMutation({
    mutationFn: saveCheckpoint,
    onSuccess: () => onMutSuccess('Checkpoint saved'),
    onError: onMutError,
  });

  const saveConfigMut = useMutation({
    mutationFn: updateTrainingConfig,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-config'] });
      queryClient.invalidateQueries({ queryKey: ['training-status'] });
      showNotification('Configuration saved', 'success');
    },
    onError: onMutError,
  });

  const isOperating = startMut.isPending || pauseMut.isPending || resumeMut.isPending || stopMut.isPending || checkpointMut.isPending || saveConfigMut.isPending;

  const handleReloadConfig = useCallback(() => {
    refetchConfig();
    showNotification('Configuration reloaded', 'success');
  }, [refetchConfig, showNotification]);

  const handleResetConfig = useCallback(() => {
    if (configSummary?.data) {
      setConfigForm({ ...configSummary.data });
      showNotification('Configuration reset to saved values', 'success');
    }
  }, [configSummary, showNotification]);

  const handleConfigFieldChange = useCallback(<K extends keyof TrainingConfigData>(field: K, value: TrainingConfigData[K]) => {
    setConfigForm(prev => prev ? { ...prev, [field]: value } : null);
  }, []);

  const handleSaveConfig = useCallback(() => {
    if (!configForm) return;
    if (configForm.learning_rate <= 0) {
      showNotification('Learning rate must be greater than 0', 'error');
      return;
    }
    if (configForm.client_count < 1) {
      showNotification('Client count must be at least 1', 'error');
      return;
    }
    if (configForm.communication_rounds < 1) {
      showNotification('Communication rounds must be at least 1', 'error');
      return;
    }
    if (configForm.local_epochs < 1) {
      showNotification('Local epochs must be at least 1', 'error');
      return;
    }
    if (configForm.batch_size < 1) {
      showNotification('Batch size must be at least 1', 'error');
      return;
    }
    if (!configForm.dataset.trim()) {
      showNotification('Dataset name is required', 'error');
      return;
    }
    saveConfigMut.mutate(configForm);
  }, [configForm, saveConfigMut, showNotification]);

  if (statusLoading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-72 rounded-lg bg-surface-container-high animate-pulse" />
        <div className="flex gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-10 w-28 rounded-lg bg-surface-container-high animate-pulse" />
          ))}
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2 p-6 h-[450px] flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          </Card>
          <Card className="p-6 h-[450px] flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          </Card>
        </div>
      </div>
    );
  }

  if (statusError) {
    return (
      <ErrorState
        message={statusErr instanceof Error ? statusErr.message : 'An unexpected error occurred.'}
        onRetry={() => refetchStatus()}
      />
    );
  }

  if (!status) {
    return (
      <EmptyState
        message="The server returned an empty response."
        onRetry={() => refetchStatus()}
      />
    );
  }

  const accuracyPct = `${(status.current_accuracy * 100).toFixed(2)}%`;
  const roundStr = `${status.current_round} / ${status.total_rounds}`;
  const epochStr = `${status.epochs_completed} / ${status.total_epochs}`;
  const roundPct = status.total_rounds > 0 ? (status.current_round / status.total_rounds) * 100 : 0;
  const epochPct = status.total_epochs > 0 ? (status.epochs_completed / status.total_epochs) * 100 : 0;

  const configUnavailable = configErr && !configLoading && !configSummary;
  const configErrorMsg = configErr ? getErrorMessage(configErr) : null;

  return (
    <div className="space-y-6">
      {notification && (
        <Notification message={notification.message} type={notification.type} onClose={dismissNotification} />
      )}

      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">Model Optimization Engine</h1>
            <StatusBadge status={status.status} />
          </div>
          <p className="text-sm text-outline mt-0.5">
            {dataUpdatedAt
              ? `Last updated ${formatTimestamp(new Date(dataUpdatedAt).toISOString())}`
              : 'Fetching live data...'}
            {phase === 'running' && (
              <span className="ml-2 text-emerald-400 text-xs">
                ● Live (2s auto-refresh)
              </span>
            )}
          </p>
        </div>
        <button
          onClick={() => refetchStatus()}
          disabled={statusRefetching || isOperating}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-container-high text-on-surface text-sm font-medium hover:bg-surface-container-highest transition-colors disabled:opacity-50 cursor-pointer"
        >
          <RefreshCw size={16} className={statusRefetching ? 'animate-spin' : ''} />
          {statusRefetching ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <Card className="p-4">
        <div className="flex flex-wrap gap-3 items-center">
          <button
            onClick={() => startMut.mutate()}
            disabled={!isIdle(status?.status) || isOperating}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            <Play size={16} fill="currentColor" />
            Start
          </button>
          <button
            onClick={() => pauseMut.mutate()}
            disabled={!isRunning(status?.status) || isOperating}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-amber-500/20 text-amber-400 text-sm font-semibold hover:bg-amber-500/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            <Pause size={16} />
            Pause
          </button>
          <button
            onClick={() => resumeMut.mutate()}
            disabled={!isPaused(status?.status) || isOperating}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-emerald-500/20 text-emerald-400 text-sm font-semibold hover:bg-emerald-500/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            <Play size={16} fill="currentColor" />
            Resume
          </button>
          <button
            onClick={() => stopMut.mutate()}
            disabled={(!isRunning(status?.status) && !isPaused(status?.status)) || isOperating}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rose-500/20 text-rose-400 text-sm font-semibold hover:bg-rose-500/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            <Square size={16} />
            Stop
          </button>
          <div className="w-px h-8 bg-outline-variant/50" />
          <button
            onClick={() => checkpointMut.mutate()}
            disabled={!isRunning(status?.status) || isOperating}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-surface-container-high text-on-surface text-sm font-medium hover:bg-surface-container-highest transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            <Save size={16} />
            Checkpoint
          </button>
          <button
            onClick={handleReloadConfig}
            disabled={isOperating}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-surface-container-high text-on-surface text-sm font-medium hover:bg-surface-container-highest transition-colors disabled:opacity-40 cursor-pointer"
          >
            <RotateCcw size={16} />
            Reload Config
          </button>
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-surface-container-high text-on-surface text-sm font-medium hover:bg-surface-container-highest transition-colors cursor-pointer"
          >
            <Settings size={16} />
            {showConfig ? 'Hide Config' : 'Config'}
            {showConfig ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>
      </Card>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatCard label="Training Status" value={phase.charAt(0).toUpperCase() + phase.slice(1)} icon={<Activity size={18} />} />
        <StatCard label="Communication Round" value={roundStr} icon={<RefreshCw size={18} />} />
        <StatCard label="Local Epoch" value={epochStr} icon={<Target size={18} />} />
        <StatCard label="Training Loss" value={status.current_loss.toFixed(4)} icon={<BarChart3 size={18} />} />
        <StatCard label="Accuracy" value={accuracyPct} icon={<BarChart3 size={18} />} />
        <StatCard label="Learning Rate" value={status.learning_rate.toExponential(2)} icon={<Brain size={18} />} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2 p-6 h-[450px] flex flex-col">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                <Activity size={20} />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-on-surface">Convergence Analysis</h3>
                <p className="text-xs text-outline">
                  {convergenceData.length > 1
                    ? `Accuracy vs. Loss over ${convergenceData.length} rounds`
                    : 'Collecting data as rounds progress...'}
                </p>
              </div>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            {convergenceData.length < 2 ? (
              <div className="flex items-center justify-center h-full text-outline text-sm gap-2">
                <div className="w-5 h-5 border-2 border-outline border-t-transparent rounded-full animate-spin" />
                Waiting for round progression...
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={convergenceData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    dataKey="round"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
                    label={{ value: 'Global Rounds', position: 'insideBottom', offset: -5, fill: 'var(--color-outline)', fontSize: 10 }}
                  />
                  <YAxis
                    yAxisId="left"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
                    label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', fill: 'var(--color-outline)', fontSize: 10 }}
                    domain={[0.7, 1]}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
                    label={{ value: 'Loss Index', angle: 90, position: 'insideRight', fill: 'var(--color-outline)', fontSize: 10 }}
                    domain={[0, 'auto']}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'var(--color-surface-container-highest)',
                      borderColor: 'var(--color-outline-variant)',
                      borderRadius: '12px',
                      fontSize: '12px',
                    }}
                    formatter={(value: number, name: string) => [
                      name === 'accuracy' ? `${(value * 100).toFixed(2)}%` : value.toFixed(4),
                      name === 'accuracy' ? 'Accuracy' : 'Loss',
                    ]}
                  />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: '10px', paddingTop: '20px' }} />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="accuracy"
                    stroke="var(--color-primary)"
                    strokeWidth={3}
                    dot={{ r: 4, fill: 'var(--color-primary)', strokeWidth: 2, stroke: 'var(--color-surface)' }}
                    activeDot={{ r: 6 }}
                    name="accuracy"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="loss"
                    stroke="var(--color-secondary)"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>

        <div className="space-y-4">
          <Card className="p-6 bg-surface-container-lowest">
            <div className="flex items-center justify-between mb-4">
              <span className="text-[10px] uppercase font-bold tracking-[0.2em] text-outline">Current Status</span>
              <StatusBadge status={status.status} />
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">Aggregation</span>
                <span className="text-sm font-mono font-bold text-on-surface">{status.aggregation_algorithm}</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">Participating Clients</span>
                <span className="text-sm font-mono font-bold text-on-surface">{status.clients_participating}</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">Elapsed Time</span>
                <span className="text-sm font-mono font-bold text-on-surface">{formatDuration(status.time_elapsed_seconds)}</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                <span className="text-xs text-outline">Est. Remaining</span>
                <span className="text-sm font-mono font-bold text-on-surface">{formatDuration(status.estimated_time_remaining)}</span>
              </div>
              <div className="flex items-center justify-between py-2">
                <span className="text-xs text-outline">Current Round</span>
                <span className="text-sm font-mono font-bold text-on-surface">{roundStr}</span>
              </div>
            </div>
          </Card>

          {isRunning(status?.status) && (
            <Card className="p-5 bg-primary/5 border-primary/20">
              <div className="flex items-center gap-3 mb-3">
                <Zap size={18} className="text-primary" />
                <h5 className="text-sm font-bold text-on-surface">Training Active</h5>
              </div>
              <p className="text-[11px] text-on-surface-variant leading-relaxed">
                Training is currently running. Auto-refreshing every 2 seconds. Metrics update in real-time as rounds complete.
              </p>
              <div className="mt-4 flex items-center justify-between">
                <span className="text-[10px] font-bold text-primary">Round {status.current_round} of {status.total_rounds}</span>
                <div className="flex gap-0.5">
                  {Array.from({ length: 12 }).map((_, i) => (
                    <div key={i} className="w-1 h-3 bg-primary/20 rounded-full animate-bounce" style={{ animationDelay: `${i * 0.1}s` }} />
                  ))}
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>

      {showConfig && (
        <Card className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                <Settings size={20} />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-on-surface">Training Configuration</h3>
                <p className="text-xs text-outline">
                  {configLoading
                    ? 'Loading configuration from server...'
                    : configUnavailable
                      ? 'Configuration endpoint unavailable'
                      : 'Edit and save training parameters'}
                </p>
              </div>
            </div>
            {configForm && !configUnavailable && (
              <div className="flex gap-2">
                <button
                  onClick={handleResetConfig}
                  disabled={isOperating}
                  className="px-4 py-2 rounded-xl border border-outline-variant text-sm font-medium text-outline hover:text-on-surface hover:bg-surface-container-high transition-colors disabled:opacity-40 cursor-pointer"
                >
                  Reset
                </button>
                <button
                  onClick={handleSaveConfig}
                  disabled={isOperating || saveConfigMut.isPending}
                  className="flex items-center gap-2 px-5 py-2 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-40 cursor-pointer"
                >
                  {saveConfigMut.isPending ? (
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Save size={16} />
                  )}
                  Save Configuration
                </button>
              </div>
            )}
          </div>

          {configLoading && (
            <div className="space-y-4">
              {Array.from({ length: 6 }).map((_, i) => <SkeletonLine key={i} />)}
            </div>
          )}

          {configUnavailable && (
            <div className="flex flex-col items-center justify-center py-8 gap-3">
              <FileJson size={32} className="text-outline/50" />
              <p className="text-sm text-outline">Configuration endpoint is not available.</p>
              <p className="text-xs text-outline/70 max-w-md text-center">
                {configErrorMsg}
              </p>
              <button
                onClick={handleReloadConfig}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-container-high text-sm font-medium text-on-surface hover:bg-surface-container-highest transition-colors cursor-pointer"
              >
                <RotateCcw size={14} />
                Retry Load
              </button>
            </div>
          )}

          {configForm && !configUnavailable && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-outline uppercase tracking-wider">Dataset</label>
                <input
                  type="text"
                  value={configForm.dataset}
                  onChange={(e) => handleConfigFieldChange('dataset', e.target.value)}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-surface-container-high border border-outline-variant/50 text-sm text-on-surface placeholder-outline focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30 transition-all"
                  placeholder="dataset_name"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-outline uppercase tracking-wider">Client Count</label>
                <input
                  type="number"
                  value={configForm.client_count}
                  onChange={(e) => handleConfigFieldChange('client_count', Math.max(1, parseInt(e.target.value) || 1))}
                  min={1}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-surface-container-high border border-outline-variant/50 text-sm text-on-surface focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30 transition-all"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-outline uppercase tracking-wider">Communication Rounds</label>
                <input
                  type="number"
                  value={configForm.communication_rounds}
                  onChange={(e) => handleConfigFieldChange('communication_rounds', Math.max(1, parseInt(e.target.value) || 1))}
                  min={1}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-surface-container-high border border-outline-variant/50 text-sm text-on-surface focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30 transition-all"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-outline uppercase tracking-wider">Local Epochs</label>
                <input
                  type="number"
                  value={configForm.local_epochs}
                  onChange={(e) => handleConfigFieldChange('local_epochs', Math.max(1, parseInt(e.target.value) || 1))}
                  min={1}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-surface-container-high border border-outline-variant/50 text-sm text-on-surface focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30 transition-all"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-outline uppercase tracking-wider">Batch Size</label>
                <input
                  type="number"
                  value={configForm.batch_size}
                  onChange={(e) => handleConfigFieldChange('batch_size', Math.max(1, parseInt(e.target.value) || 1))}
                  min={1}
                  step={1}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-surface-container-high border border-outline-variant/50 text-sm text-on-surface focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30 transition-all"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-outline uppercase tracking-wider">Learning Rate</label>
                <input
                  type="number"
                  value={configForm.learning_rate}
                  onChange={(e) => handleConfigFieldChange('learning_rate', isNaN(parseFloat(e.target.value)) ? 0.001 : parseFloat(e.target.value))}
                  min={0.0001}
                  step={0.0001}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-surface-container-high border border-outline-variant/50 text-sm text-on-surface focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30 transition-all"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-outline uppercase tracking-wider">Optimizer</label>
                <select
                  value={configForm.optimizer}
                  onChange={(e) => handleConfigFieldChange('optimizer', e.target.value)}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-surface-container-high border border-outline-variant/50 text-sm text-on-surface focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30 transition-all"
                >
                  {OPTIMIZER_OPTIONS.map((opt) => (
                    <option key={opt} value={opt}>{opt.toUpperCase()}</option>
                  ))}
                </select>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-outline uppercase tracking-wider">Scheduler</label>
                <select
                  value={configForm.scheduler}
                  onChange={(e) => handleConfigFieldChange('scheduler', e.target.value)}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-surface-container-high border border-outline-variant/50 text-sm text-on-surface focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30 transition-all"
                >
                  {SCHEDULER_OPTIONS.map((sch) => (
                    <option key={sch} value={sch}>{sch.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}</option>
                  ))}
                </select>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-outline uppercase tracking-wider">Aggregation Strategy</label>
                <select
                  value={configForm.aggregation_strategy}
                  onChange={(e) => handleConfigFieldChange('aggregation_strategy', e.target.value)}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-surface-container-high border border-outline-variant/50 text-sm text-on-surface focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30 transition-all"
                >
                  {STRATEGY_OPTIONS.map((str) => (
                    <option key={str} value={str}>{str}</option>
                  ))}
                </select>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-outline uppercase tracking-wider">Knowledge Transfer</label>
                <div className="flex items-center gap-3 h-10">
                  <button
                    onClick={() => handleConfigFieldChange('knowledge_transfer_enabled', !configForm.knowledge_transfer_enabled)}
                    className={cn(
                      'relative w-11 h-6 rounded-full transition-colors',
                      configForm.knowledge_transfer_enabled ? 'bg-primary' : 'bg-surface-container-high',
                    )}
                  >
                    <div className={cn(
                      'absolute top-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform',
                      configForm.knowledge_transfer_enabled ? 'translate-x-[22px]' : 'translate-x-0.5',
                    )} />
                  </button>
                  <span className="text-sm text-on-surface-variant">
                    {configForm.knowledge_transfer_enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-outline uppercase tracking-wider">Personalization</label>
                <div className="flex items-center gap-3 h-10">
                  <button
                    onClick={() => handleConfigFieldChange('personalization_enabled', !configForm.personalization_enabled)}
                    className={cn(
                      'relative w-11 h-6 rounded-full transition-colors',
                      configForm.personalization_enabled ? 'bg-primary' : 'bg-surface-container-high',
                    )}
                  >
                    <div className={cn(
                      'absolute top-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform',
                      configForm.personalization_enabled ? 'translate-x-[22px]' : 'translate-x-0.5',
                    )} />
                  </button>
                  <span className="text-sm text-on-surface-variant">
                    {configForm.personalization_enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </Card>
      )}

      <Card className="p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
            <Activity size={20} />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-on-surface">Live Progress</h3>
            <p className="text-xs text-outline">Real-time training metrics and round completion</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-outline font-medium">Round Progress</span>
                <span className="text-xs font-mono text-on-surface font-bold">{roundPct.toFixed(1)}%</span>
              </div>
              <div className="w-full h-2.5 rounded-full bg-surface-container-high overflow-hidden">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-700"
                  style={{ width: `${roundPct}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-outline font-medium">Epoch Progress</span>
                <span className="text-xs font-mono text-on-surface font-bold">{epochPct.toFixed(1)}%</span>
              </div>
              <div className="w-full h-2.5 rounded-full bg-surface-container-high overflow-hidden">
                <div
                  className="h-full rounded-full bg-secondary transition-all duration-700"
                  style={{ width: `${epochPct}%` }}
                />
              </div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-surface-container-high/50 rounded-xl p-4">
              <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Training Loss</span>
              <p className="text-lg font-mono font-bold text-on-surface mt-1">{status.current_loss.toFixed(4)}</p>
            </div>
            <div className="bg-surface-container-high/50 rounded-xl p-4">
              <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Accuracy</span>
              <p className="text-lg font-mono font-bold text-on-surface mt-1">{accuracyPct}</p>
            </div>
            <div className="bg-surface-container-high/50 rounded-xl p-4">
              <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Clients</span>
              <p className="text-lg font-mono font-bold text-on-surface mt-1">{status.clients_participating}</p>
            </div>
            <div className="bg-surface-container-high/50 rounded-xl p-4">
              <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Aggregation</span>
              <p className="text-base font-mono font-bold text-on-surface mt-1 truncate" title={status.aggregation_algorithm}>{status.aggregation_algorithm}</p>
            </div>
          </div>
        </div>

        <div className="border-t border-outline-variant/50 pt-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Aggregated Prototypes</span>
              <p className="text-lg font-mono font-bold text-on-surface mt-1">—</p>
              <p className="text-[10px] text-outline">Not available</p>
            </div>
            <div>
              <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Knowledge Transfer</span>
              <p className="text-lg font-mono font-bold text-on-surface mt-1">
                {isRunning(status?.status) ? 'Active' : '—'}
              </p>
              <p className="text-[10px] text-outline">
                {isRunning(status?.status) ? 'Cross-modal active' : 'Waiting'}
              </p>
            </div>
            <div>
              <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Personalization</span>
              <p className="text-lg font-mono font-bold text-on-surface mt-1">
                {isRunning(status?.status) ? 'Active' : '—'}
              </p>
              <p className="text-[10px] text-outline">
                {isRunning(status?.status) ? 'Client-adaptive' : 'Waiting'}
              </p>
            </div>
            <div>
              <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Checkpoint</span>
              <p className="text-lg font-mono font-bold text-on-surface mt-1">
                {isRunning(status?.status) ? 'Saving...' : '—'}
              </p>
              <p className="text-[10px] text-outline">Per round interval</p>
            </div>
          </div>
        </div>
      </Card>

      <Card className="overflow-hidden">
        <div className="px-6 py-4 border-b border-outline-variant flex items-center justify-between bg-surface-container-low/50">
          <h3 className="text-sm font-bold text-on-surface">Round Execution History</h3>
        </div>
        {convergenceData.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 gap-3">
            <Activity size={28} className="text-outline/40" />
            <p className="text-sm text-outline">No round history yet. Start or resume training to populate history.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-surface-container-low/30 text-[10px] uppercase tracking-[0.15em] font-bold text-outline">
                  <th className="px-6 py-4 font-bold border-b border-outline-variant">Round #</th>
                  <th className="px-6 py-4 font-bold border-b border-outline-variant">Status</th>
                  <th className="px-6 py-4 font-bold border-b border-outline-variant">Loss</th>
                  <th className="px-6 py-4 font-bold border-b border-outline-variant text-right">Accuracy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/30 text-xs">
                {[...convergenceData].reverse().map((point) => (
                  <tr key={point.round} className="hover:bg-surface-container-high/20 transition-colors">
                    <td className="px-6 py-4 font-mono font-bold text-on-surface">R-{point.round}</td>
                    <td className="px-6 py-4">
                      <StatusBadge
                        status={
                          point.round === status.current_round && isRunning(status?.status)
                            ? 'running'
                            : 'completed'
                        }
                      />
                    </td>
                    <td className="px-6 py-4 text-outline font-mono">{point.loss.toFixed(4)}</td>
                    <td className="px-6 py-4 text-right font-mono font-bold text-primary">
                      {(point.accuracy * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
};
