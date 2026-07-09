import { useState, useMemo } from 'react';
import { RefreshCw, Activity, Server, Cpu, Layers, ArrowRightLeft, FlaskConical, BarChart3, Clock, Radio, Search, Trash2, AlertTriangle } from 'lucide-react';

import { useRealtime, useConnectionStatus, useEventHistory, useRealtimeDashboard } from '../../realtime/hooks';
import { Card } from '../ui/Card';
import { StatusBadge } from '../ui/StatusBadge';
import { ConnectionStatusIndicator, TransportLabel } from '../../realtime/status';
import { EVENT_CATEGORIES, EVENT_SEVERITIES, getCategoryLabel, filterEvents } from '../../realtime/events';
import type { RealtimeEventCategory, RealtimeEventSeverity, RealtimeEvent } from '../../realtime/types';

function formatTime(ts: number): string {
  try {
    return new Date(ts).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '—';
  }
}

const SEVERITY_STYLES: Record<RealtimeEventSeverity, string> = {
  info: 'border-l-primary/30',
  success: 'border-l-emerald-500/40',
  warning: 'border-l-amber-500/40',
  error: 'border-l-rose-500/40',
};

function SkeletonLive() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <Card key={i} className="p-4">
            <div className="h-3 w-16 rounded-full bg-surface-container-high animate-pulse mb-2" />
            <div className="h-6 w-20 rounded-lg bg-surface-container-high animate-pulse" />
          </Card>
        ))}
      </div>
    </div>
  );
}

export const LiveDashboard = () => {
  const { events, clearHistory, removeEvent, refetchAll } = useRealtime();
  const { connectionStatus } = useConnectionStatus();
  const { data, isLoading } = useRealtimeDashboard();

  const [showFilters, setShowFilters] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<RealtimeEventCategory | 'all'>('all');
  const [severityFilter, setSeverityFilter] = useState<RealtimeEventSeverity | 'all'>('all');

  const filteredEvents = useMemo(() => {
    const cats = categoryFilter === 'all' ? undefined : [categoryFilter];
    const sevs = severityFilter === 'all' ? undefined : [severityFilter];
    return filterEvents(events, { categories: cats, severities: sevs, search: searchQuery || undefined });
  }, [events, categoryFilter, severityFilter, searchQuery]);

  if (isLoading) return <SkeletonLive />;

  const commRate = data?.communicationRate ?? 0;
  const accuracyPct = data ? `${(data.globalAccuracy * 100).toFixed(1)}%` : '—';
  const lossVal = data?.globalLoss.toFixed(4) ?? '—';
  const clientsStr = data ? `${data.activeClients} / ${data.totalClients}` : '—';
  const roundStr = data ? `${data.currentRound} / ${data.totalRounds}` : '—';

  const isPolling = connectionStatus === 'polling';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">Live Monitor</h1>
            <ConnectionStatusIndicator size="md" />
          </div>
          <div className="flex items-center gap-3 mt-0.5">
            <TransportLabel />
            <span className="text-[10px] text-outline">·</span>
            <span className="text-[10px] font-mono text-outline">
              {events.length} events
            </span>
            {data?.lastUpdated && (
              <>
                <span className="text-[10px] text-outline">·</span>
                <span className="text-[10px] font-mono text-outline">
                  Updated {formatTime(new Date(data.lastUpdated).getTime())}
                </span>
              </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={refetchAll}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-container-high text-on-surface text-sm font-medium hover:bg-surface-container-highest transition-colors cursor-pointer"
          >
            <RefreshCw size={16} />
            Refresh
          </button>
          <button
            onClick={clearHistory}
            disabled={events.length === 0}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-container-high text-on-surface text-sm font-medium hover:bg-surface-container-highest transition-colors disabled:opacity-40 cursor-pointer"
          >
            <Trash2 size={16} />
            Clear
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <Activity size={14} className={isPolling ? 'text-sky-400' : 'text-primary'} />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-outline">Current Round</span>
          </div>
          <p className="text-lg font-display font-bold text-on-surface">{roundStr}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <FlaskConical size={14} className="text-secondary" />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-outline">Running Exps</span>
          </div>
          <p className="text-lg font-display font-bold text-on-surface">{data?.experimentsRunning ?? '—'}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <Server size={14} className="text-emerald-400" />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-outline">Active Clients</span>
          </div>
          <p className="text-lg font-display font-bold text-on-surface">{clientsStr}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <Radio size={14} className="text-amber-400" />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-outline">Comm Rate</span>
          </div>
          <p className="text-lg font-display font-bold text-on-surface">{commRate}/hr</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <Layers size={14} className="text-purple-400" />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-outline">Prototype Updates</span>
          </div>
          <p className="text-lg font-display font-bold text-on-surface">{data?.prototypeCount ?? 0}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <ArrowRightLeft size={14} className="text-cyan-400" />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-outline">KT Events</span>
          </div>
          <p className="text-lg font-display font-bold text-on-surface">{data?.knowledgeTransferCount ?? 0}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <BarChart3 size={14} className="text-primary" />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-outline">Accuracy</span>
          </div>
          <p className="text-lg font-display font-bold text-on-surface">{accuracyPct}</p>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <Cpu size={14} className="text-rose-400" />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-outline">Training Loss</span>
          </div>
          <p className="text-lg font-display font-bold text-on-surface">{lossVal}</p>
        </Card>
      </div>

      {/* Server Status + Event History grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Server Status */}
        <Card className="p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-on-surface">Server Status</h3>
            <ConnectionStatusIndicator size="sm" />
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs text-outline">Transport</span>
              <TransportLabel />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-outline">Training</span>
              <StatusBadge status={data?.trainingStatus ?? 'unknown'} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-outline">Uptime</span>
              <span className="text-sm font-mono text-on-surface">{data?.uptimeHours.toFixed(1) ?? '—'}h</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-outline">Rounds</span>
              <span className="text-sm font-mono text-on-surface">{roundStr}</span>
            </div>
            <div className="border-t border-outline-variant/50 pt-3 mt-3">
              <div className="flex items-center gap-2 text-[10px] text-outline uppercase tracking-wider font-bold mb-1">
                <Clock size={12} />
                Last Poll
              </div>
              <p className="text-xs font-mono text-on-surface">
                {data?.lastUpdated ? formatTime(new Date(data.lastUpdated).getTime()) : '—'}
              </p>
            </div>
          </div>
        </Card>

        {/* Event History */}
        <Card className="lg:col-span-2 p-5 flex flex-col min-h-[400px] max-h-[600px]">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-on-surface">Event History</h3>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="text-[10px] font-mono text-outline hover:text-on-surface transition-colors cursor-pointer"
            >
              {showFilters ? 'Hide filters' : 'Show filters'}
            </button>
          </div>

          {showFilters && (
            <div className="flex flex-wrap gap-2 mb-4 p-3 rounded-xl bg-surface-container-high">
              <div className="relative flex-1 min-w-[160px]">
                <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-outline" />
                <input
                  type="text"
                  placeholder="Search events..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-7 pr-2.5 py-1.5 rounded-lg bg-surface-container border border-outline-variant/50 text-xs text-on-surface focus:outline-none focus:border-primary/50"
                />
              </div>
              <select
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value as RealtimeEventCategory | 'all')}
                className="px-2.5 py-1.5 rounded-lg bg-surface-container border border-outline-variant/50 text-xs text-on-surface focus:outline-none focus:border-primary/50"
              >
                <option value="all">All Categories</option>
                {EVENT_CATEGORIES.map((cat) => (
                  <option key={cat} value={cat}>{getCategoryLabel(cat)}</option>
                ))}
              </select>
              <select
                value={severityFilter}
                onChange={(e) => setSeverityFilter(e.target.value as RealtimeEventSeverity | 'all')}
                className="px-2.5 py-1.5 rounded-lg bg-surface-container border border-outline-variant/50 text-xs text-on-surface focus:outline-none focus:border-primary/50"
              >
                <option value="all">All Severities</option>
                {EVENT_SEVERITIES.map((sev) => (
                  <option key={sev} value={sev}>{sev.charAt(0).toUpperCase() + sev.slice(1)}</option>
                ))}
              </select>
            </div>
          )}

          <div className="flex-1 overflow-y-auto space-y-1 custom-scrollbar">
            {filteredEvents.length === 0 && (
              <div className="flex flex-col items-center justify-center h-48 gap-2">
                <AlertTriangle size={20} className="text-outline/40" />
                <p className="text-xs text-outline">No events recorded</p>
              </div>
            )}
            {filteredEvents.map((event) => (
              <div
                key={event.id}
                className={`flex items-start gap-3 p-2.5 rounded-lg hover:bg-surface-container-high group border-l-2 ${SEVERITY_STYLES[event.severity]} transition-colors`}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-on-surface truncate">{event.title}</span>
                    <StatusBadge status={event.category === 'server' ? 'active' : event.severity} className="shrink-0" />
                  </div>
                  <p className="text-[11px] text-outline mt-0.5 line-clamp-2">{event.description}</p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-[10px] font-mono text-outline">{formatTime(event.timestamp)}</span>
                  <button
                    onClick={() => removeEvent(event.id)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity text-outline hover:text-rose-400 cursor-pointer"
                    title="Remove event"
                  >
                    ×
                  </button>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
};
