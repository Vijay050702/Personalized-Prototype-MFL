import { useState, useMemo, useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Search,
  RefreshCw,
  SlidersHorizontal,
  X,
  Eye,
  Cpu,
  MapPin,
  HardDrive,
  Activity,
  Clock,
  Calendar,
  Zap,
  AlertTriangle,
  Smartphone,
  Server,
  Monitor,
} from 'lucide-react';

import { fetchClients } from '../api/clients';
import { Card } from '../components/ui/Card';
import { StatusBadge } from '../components/ui/StatusBadge';
import type { ClientResponse } from '../types';

const PAGE_SIZES = [5, 10, 20, 50];
const AUTO_REFRESH_MS = 5000;

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}

function formatBytes(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(1)} KB`;
  return `${bytes} B`;
}

function formatTimestamp(iso: string): string {
  const date = new Date(iso);
  const now = Date.now();
  const diffMs = now - date.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay}d ago`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

function getDeviceIcon(device: string) {
  const d = device.toLowerCase();
  if (d.includes('phone') || d.includes('pixel') || d.includes('iphone')) return Smartphone;
  if (d.includes('jetson') || d.includes('gpu') || d.includes('rtx') || d.includes('a100')) return Monitor;
  if (d.includes('raspberry') || d.includes('iot') || d.includes('sensor')) return Cpu;
  return Server;
}

/* ------------------------------------------------------------------ */
/*  Skeleton                                                           */
/* ------------------------------------------------------------------ */

function SkeletonRow() {
  return (
    <tr className="animate-pulse border-b border-outline-variant/30">
      {Array.from({ length: 8 }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <div className="h-4 rounded bg-surface-container-high" style={{ width: `${60 + Math.random() * 30}%` }} />
        </td>
      ))}
    </tr>
  );
}

/* ------------------------------------------------------------------ */
/*  Empty state                                                        */
/* ------------------------------------------------------------------ */

function EmptyState({ onRefresh }: { onRefresh: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <div className="w-16 h-16 rounded-2xl bg-surface-container-high flex items-center justify-center">
        <Cpu size={32} className="text-outline" />
      </div>
      <h3 className="text-lg font-display font-bold text-on-surface">No clients found</h3>
      <p className="text-sm text-outline max-w-sm text-center">
        No clients match your search criteria. Try adjusting your filters.
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

/* ------------------------------------------------------------------ */
/*  Error state                                                        */
/* ------------------------------------------------------------------ */

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <div className="w-14 h-14 rounded-2xl bg-rose-500/10 flex items-center justify-center">
        <AlertTriangle size={28} className="text-rose-400" />
      </div>
      <h3 className="text-lg font-display font-bold text-on-surface">Failed to load clients</h3>
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

/* ------------------------------------------------------------------ */
/*  Detail Panel                                                       */
/* ------------------------------------------------------------------ */

function DetailPanel({
  client,
  onClose,
}: {
  client: ClientResponse;
  onClose: () => void;
}) {
  const DeviceIcon = getDeviceIcon(client.device);

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/40 backdrop-blur-sm">
      <Card className="w-full max-w-lg h-full overflow-y-auto rounded-none rounded-l-2xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center">
              <DeviceIcon size={20} className="text-primary" />
            </div>
            <div>
              <h2 className="text-lg font-display font-bold text-on-surface">{client.name}</h2>
              <p className="text-[10px] font-mono text-outline uppercase">{client.id}</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-surface-container-high transition-colors cursor-pointer">
            <X size={20} className="text-outline" />
          </button>
        </div>

        <div className="mb-4">
          <StatusBadge status={client.status} />
        </div>

        <div className="space-y-4 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Client Profile</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Device</p>
              <p className="text-sm font-semibold text-on-surface">{client.device}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Region</p>
              <p className="text-sm font-semibold text-on-surface">{client.region}</p>
            </div>
          </div>
        </div>

        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-bold text-on-surface">Training Metrics</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Accuracy</p>
              <p className="text-lg font-display font-bold text-primary">{(client.accuracy * 100).toFixed(1)}%</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Loss</p>
              <p className="text-lg font-display font-bold text-on-surface">{client.loss.toFixed(4)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Last Round</p>
              <p className="text-lg font-display font-bold text-on-surface">#{client.last_round}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Data Size</p>
              <p className="text-lg font-display font-bold text-on-surface">{formatBytes(client.data_size)}</p>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-bold text-on-surface">Participation</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Joined</p>
              <p className="text-sm font-semibold text-on-surface">{formatDate(client.joined_at)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Last Seen</p>
              <p className="text-sm font-semibold text-on-surface">{formatTimestamp(client.last_communication)}</p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Clients Page                                                  */
/* ------------------------------------------------------------------ */

export const Clients = () => {
  /* View state */
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [sortField, setSortField] = useState<string>('name');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(10);
  const [selectedClient, setSelectedClient] = useState<ClientResponse | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  /* Query */
  const {
    data: listResponse,
    isLoading,
    isError,
    error,
    refetch,
    isRefetching,
    dataUpdatedAt,
  } = useQuery({
    queryKey: ['clients'],
    queryFn: fetchClients,
    refetchInterval: AUTO_REFRESH_MS,
    retry: 2,
    staleTime: 3000,
  });

  /* Auto-refresh indicator reset when data updates */
  const [lastRefresh, setLastRefresh] = useState<string>('');
  useEffect(() => {
    if (dataUpdatedAt) {
      setLastRefresh(new Date(dataUpdatedAt).toLocaleTimeString());
    }
  }, [dataUpdatedAt]);

  /* Derived data */
  const allClients: ClientResponse[] = listResponse?.data ?? [];

  const statuses = useMemo(() => {
    const set = new Set(allClients.map((c) => c.status));
    return Array.from(set).sort();
  }, [allClients]);

  const filtered = useMemo(() => {
    let list = allClients;

    if (search) {
      const q = search.toLowerCase();
      list = list.filter(
        (c) =>
          c.name.toLowerCase().includes(q) ||
          c.id.toLowerCase().includes(q) ||
          c.device.toLowerCase().includes(q) ||
          c.region.toLowerCase().includes(q),
      );
    }

    if (statusFilter) {
      list = list.filter((c) => c.status === statusFilter);
    }

    list.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case 'name':
          cmp = a.name.localeCompare(b.name);
          break;
        case 'id':
          cmp = a.id.localeCompare(b.id);
          break;
        case 'status':
          cmp = a.status.localeCompare(b.status);
          break;
        case 'device':
          cmp = a.device.localeCompare(b.device);
          break;
        case 'region':
          cmp = a.region.localeCompare(b.region);
          break;
        case 'accuracy':
          cmp = a.accuracy - b.accuracy;
          break;
        case 'loss':
          cmp = a.loss - b.loss;
          break;
        case 'last_round':
          cmp = a.last_round - b.last_round;
          break;
        case 'data_size':
          cmp = a.data_size - b.data_size;
          break;
        default:
          cmp = 0;
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return list;
  }, [allClients, search, statusFilter, sortField, sortDir]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(page, totalPages - 1);
  const paged = filtered.slice(safePage * pageSize, (safePage + 1) * pageSize);

  /* Handlers */
  const handleSort = (field: string) => {
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

  function SortIcon({ field }: { field: string }) {
    if (sortField !== field) return <span className="text-outline/30 ml-1">&#8597;</span>;
    return <span className="text-primary ml-1">{sortDir === 'asc' ? '&#8593;' : '&#8595;'}</span>;
  }

  /* ---- render ---- */
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">Client Manager</h1>
          <p className="text-sm text-outline">Monitor and manage federated learning clients</p>
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
            placeholder="Search by name, ID, device, or region..."
            className="w-full pl-9 pr-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm placeholder:text-outline/50 focus:outline-none focus:border-primary transition-colors"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Statuses</option>
          {statuses.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <div className="flex items-center gap-2 text-xs text-outline">
          <SlidersHorizontal size={14} />
          <span>{filtered.length} client{filtered.length !== 1 ? 's' : ''}</span>
        </div>
      </div>

      {/* Loading skeleton */}
      {isLoading && (
        <Card className="overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-outline-variant/30 text-left text-[10px] uppercase tracking-widest font-bold text-outline">
                {['Client', 'Status', 'Device', 'Region', 'Accuracy', 'Loss', 'Round', 'Last Seen'].map((h) => (
                  <th key={h} className="px-4 py-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)}
            </tbody>
          </table>
        </Card>
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
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-outline-variant/30 text-left text-[10px] uppercase tracking-widest font-bold text-outline">
                  <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('name')}>
                    Client <SortIcon field="name" />
                  </th>
                  <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('status')}>
                    Status <SortIcon field="status" />
                  </th>
                  <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('device')}>
                    Device <SortIcon field="device" />
                  </th>
                  <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('region')}>
                    Region <SortIcon field="region" />
                  </th>
                  <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('accuracy')}>
                    Accuracy <SortIcon field="accuracy" />
                  </th>
                  <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('loss')}>
                    Loss <SortIcon field="loss" />
                  </th>
                  <th className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none" onClick={() => handleSort('last_round')}>
                    Round <SortIcon field="last_round" />
                  </th>
                  <th className="px-4 py-3">Last Seen</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {paged.map((client) => {
                  const DeviceIcon = getDeviceIcon(client.device);
                  return (
                    <tr
                      key={client.id}
                      className="border-b border-outline-variant/20 hover:bg-surface-container-low transition-colors"
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-lg bg-surface-container-high flex items-center justify-center">
                            <DeviceIcon size={16} className="text-primary" />
                          </div>
                          <div>
                            <p className="text-sm font-semibold text-on-surface">{client.name}</p>
                            <p className="text-[10px] font-mono text-outline">{client.id}</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={client.status} />
                      </td>
                      <td className="px-4 py-3 text-sm text-on-surface-variant">{client.device}</td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-1.5 text-sm text-on-surface-variant">
                          <MapPin size={12} className="text-outline shrink-0" />
                          <span>{client.region}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 max-w-[60px] h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary rounded-full transition-all"
                              style={{ width: `${client.accuracy * 100}%` }}
                            />
                          </div>
                          <span className="text-sm font-mono font-semibold text-primary">{(client.accuracy * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm font-mono text-on-surface">{client.loss.toFixed(4)}</td>
                      <td className="px-4 py-3 text-sm font-mono text-on-surface">#{client.last_round}</td>
                      <td className="px-4 py-3 text-sm text-outline">{formatTimestamp(client.last_communication)}</td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => setSelectedClient(client)}
                          className="p-1.5 rounded-lg hover:bg-surface-container-high text-outline hover:text-on-surface transition-colors cursor-pointer"
                          title="View details"
                        >
                          <Eye size={16} />
                        </button>
                      </td>
                    </tr>
                  );
                })}
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
      )}

      {/* Detail panel */}
      {selectedClient && (
        <DetailPanel
          client={selectedClient}
          onClose={() => setSelectedClient(null)}
        />
      )}
    </div>
  );
};
