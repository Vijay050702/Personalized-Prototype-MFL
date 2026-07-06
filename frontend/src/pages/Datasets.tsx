import { useState, useMemo, type FormEvent } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Plus,
  Search,
  Grid3X3,
  Table2,
  Download,
  FlaskConical,
  FlaskConical as PreprocessIcon,
  Share2,
  Unlink,
  Trash2,
  AlertTriangle,
  X,
  Eye,
  Database,
  RefreshCw,
  SlidersHorizontal,
} from 'lucide-react';

import {
  fetchDatasets,
  fetchDatasetDetail,
  registerDataset,
  downloadDataset,
  preprocessDataset,
  partitionDataset,
  simulateMissingModality,
  deleteDataset,
} from '../api/datasets';
import { Card } from '../components/ui/Card';
import { StatusBadge } from '../components/ui/StatusBadge';
import type {
  DatasetResponse,
  DatasetMetadataResponse,
  DatasetRegistrationRequest,
  PartitionRequest,
  MissingModalityRequest,
  OperationResponse,
  PartitionResponse,
} from '../types';

const PAGE_SIZES = [5, 10, 20, 50];

/* ------------------------------------------------------------------------- */
/*  Helpers                                                                  */
/* ------------------------------------------------------------------------- */

function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}

function formatBytes(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(1)} MB`;
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function truncate(str: string, len: number): string {
  return str.length > len ? `${str.slice(0, len)}...` : str;
}

/* ------------------------------------------------------------------------- */
/*  Skeleton                                                                 */
/* ------------------------------------------------------------------------- */

function SkeletonRow() {
  return (
    <tr className="animate-pulse border-b border-outline-variant/30">
      {Array.from({ length: 7 }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <div className="h-4 rounded bg-surface-container-high" style={{ width: `${60 + Math.random() * 30}%` }} />
        </td>
      ))}
    </tr>
  );
}

function SkeletonGridCard() {
  return (
    <Card className="p-5 animate-pulse">
      <div className="flex items-start justify-between mb-4">
        <div className="w-12 h-12 rounded-2xl bg-surface-container-high" />
        <div className="h-5 w-20 rounded-md bg-surface-container-high" />
      </div>
      <div className="h-5 w-3/4 rounded bg-surface-container-high mb-2" />
      <div className="h-3 w-1/2 rounded bg-surface-container-high mb-6" />
      <div className="grid grid-cols-2 gap-3">
        <div className="h-16 rounded-xl bg-surface-container-high" />
        <div className="h-16 rounded-xl bg-surface-container-high" />
      </div>
    </Card>
  );
}

/* ------------------------------------------------------------------------- */
/*  Empty state                                                              */
/* ------------------------------------------------------------------------- */

function EmptyState({ onRegister }: { onRegister: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <div className="w-16 h-16 rounded-2xl bg-surface-container-high flex items-center justify-center">
        <Database size={32} className="text-outline" />
      </div>
      <h3 className="text-lg font-display font-bold text-on-surface">No datasets found</h3>
      <p className="text-sm text-outline max-w-sm text-center">
        Get started by registering a dataset from the federation.
      </p>
      <button
        onClick={onRegister}
        className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity cursor-pointer"
      >
        <Plus size={16} />
        Register Dataset
      </button>
    </div>
  );
}

/* ------------------------------------------------------------------------- */
/*  Error state                                                              */
/* ------------------------------------------------------------------------- */

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <div className="w-14 h-14 rounded-2xl bg-rose-500/10 flex items-center justify-center">
        <AlertTriangle size={28} className="text-rose-400" />
      </div>
      <h3 className="text-lg font-display font-bold text-on-surface">Failed to load datasets</h3>
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

/* ------------------------------------------------------------------------- */
/*  Register Modal                                                           */
/* ------------------------------------------------------------------------- */

function RegisterModal({
  open,
  onClose,
  onRegister,
  isLoading,
}: {
  open: boolean;
  onClose: () => void;
  onRegister: (data: DatasetRegistrationRequest) => void;
  isLoading: boolean;
}) {
  const [name, setName] = useState('');
  const [modality, setModality] = useState('image');
  const [path, setPath] = useState('');

  if (!open) return null;

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    onRegister({ name: name.trim(), modality, path: path.trim() || undefined });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <Card className="w-full max-w-lg mx-4 p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-display font-bold text-on-surface">Register Dataset</h2>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-surface-container-high transition-colors cursor-pointer">
            <X size={20} className="text-outline" />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-outline mb-1.5">Dataset Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. CIFAR-10"
              className="w-full px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm placeholder:text-outline/50 focus:outline-none focus:border-primary transition-colors"
              required
              disabled={isLoading}
            />
          </div>
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-outline mb-1.5">Modality</label>
            <select
              value={modality}
              onChange={(e) => setModality(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
              disabled={isLoading}
            >
              <option value="image">Image</option>
              <option value="text">Text</option>
              <option value="tabular">Tabular</option>
              <option value="audio">Audio</option>
              <option value="multimodal">Multimodal</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-outline mb-1.5">Path (optional)</label>
            <input
              type="text"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="e.g. /data/datasets/cifar10"
              className="w-full px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm placeholder:text-outline/50 focus:outline-none focus:border-primary transition-colors"
              disabled={isLoading}
            />
          </div>
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl border border-outline-variant text-sm font-medium text-on-surface hover:bg-surface-container-high transition-colors cursor-pointer"
              disabled={isLoading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-50 cursor-pointer"
              disabled={isLoading || !name.trim()}
            >
              {isLoading ? 'Registering...' : 'Register'}
            </button>
          </div>
        </form>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------------- */
/*  Detail Panel                                                             */
/* ------------------------------------------------------------------------- */

function DetailPanel({
  dataset,
  metadata,
  onClose,
  onDownload,
  onPreprocess,
  onPartition,
  onMissingModality,
  onDelete,
  isDownloading,
  isPreprocessing,
}: {
  dataset: DatasetResponse;
  metadata: DatasetMetadataResponse | null;
  onClose: () => void;
  onDownload: () => void;
  onPreprocess: () => void;
  onPartition: () => void;
  onMissingModality: () => void;
  onDelete: () => void;
  isDownloading: boolean;
  isPreprocessing: boolean;
}) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/40 backdrop-blur-sm">
      <Card className="w-full max-w-lg h-full overflow-y-auto rounded-none rounded-l-2xl p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-display font-bold text-on-surface">{dataset.name}</h2>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-surface-container-high transition-colors cursor-pointer">
            <X size={20} className="text-outline" />
          </button>
        </div>

        <div className="space-y-4 mb-6">
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Type</p>
              <p className="text-sm font-semibold text-on-surface">{dataset.type}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Modality</p>
              <p className="text-sm font-semibold text-on-surface">{dataset.modality}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Size</p>
              <p className="text-sm font-semibold text-on-surface">{formatBytes(dataset.size_mb)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Samples</p>
              <p className="text-sm font-semibold text-on-surface">{formatNumber(dataset.samples)}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Classes</p>
              <p className="text-sm font-semibold text-on-surface">{dataset.classes}</p>
            </div>
            <div className="p-3 bg-surface-container-low rounded-xl">
              <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Distribution</p>
              <p className="text-sm font-semibold text-on-surface">{dataset.distribution}</p>
            </div>
          </div>

          {metadata && (
            <div className="space-y-3 pt-4 border-t border-outline-variant/50">
              <h3 className="text-sm font-bold text-on-surface">Processing Status</h3>
              <div className="flex items-center justify-between">
                <span className="text-xs text-outline">Download</span>
                <StatusBadge status={metadata.download_status} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-outline">Preprocessing</span>
                <StatusBadge status={metadata.preprocessing_status} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-outline">Partition</span>
                <StatusBadge status={metadata.partition_status} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-outline">Missing Modality</span>
                <span className="text-sm font-mono text-on-surface">{(metadata.missing_modality_ratio * 100).toFixed(1)}%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-outline">Clients</span>
                <span className="text-sm font-mono text-on-surface">{metadata.client_count}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-outline">Classes</span>
                <span className="text-sm font-mono text-on-surface">{metadata.num_classes}</span>
              </div>
            </div>
          )}
        </div>

        <div className="space-y-2">
          <button
            onClick={onDownload}
            disabled={isDownloading}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-50 cursor-pointer"
          >
            <Download size={16} />
            {isDownloading ? 'Downloading...' : 'Download'}
          </button>
          <button
            onClick={onPreprocess}
            disabled={isPreprocessing}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-surface-container-high text-on-surface text-sm font-semibold hover:bg-surface-container-highest transition-colors disabled:opacity-50 cursor-pointer"
          >
            <PreprocessIcon size={16} />
            {isPreprocessing ? 'Preprocessing...' : 'Preprocess'}
          </button>
          <button
            onClick={onPartition}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-surface-container-high text-on-surface text-sm font-semibold hover:bg-surface-container-highest transition-colors cursor-pointer"
          >
            <Share2 size={16} />
            Partition
          </button>
          <button
            onClick={onMissingModality}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-surface-container-high text-on-surface text-sm font-semibold hover:bg-surface-container-highest transition-colors cursor-pointer"
          >
            <Unlink size={16} />
            Missing Modality
          </button>
          <button
            onClick={onDelete}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-rose-500/10 text-rose-400 text-sm font-semibold hover:bg-rose-500/20 transition-colors cursor-pointer"
          >
            <Trash2 size={16} />
            Delete Dataset
          </button>
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------------- */
/*  Partition Modal                                                          */
/* ------------------------------------------------------------------------- */

function PartitionModal({
  open,
  datasetName,
  onClose,
  onPartition,
  isLoading,
}: {
  open: boolean;
  datasetName: string;
  onClose: () => void;
  onPartition: (data: PartitionRequest) => void;
  isLoading: boolean;
}) {
  const [strategy, setStrategy] = useState('iid');
  const [numClients, setNumClients] = useState(10);
  const [alpha, setAlpha] = useState(0.5);
  const [seed, setSeed] = useState(42);
  const [balanced, setBalanced] = useState(true);
  const [shardsPerClient, setShardsPerClient] = useState(2);

  if (!open) return null;

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    onPartition({
      dataset_name: datasetName,
      strategy,
      num_clients: numClients,
      alpha: strategy === 'dirichlet' ? alpha : undefined,
      seed,
      balanced,
      shards_per_client: strategy === 'shard' ? shardsPerClient : undefined,
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <Card className="w-full max-w-md mx-4 p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-display font-bold text-on-surface">Partition Dataset</h2>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-surface-container-high transition-colors cursor-pointer">
            <X size={20} className="text-outline" />
          </button>
        </div>
        <p className="text-xs text-outline mb-4">Partitioning: <span className="text-on-surface font-semibold">{datasetName}</span></p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-outline mb-1.5">Strategy</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
              disabled={isLoading}
            >
              <option value="iid">IID (Independent & Identically Distributed)</option>
              <option value="dirichlet">Dirichlet (Non-IID)</option>
              <option value="shard">Shard (Non-IID)</option>
            </select>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-outline mb-1.5">Clients</label>
              <input
                type="number"
                min={1}
                max={1000}
                value={numClients}
                onChange={(e) => setNumClients(Number(e.target.value))}
                className="w-full px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
                disabled={isLoading}
              />
            </div>
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-outline mb-1.5">Seed</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
                disabled={isLoading}
              />
            </div>
          </div>
          {strategy === 'dirichlet' && (
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-outline mb-1.5">Alpha (Dirichlet concentration)</label>
              <input
                type="number"
                min={0.01}
                step={0.1}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                className="w-full px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
                disabled={isLoading}
              />
            </div>
          )}
          {strategy === 'shard' && (
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-outline mb-1.5">Shards per Client</label>
              <input
                type="number"
                min={1}
                value={shardsPerClient}
                onChange={(e) => setShardsPerClient(Number(e.target.value))}
                className="w-full px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
                disabled={isLoading}
              />
            </div>
          )}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={balanced}
              onChange={(e) => setBalanced(e.target.checked)}
              className="rounded border-outline-variant text-primary focus:ring-primary"
              disabled={isLoading}
            />
            <span className="text-xs text-on-surface font-medium">Balanced partition</span>
          </label>
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl border border-outline-variant text-sm font-medium text-on-surface hover:bg-surface-container-high transition-colors cursor-pointer"
              disabled={isLoading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-50 cursor-pointer"
              disabled={isLoading}
            >
              {isLoading ? 'Partitioning...' : 'Partition'}
            </button>
          </div>
        </form>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------------- */
/*  Missing Modality Modal                                                   */
/* ------------------------------------------------------------------------- */

function MissingModalityModal({
  open,
  datasetName,
  onClose,
  onSimulate,
  isLoading,
}: {
  open: boolean;
  datasetName: string;
  onClose: () => void;
  onSimulate: (data: MissingModalityRequest) => void;
  isLoading: boolean;
}) {
  const [strategy, setStrategy] = useState('random');
  const [missingRatio, setMissingRatio] = useState(0.3);
  const [seed, setSeed] = useState(42);

  if (!open) return null;

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    onSimulate({
      dataset_name: datasetName,
      strategy,
      missing_ratio: missingRatio,
      seed,
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <Card className="w-full max-w-md mx-4 p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-display font-bold text-on-surface">Simulate Missing Modality</h2>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-surface-container-high transition-colors cursor-pointer">
            <X size={20} className="text-outline" />
          </button>
        </div>
        <p className="text-xs text-outline mb-4">Dataset: <span className="text-on-surface font-semibold">{datasetName}</span></p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-outline mb-1.5">Strategy</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
              disabled={isLoading}
            >
              <option value="random">Random</option>
              <option value="modality_wise">Modality-wise</option>
              <option value="client_wise">Client-wise</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-outline mb-1.5">Missing Ratio</label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={missingRatio}
              onChange={(e) => setMissingRatio(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
              disabled={isLoading}
            />
          </div>
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-outline mb-1.5">Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
              disabled={isLoading}
            />
          </div>
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl border border-outline-variant text-sm font-medium text-on-surface hover:bg-surface-container-high transition-colors cursor-pointer"
              disabled={isLoading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-50 cursor-pointer"
              disabled={isLoading}
            >
              {isLoading ? 'Simulating...' : 'Simulate'}
            </button>
          </div>
        </form>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------------- */
/*  Delete Confirmation                                                      */
/* ------------------------------------------------------------------------- */

function DeleteConfirm({
  open,
  datasetName,
  onClose,
  onConfirm,
  isLoading,
}: {
  open: boolean;
  datasetName: string;
  onClose: () => void;
  onConfirm: () => void;
  isLoading: boolean;
}) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <Card className="w-full max-w-sm mx-4 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-rose-500/10 flex items-center justify-center">
            <AlertTriangle size={20} className="text-rose-400" />
          </div>
          <div>
            <h2 className="text-lg font-display font-bold text-on-surface">Delete Dataset</h2>
            <p className="text-xs text-outline">This action cannot be undone.</p>
          </div>
        </div>
        <p className="text-sm text-on-surface mb-6">
          Are you sure you want to delete <strong>{datasetName}</strong>? All partitions and processed data will be removed.
        </p>
        <div className="flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl border border-outline-variant text-sm font-medium text-on-surface hover:bg-surface-container-high transition-colors cursor-pointer"
            disabled={isLoading}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 rounded-xl bg-rose-500 text-white text-sm font-semibold hover:bg-rose-600 transition-colors disabled:opacity-50 cursor-pointer"
            disabled={isLoading}
          >
            {isLoading ? 'Deleting...' : 'Delete'}
          </button>
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------------- */
/*  Toast Notification                                                       */
/* ------------------------------------------------------------------------- */

function Toast({
  message,
  type,
  onClose,
}: {
  message: string;
  type: 'success' | 'error';
  onClose: () => void;
}) {
  return (
    <div
      className={cn(
        'fixed bottom-6 right-6 z-[60] flex items-center gap-3 px-5 py-3 rounded-xl shadow-lg text-sm font-medium animate-in slide-in-from-bottom-2',
        type === 'success' ? 'bg-emerald-600 text-white' : 'bg-rose-600 text-white',
      )}
    >
      <span>{message}</span>
      <button onClick={onClose} className="p-0.5 hover:opacity-80 transition-opacity cursor-pointer">
        <X size={16} />
      </button>
    </div>
  );
}

/* ------------------------------------------------------------------------- */
/*  Main Datasets Page                                                       */
/* ------------------------------------------------------------------------- */

export const Datasets = () => {
  const queryClient = useQueryClient();

  /* view state */
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('table');
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState('');
  const [sortField, setSortField] = useState<string>('name');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(10);

  /* modal state */
  const [showRegister, setShowRegister] = useState(false);
  const [detailDataset, setDetailDataset] = useState<DatasetResponse | null>(null);
  const [showPartition, setShowPartition] = useState(false);
  const [showMissingModality, setShowMissingModality] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  /* toast */
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

  /* queries */
  const {
    data: listResponse,
    isLoading,
    isError,
    error,
    refetch,
    isRefetching,
  } = useQuery({
    queryKey: ['datasets'],
    queryFn: fetchDatasets,
    retry: 2,
    staleTime: 10000,
  });

  const {
    data: detailResponse,
    isLoading: detailLoading,
  } = useQuery({
    queryKey: ['dataset-detail', detailDataset?.name],
    queryFn: () => fetchDatasetDetail(detailDataset!.name),
    enabled: !!detailDataset,
    retry: 1,
    staleTime: 5000,
  });

  const metadata: DatasetMetadataResponse | null = detailResponse?.data ?? null;

  /* mutations */
  const registerMut = useMutation({
    mutationFn: registerDataset,
    onSuccess: (res: OperationResponse) => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      setShowRegister(false);
      setToast({ message: res.message || `Dataset "${res.dataset_name}" registered`, type: 'success' });
    },
    onError: (err: Error) => setToast({ message: err.message, type: 'error' }),
  });

  const downloadMut = useMutation({
    mutationFn: downloadDataset,
    onSuccess: (res: OperationResponse) => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      queryClient.invalidateQueries({ queryKey: ['dataset-detail'] });
      setToast({ message: res.message || `Download started for "${res.dataset_name}"`, type: 'success' });
    },
    onError: (err: Error) => setToast({ message: err.message, type: 'error' }),
  });

  const preprocessMut = useMutation({
    mutationFn: preprocessDataset,
    onSuccess: (res: OperationResponse) => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      queryClient.invalidateQueries({ queryKey: ['dataset-detail'] });
      setToast({ message: res.message || `Preprocessing started for "${res.dataset_name}"`, type: 'success' });
    },
    onError: (err: Error) => setToast({ message: err.message, type: 'error' }),
  });

  const partitionMut = useMutation({
    mutationFn: partitionDataset,
    onSuccess: (res: PartitionResponse) => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      queryClient.invalidateQueries({ queryKey: ['dataset-detail'] });
      setShowPartition(false);
      setToast({ message: `Partitioned "${res.dataset_name}" (${res.strategy}, ${res.num_clients} clients)`, type: 'success' });
    },
    onError: (err: Error) => setToast({ message: err.message, type: 'error' }),
  });

  const missingModMut = useMutation({
    mutationFn: simulateMissingModality,
    onSuccess: (res: OperationResponse) => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      queryClient.invalidateQueries({ queryKey: ['dataset-detail'] });
      setShowMissingModality(false);
      setToast({ message: res.message || `Missing modality simulated for "${res.dataset_name}"`, type: 'success' });
    },
    onError: (err: Error) => setToast({ message: err.message, type: 'error' }),
  });

  const deleteMut = useMutation({
    mutationFn: deleteDataset,
    onSuccess: (res: OperationResponse) => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      setDeleteTarget(null);
      setDetailDataset(null);
      setToast({ message: res.message || `Dataset "${res.dataset_name}" deleted`, type: 'success' });
    },
    onError: (err: Error) => setToast({ message: err.message, type: 'error' }),
  });

  /* derived data */
  const allDatasets: DatasetResponse[] = listResponse?.data ?? [];

  const types = useMemo(() => {
    const set = new Set(allDatasets.map((d) => d.type));
    return Array.from(set).sort();
  }, [allDatasets]);

  const filtered = useMemo(() => {
    let list = allDatasets;

    if (search) {
      const q = search.toLowerCase();
      list = list.filter(
        (d) =>
          d.name.toLowerCase().includes(q) ||
          d.type.toLowerCase().includes(q) ||
          d.modality.toLowerCase().includes(q),
      );
    }

    if (filterType) {
      list = list.filter((d) => d.type === filterType);
    }

    list.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case 'name':
          cmp = a.name.localeCompare(b.name);
          break;
        case 'type':
          cmp = a.type.localeCompare(b.type);
          break;
        case 'modality':
          cmp = a.modality.localeCompare(b.modality);
          break;
        case 'size_mb':
          cmp = a.size_mb - b.size_mb;
          break;
        case 'samples':
          cmp = a.samples - b.samples;
          break;
        case 'classes':
          cmp = a.classes - b.classes;
          break;
        default:
          cmp = 0;
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return list;
  }, [allDatasets, search, filterType, sortField, sortDir]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(page, totalPages - 1);
  const paged = filtered.slice(safePage * pageSize, (safePage + 1) * pageSize);

  /* handlers */
  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDir('asc');
    }
  };

  const handleViewDetail = (ds: DatasetResponse) => {
    setDetailDataset(ds);
  };

  const handleRegister = (data: DatasetRegistrationRequest) => {
    registerMut.mutate(data);
  };

  const handleDownload = () => {
    if (detailDataset) downloadMut.mutate({ dataset_name: detailDataset.name });
  };

  const handlePreprocess = () => {
    if (detailDataset) preprocessMut.mutate({ dataset_name: detailDataset.name });
  };

  const handlePartition = (data: PartitionRequest) => {
    partitionMut.mutate(data);
  };

  const handleMissingModality = (data: MissingModalityRequest) => {
    missingModMut.mutate(data);
  };

  const handleDelete = () => {
    if (deleteTarget) deleteMut.mutate(deleteTarget);
  };

  /* Sort indicator */
  function SortIcon({ field }: { field: string }) {
    if (sortField !== field) return <span className="text-outline/30 ml-1">&#8597;</span>;
    return <span className="text-primary ml-1">{sortDir === 'asc' ? '&#8593;' : '&#8595;'}</span>;
  }

  /* ---- render ---- */
  return (
    <div className="space-y-6">
      {/* Toast */}
      {toast && (
        <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />
      )}

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">Dataset Manager</h1>
          <p className="text-sm text-outline">Browse, register, and manage datasets in the federation</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => { setViewMode('table'); }}
            className={cn(
              'p-2 rounded-lg transition-colors cursor-pointer',
              viewMode === 'table' ? 'bg-primary text-on-primary' : 'bg-surface-container-high text-outline hover:text-on-surface',
            )}
            title="Table view"
          >
            <Table2 size={18} />
          </button>
          <button
            onClick={() => { setViewMode('grid'); }}
            className={cn(
              'p-2 rounded-lg transition-colors cursor-pointer',
              viewMode === 'grid' ? 'bg-primary text-on-primary' : 'bg-surface-container-high text-outline hover:text-on-surface',
            )}
            title="Grid view"
          >
            <Grid3X3 size={18} />
          </button>
          <div className="w-px h-6 bg-outline-variant/50" />
          <button
            onClick={() => setShowRegister(true)}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity cursor-pointer"
          >
            <Plus size={18} />
            Register
          </button>
        </div>
      </div>

      {/* Search & filters */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-outline" />
          <input
            type="text"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0); }}
            placeholder="Search datasets..."
            className="w-full pl-9 pr-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm placeholder:text-outline/50 focus:outline-none focus:border-primary transition-colors"
          />
        </div>
        <select
          value={filterType}
          onChange={(e) => { setFilterType(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-xl bg-surface-container-low border border-outline-variant text-on-surface text-sm focus:outline-none focus:border-primary transition-colors"
        >
          <option value="">All Types</option>
          {types.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <div className="flex items-center gap-2 text-xs text-outline">
          <SlidersHorizontal size={14} />
          <span>{filtered.length} dataset{filtered.length !== 1 ? 's' : ''}</span>
        </div>
      </div>

      {/* Loading skeleton */}
      {isLoading && (
        viewMode === 'table' ? (
          <Card className="overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="border-b border-outline-variant/30 text-left text-[10px] uppercase tracking-widest font-bold text-outline">
                  {['Name', 'Type', 'Modality', 'Size', 'Samples', 'Classes', 'Distribution'].map((h) => (
                    <th key={h} className="px-4 py-3">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)}
              </tbody>
            </table>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {Array.from({ length: 6 }).map((_, i) => <SkeletonGridCard key={i} />)}
          </div>
        )
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
        <EmptyState onRegister={() => setShowRegister(true)} />
      )}

      {/* Data table */}
      {!isLoading && !isError && filtered.length > 0 && viewMode === 'table' && (
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-outline-variant/30 text-left text-[10px] uppercase tracking-widest font-bold text-outline">
                  <th
                    className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none"
                    onClick={() => handleSort('name')}
                  >
                    Name <SortIcon field="name" />
                  </th>
                  <th
                    className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none"
                    onClick={() => handleSort('type')}
                  >
                    Type <SortIcon field="type" />
                  </th>
                  <th
                    className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none"
                    onClick={() => handleSort('modality')}
                  >
                    Modality <SortIcon field="modality" />
                  </th>
                  <th
                    className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none"
                    onClick={() => handleSort('size_mb')}
                  >
                    Size <SortIcon field="size_mb" />
                  </th>
                  <th
                    className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none"
                    onClick={() => handleSort('samples')}
                  >
                    Samples <SortIcon field="samples" />
                  </th>
                  <th
                    className="px-4 py-3 cursor-pointer hover:text-on-surface transition-colors select-none"
                    onClick={() => handleSort('classes')}
                  >
                    Classes <SortIcon field="classes" />
                  </th>
                  <th className="px-4 py-3">Distribution</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {paged.map((ds) => (
                  <tr
                    key={ds.id}
                    className="border-b border-outline-variant/20 hover:bg-surface-container-low transition-colors"
                  >
                    <td className="px-4 py-3">
                      <button
                        onClick={() => handleViewDetail(ds)}
                        className="text-sm font-semibold text-on-surface hover:text-primary transition-colors text-left cursor-pointer"
                      >
                        {truncate(ds.name, 30)}
                      </button>
                    </td>
                    <td className="px-4 py-3 text-sm text-on-surface-variant">{ds.type}</td>
                    <td className="px-4 py-3">
                      <StatusBadge status={ds.modality} />
                    </td>
                    <td className="px-4 py-3 text-sm font-mono text-on-surface">{formatBytes(ds.size_mb)}</td>
                    <td className="px-4 py-3 text-sm font-mono text-on-surface">{formatNumber(ds.samples)}</td>
                    <td className="px-4 py-3 text-sm font-mono text-on-surface">{ds.classes}</td>
                    <td className="px-4 py-3 text-sm text-on-surface-variant">{ds.distribution}</td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={() => handleViewDetail(ds)}
                          className="p-1.5 rounded-lg hover:bg-surface-container-high text-outline hover:text-on-surface transition-colors cursor-pointer"
                          title="View details"
                        >
                          <Eye size={16} />
                        </button>
                        <button
                          onClick={() => {
                            setDetailDataset(ds);
                            downloadMut.mutate({ dataset_name: ds.name });
                          }}
                          className="p-1.5 rounded-lg hover:bg-surface-container-high text-outline hover:text-on-surface transition-colors cursor-pointer"
                          title="Download"
                        >
                          <Download size={16} />
                        </button>
                        <button
                          onClick={() => {
                            setDetailDataset(ds);
                            setShowPartition(true);
                          }}
                          className="p-1.5 rounded-lg hover:bg-surface-container-high text-outline hover:text-on-surface transition-colors cursor-pointer"
                          title="Partition"
                        >
                          <Share2 size={16} />
                        </button>
                        <button
                          onClick={() => setDeleteTarget(ds.name)}
                          className="p-1.5 rounded-lg hover:bg-rose-500/10 text-outline hover:text-rose-400 transition-colors cursor-pointer"
                          title="Delete"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
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
      )}

      {/* Data grid */}
      {!isLoading && !isError && filtered.length > 0 && viewMode === 'grid' && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {paged.map((ds) => (
            <Card key={ds.id} className="p-5 group cursor-pointer hover:border-primary/50 transition-all" onClick={() => handleViewDetail(ds)}>
              <div className="flex items-start justify-between mb-4">
                <div className="w-12 h-12 rounded-2xl bg-surface-container-high flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-on-primary transition-colors">
                  <Database size={24} />
                </div>
                <StatusBadge status={ds.modality} />
              </div>
              <h3 className="text-lg font-bold text-on-surface mb-1">{ds.name}</h3>
              <p className="text-xs text-outline mb-6">Type: {ds.type} &middot; {ds.distribution}</p>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-surface-container-low rounded-xl">
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Size</p>
                  <p className="text-xl font-display font-bold text-on-surface">{formatBytes(ds.size_mb)}</p>
                </div>
                <div className="p-3 bg-surface-container-low rounded-xl">
                  <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Samples</p>
                  <p className="text-xl font-display font-bold text-on-surface">{formatNumber(ds.samples)}</p>
                </div>
              </div>
              <div className="mt-4 flex items-center justify-between text-xs text-outline">
                <span>{ds.classes} classes</span>
                <span>Client: {truncate(ds.client_id, 12)}</span>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Grid pagination */}
      {!isLoading && !isError && filtered.length > 0 && viewMode === 'grid' && (
        <div className="flex items-center justify-center gap-4 text-xs text-outline">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={safePage === 0}
            className="px-3 py-1.5 rounded-lg border border-outline-variant hover:bg-surface-container-high disabled:opacity-30 transition-colors cursor-pointer"
          >
            Previous
          </button>
          <span>Page {safePage + 1} of {totalPages}</span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={safePage >= totalPages - 1}
            className="px-3 py-1.5 rounded-lg border border-outline-variant hover:bg-surface-container-high disabled:opacity-30 transition-colors cursor-pointer"
          >
            Next
          </button>
        </div>
      )}

      {/* Modals */}
      <RegisterModal
        open={showRegister}
        onClose={() => setShowRegister(false)}
        onRegister={handleRegister}
        isLoading={registerMut.isPending}
      />

      {detailDataset && (
        <DetailPanel
          dataset={detailDataset}
          metadata={detailLoading ? null : metadata}
          onClose={() => setDetailDataset(null)}
          onDownload={handleDownload}
          onPreprocess={handlePreprocess}
          onPartition={() => setShowPartition(true)}
          onMissingModality={() => setShowMissingModality(true)}
          onDelete={() => setDeleteTarget(detailDataset.name)}
          isDownloading={downloadMut.isPending}
          isPreprocessing={preprocessMut.isPending}
        />
      )}

      <PartitionModal
        open={showPartition}
        datasetName={detailDataset?.name ?? ''}
        onClose={() => setShowPartition(false)}
        onPartition={handlePartition}
        isLoading={partitionMut.isPending}
      />

      <MissingModalityModal
        open={showMissingModality}
        datasetName={detailDataset?.name ?? ''}
        onClose={() => setShowMissingModality(false)}
        onSimulate={handleMissingModality}
        isLoading={missingModMut.isPending}
      />

      <DeleteConfirm
        open={!!deleteTarget}
        datasetName={deleteTarget ?? ''}
        onClose={() => setDeleteTarget(null)}
        onConfirm={handleDelete}
        isLoading={deleteMut.isPending}
      />
    </div>
  );
};
