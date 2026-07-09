import { useState, useEffect, useMemo, useCallback, useRef, type ReactNode } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import {
  RefreshCw, AlertTriangle, Save, RotateCcw, X, Settings as SettingsIcon,
  Cpu, Shield, Network, Database, Server, Sliders, FileJson, Eye,
} from 'lucide-react';

import { fetchSettings, updateSettings } from '../api/settings';
import { Card } from '../components/ui/Card';
import type { SettingsResponse } from '../types';

type SettingsTab = 'general' | 'training' | 'federated' | 'prototype' | 'knowledge_transfer' | 'personalization' | 'logging' | 'about';

function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}

function formatTimestamp(iso: string): string {
  try {
    const date = new Date(iso);
    return date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return iso;
  }
}

function getErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const data = error.response?.data as Record<string, unknown> | undefined;
    if (data?.detail) return String(data.detail);
    if (data?.message) return String(data.message);
    if (error.response?.status === 404) return 'Settings endpoint not found.';
    if (error.response?.status === 422) return 'Invalid settings values. Please check your inputs.';
    if (error.response?.status === 405) return 'Settings cannot be saved. The backend does not support updates.';
    if (error.response?.status === 500) return 'Server error. Please try again later.';
    if (error.code === 'ECONNABORTED') return 'Request timed out. Please try again.';
    if (!error.response) return 'Backend is unavailable. Please check your connection.';
  }
  if (error instanceof Error) return error.message;
  return 'An unexpected error occurred.';
}

function getFieldError(field: string, value: string | number | boolean): string | null {
  if (typeof value === 'string') {
    if (!value.trim()) return `${field} is required.`;
  }
  if (typeof value === 'number') {
    if (field === 'Learning Rate' && (value <= 0 || value > 10)) return 'Must be between 0 and 10.';
    if (field === 'Batch Size' && (value < 1 || value > 4096 || !Number.isInteger(value))) return 'Must be an integer between 1 and 4096.';
    if ((field === 'Local Epochs' || field === 'Total Rounds' || field === 'Clients Per Round' || field === 'Min Clients') && (value < 1 || !Number.isInteger(value))) return 'Must be a positive integer.';
    if (field === 'Prototype Dimension' && (value < 16 || value > 4096 || !Number.isInteger(value))) return 'Must be an integer between 16 and 4096.';
  }
  return null;
}

const TABS: { key: SettingsTab; label: string; icon: ReactNode }[] = [
  { key: 'general', label: 'General', icon: <SettingsIcon size={16} /> },
  { key: 'training', label: 'Training', icon: <Cpu size={16} /> },
  { key: 'federated', label: 'Federated Learning', icon: <Network size={16} /> },
  { key: 'prototype', label: 'Prototype', icon: <Database size={16} /> },
  { key: 'knowledge_transfer', label: 'Knowledge Transfer', icon: <Shield size={16} /> },
  { key: 'personalization', label: 'Personalization', icon: <Sliders size={16} /> },
  { key: 'logging', label: 'Logging', icon: <FileJson size={16} /> },
  { key: 'about', label: 'About', icon: <Server size={16} /> },
];

function SkeletonPanel() {
  return (
    <div className="space-y-6">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="space-y-2">
          <div className="h-3 w-28 rounded-full bg-surface-container-high animate-pulse" />
          <div className="h-10 w-full rounded-xl bg-surface-container-high animate-pulse" />
        </div>
      ))}
    </div>
  );
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
      <div className="w-14 h-14 rounded-2xl bg-rose-500/10 flex items-center justify-center">
        <AlertTriangle size={28} className="text-rose-400" />
      </div>
      <h2 className="text-lg font-display font-bold text-on-surface">Failed to load settings</h2>
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

function EmptyState({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
      <div className="w-14 h-14 rounded-2xl bg-outline/10 flex items-center justify-center">
        <SettingsIcon size={28} className="text-outline" />
      </div>
      <h2 className="text-lg font-display font-bold text-on-surface">No settings available</h2>
      <p className="text-sm text-outline max-w-md text-center">The server returned an empty response for settings.</p>
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

interface FieldProps {
  label: string;
  value: string | number | boolean;
  onChange: (value: string | number | boolean) => void;
  type?: 'text' | 'number' | 'boolean' | 'select';
  options?: { label: string; value: string }[];
  min?: number;
  max?: number;
  step?: number;
  error?: string | null;
  disabled?: boolean;
}

function Field({ label, value, onChange, type = 'text', options, min, max, step, error, disabled }: FieldProps) {
  const id = `field-${label.replace(/\s+/g, '-').toLowerCase()}`;
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="text-xs font-medium text-outline uppercase tracking-wider">{label}</label>
      {type === 'boolean' ? (
        <button
          id={id}
          type="button"
          onClick={() => onChange(!value)}
          disabled={disabled}
          className={cn(
            'relative inline-flex h-7 w-12 items-center rounded-full transition-colors cursor-pointer disabled:opacity-50',
            value ? 'bg-primary' : 'bg-surface-container-high',
          )}
        >
          <span className={cn('inline-block h-5 w-5 transform rounded-full bg-white transition-transform', value ? 'translate-x-6' : 'translate-x-1')} />
        </button>
      ) : type === 'select' ? (
        <select
          id={id}
          value={String(value)}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className={cn(
            'w-full px-3.5 py-2.5 rounded-xl bg-surface-container-high border text-sm text-on-surface focus:outline-none focus:border-primary/50 transition-all disabled:opacity-50',
            error ? 'border-rose-500/50' : 'border-outline-variant/50',
          )}
        >
          {options?.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      ) : type === 'number' ? (
        <input
          id={id}
          type="number"
          value={value}
          onChange={(e) => onChange(type === 'number' ? (e.target.value === '' ? '' : Number(e.target.value)) : e.target.value)}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          className={cn(
            'w-full px-3.5 py-2.5 rounded-xl bg-surface-container-high border text-sm text-on-surface font-mono focus:outline-none focus:border-primary/50 transition-all disabled:opacity-50',
            error ? 'border-rose-500/50' : 'border-outline-variant/50',
          )}
        />
      ) : (
        <input
          id={id}
          type="text"
          value={String(value)}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className={cn(
            'w-full px-3.5 py-2.5 rounded-xl bg-surface-container-high border text-sm text-on-surface font-mono focus:outline-none focus:border-primary/50 transition-all disabled:opacity-50',
            error ? 'border-rose-500/50' : 'border-outline-variant/50',
          )}
        />
      )}
      {error && <p className="text-[11px] text-rose-400 mt-0.5">{error}</p>}
    </div>
  );
}

function ComingSoonPanel({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 gap-3">
      <div className="w-12 h-12 rounded-2xl bg-outline/10 flex items-center justify-center">
        <SettingsIcon size={24} className="text-outline/40" />
      </div>
      <h3 className="text-sm font-semibold text-on-surface">{title}</h3>
      <p className="text-xs text-outline text-center max-w-sm">{description}</p>
    </div>
  );
}

export const Settings = () => {
  const [activeTab, setActiveTab] = useState<SettingsTab>('general');
  const [formValues, setFormValues] = useState<SettingsResponse | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string | null>>({});
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const [pendingTab, setPendingTab] = useState<SettingsTab | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout>>();

  const {
    data: settingsSummary,
    isLoading,
    isError,
    error,
    refetch,
    isRefetching,
    dataUpdatedAt,
  } = useQuery({
    queryKey: ['settings'],
    queryFn: fetchSettings,
    retry: 2,
    staleTime: 30000,
  });

  const settings = settingsSummary?.data ?? null;

  useEffect(() => {
    if (settings && !formValues) {
      setFormValues({ ...settings });
    }
  }, [settings, formValues]);

  useEffect(() => {
    return () => { if (toastTimer.current) clearTimeout(toastTimer.current); };
  }, []);

  const showToast = useCallback((message: string, type: 'success' | 'error') => {
    setToast({ message, type });
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 4000);
  }, []);

  const mutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: () => {
      showToast('Settings saved successfully.', 'success');
    },
    onError: (err: Error) => {
      showToast(getErrorMessage(err), 'error');
    },
  });

  const hasChanges = useMemo(() => {
    if (!settings || !formValues) return false;
    return (Object.keys(settings) as (keyof SettingsResponse)[]).some((key) => {
      return String(settings[key]) !== String(formValues[key]);
    });
  }, [settings, formValues]);

  const validateForm = useCallback((): boolean => {
    if (!formValues) return false;
    const errors: Record<string, string | null> = {};
    const fields = [
      { label: 'Learning Rate', value: formValues.learning_rate },
      { label: 'Batch Size', value: formValues.batch_size },
      { label: 'Local Epochs', value: formValues.local_epochs },
      { label: 'Total Rounds', value: formValues.total_rounds },
      { label: 'Clients Per Round', value: formValues.clients_per_round },
      { label: 'Min Clients', value: formValues.min_clients },
      { label: 'Prototype Dimension', value: formValues.prototype_dimension },
    ] as const;
    for (const f of fields) {
      errors[f.label] = getFieldError(f.label, f.value);
    }
    if (!formValues.federation_strategy.trim()) errors['Federation Strategy'] = 'Required.';
    if (!formValues.aggregation_algorithm.trim()) errors['Aggregation Algorithm'] = 'Required.';
    if (!formValues.model_architecture.trim()) errors['Model Architecture'] = 'Required.';
    if (!formValues.communication_protocol.trim()) errors['Communication Protocol'] = 'Required.';
    setFieldErrors(errors);
    return !Object.values(errors).some(Boolean);
  }, [formValues]);

  const handleSave = useCallback(() => {
    if (!formValues || !validateForm()) return;
    mutation.mutate(formValues);
  }, [formValues, validateForm, mutation]);

  const handleReset = useCallback(() => {
    if (settings) {
      setFormValues({ ...settings });
      setFieldErrors({});
    }
  }, [settings]);

  const updateField = useCallback((key: keyof SettingsResponse, value: string | number | boolean) => {
    setFormValues((prev) => (prev ? { ...prev, [key]: value } : prev));
    setFieldErrors((prev) => ({ ...prev, [key]: null }));
  }, []);

  const handleTabClick = useCallback((tab: SettingsTab) => {
    if (hasChanges) {
      setPendingTab(tab);
    } else {
      setActiveTab(tab);
    }
  }, [hasChanges]);

  const confirmTabChange = useCallback(() => {
    if (pendingTab) {
      if (settings) setFormValues({ ...settings });
      setFieldErrors({});
      setActiveTab(pendingTab);
      setPendingTab(null);
    }
  }, [pendingTab, settings]);

  const cancelTabChange = useCallback(() => {
    setPendingTab(null);
  }, []);

  const requiredStr = (v: string) => v || '—';

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-48 rounded-lg bg-surface-container-high animate-pulse" />
        <div className="flex gap-6">
          <div className="w-48 space-y-1">
            {Array.from({ length: 5 }).map((_, i) => <div key={i} className="h-10 w-full rounded-lg bg-surface-container-high animate-pulse" />)}
          </div>
          <div className="flex-1"><SkeletonPanel /></div>
        </div>
      </div>
    );
  }

  if (isError) {
    return <ErrorState message={getErrorMessage(error)} onRetry={() => refetch()} />;
  }

  if (!settings || !formValues) {
    return <EmptyState onRetry={() => refetch()} />;
  }

  const renderGeneral = () => (
    <div className="space-y-5">
      <Field label="Federation Strategy" value={formValues.federation_strategy} onChange={(v) => updateField('federation_strategy', v)} type="select" options={[
        { label: 'Personalized', value: 'personalized' },
        { label: 'Standard', value: 'standard' },
        { label: 'Clustered', value: 'clustered' },
        { label: 'Hierarchical', value: 'hierarchical' },
      ]} error={fieldErrors['Federation Strategy']} />
      <Field label="Model Architecture" value={formValues.model_architecture} onChange={(v) => updateField('model_architecture', v)} type="select" options={[
        { label: 'ResNet-18', value: 'ResNet-18' },
        { label: 'ResNet-34', value: 'ResNet-34' },
        { label: 'ResNet-50', value: 'ResNet-50' },
        { label: 'MobileNetV2', value: 'MobileNetV2' },
        { label: 'EfficientNet-B0', value: 'EfficientNet-B0' },
        { label: 'ViT-Base', value: 'ViT-Base' },
      ]} error={fieldErrors['Model Architecture']} />
      <Field label="Communication Protocol" value={formValues.communication_protocol} onChange={(v) => updateField('communication_protocol', v)} type="select" options={[
        { label: 'gRPC', value: 'gRPC' },
        { label: 'HTTP/2', value: 'HTTP/2' },
        { label: 'WebSocket', value: 'WebSocket' },
        { label: 'MQTT', value: 'MQTT' },
      ]} error={fieldErrors['Communication Protocol']} />
      <Field label="Encryption Enabled" value={formValues.encryption_enabled} onChange={(v) => updateField('encryption_enabled', v)} type="boolean" />
    </div>
  );

  const renderTraining = () => (
    <div className="space-y-5">
      <Field label="Learning Rate" value={formValues.learning_rate} onChange={(v) => updateField('learning_rate', Number(v))} type="number" min={0} max={10} step={0.0001} error={fieldErrors['Learning Rate']} />
      <Field label="Batch Size" value={formValues.batch_size} onChange={(v) => updateField('batch_size', Number(v))} type="number" min={1} max={4096} step={1} error={fieldErrors['Batch Size']} />
      <Field label="Local Epochs" value={formValues.local_epochs} onChange={(v) => updateField('local_epochs', Number(v))} type="number" min={1} max={100} step={1} error={fieldErrors['Local Epochs']} />
      <Field label="Total Rounds" value={formValues.total_rounds} onChange={(v) => updateField('total_rounds', Number(v))} type="number" min={1} max={10000} step={1} error={fieldErrors['Total Rounds']} />
    </div>
  );

  const renderFederated = () => (
    <div className="space-y-5">
      <Field label="Aggregation Algorithm" value={formValues.aggregation_algorithm} onChange={(v) => updateField('aggregation_algorithm', v)} type="select" options={[
        { label: 'FedAvg', value: 'FedAvg' },
        { label: 'FedProx', value: 'FedProx' },
        { label: 'SCAFFOLD', value: 'SCAFFOLD' },
        { label: 'pFedProto', value: 'pFedProto' },
      ]} error={fieldErrors['Aggregation Algorithm']} />
      <Field label="Clients Per Round" value={formValues.clients_per_round} onChange={(v) => updateField('clients_per_round', Number(v))} type="number" min={1} max={1000} step={1} error={fieldErrors['Clients Per Round']} />
      <Field label="Min Clients" value={formValues.min_clients} onChange={(v) => updateField('min_clients', Number(v))} type="number" min={1} max={100} step={1} error={fieldErrors['Min Clients']} />
    </div>
  );

  const renderPrototype = () => (
    <div className="space-y-5">
      <Field label="Prototype Dimension" value={formValues.prototype_dimension} onChange={(v) => updateField('prototype_dimension', Number(v))} type="number" min={16} max={4096} step={16} error={fieldErrors['Prototype Dimension']} />
      <div className="p-4 bg-surface-container-high rounded-xl">
        <p className="text-xs text-outline">Prototype dimensions determine the size of learned prototype representations for each modality. Higher dimensions capture more detail but increase memory and communication costs.</p>
      </div>
    </div>
  );

  const renderAbout = () => (
    <div className="space-y-5">
      <div className="p-5 bg-surface-container-high rounded-xl">
        <h3 className="text-sm font-bold text-on-surface mb-4">System Information</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
            <span className="text-xs text-outline">Version</span>
            <span className="text-sm font-mono font-bold text-on-surface">2.1.0-stable</span>
          </div>
          <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
            <span className="text-xs text-outline">Federation Strategy</span>
            <span className="text-sm font-mono text-primary">{requiredStr(formValues.federation_strategy)}</span>
          </div>
          <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
            <span className="text-xs text-outline">Aggregation Algorithm</span>
            <span className="text-sm font-mono text-primary">{requiredStr(formValues.aggregation_algorithm)}</span>
          </div>
          <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
            <span className="text-xs text-outline">Model Architecture</span>
            <span className="text-sm font-mono text-on-surface">{requiredStr(formValues.model_architecture)}</span>
          </div>
          <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
            <span className="text-xs text-outline">Encryption</span>
            <span className="text-sm font-mono text-on-surface">{formValues.encryption_enabled ? 'Enabled' : 'Disabled'}</span>
          </div>
          <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
            <span className="text-xs text-outline">Backend</span>
            <span className="text-sm font-mono text-on-surface">FastAPI · Python 3.11</span>
          </div>
          <div className="flex items-center justify-between py-2">
            <span className="text-xs text-outline">Last Updated</span>
            <span className="text-xs font-mono text-outline">{dataUpdatedAt ? formatTimestamp(new Date(dataUpdatedAt).toISOString()) : '—'}</span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'general': return renderGeneral();
      case 'training': return renderTraining();
      case 'federated': return renderFederated();
      case 'prototype': return renderPrototype();
      case 'knowledge_transfer': return <ComingSoonPanel title="Knowledge Transfer Configuration" description="Knowledge transfer settings will be available in a future update. This panel will allow configuration of cross-modal transfer strategies, alignment methods, and transfer thresholds." />;
      case 'personalization': return <ComingSoonPanel title="Personalization Configuration" description="Personalization settings will be available in a future update. This panel will allow configuration of client-specific adaptation parameters." />;
      case 'logging': return <ComingSoonPanel title="Logging Configuration" description="Logging settings will be available in a future update. This panel will allow configuration of log levels, retention policies, and export settings." />;
      case 'about': return renderAbout();
      default: return renderGeneral();
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">System Settings</h1>
          </div>
          <p className="text-sm text-outline mt-0.5">
            {dataUpdatedAt
              ? `Last updated ${formatTimestamp(new Date(dataUpdatedAt).toISOString())}`
              : 'Fetching live data...'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleReset}
            disabled={!hasChanges || mutation.isPending}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-container-high text-on-surface text-sm font-medium hover:bg-surface-container-highest transition-colors disabled:opacity-40 cursor-pointer"
          >
            <RotateCcw size={16} />
            Reset
          </button>
          <button
            onClick={handleSave}
            disabled={!hasChanges || mutation.isPending}
            className="flex items-center gap-2 px-5 py-2 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-40 cursor-pointer"
          >
            <Save size={16} className={mutation.isPending ? 'animate-pulse' : ''} />
            {mutation.isPending ? 'Saving...' : 'Save'}
          </button>
          <button
            onClick={() => refetch()}
            disabled={isRefetching}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface-container-high text-on-surface text-sm font-medium hover:bg-surface-container-highest transition-colors disabled:opacity-50 cursor-pointer"
          >
            <RefreshCw size={16} className={isRefetching ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {hasChanges && (
        <div className="px-4 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center gap-2">
          <span className="text-amber-400 text-xs font-medium">Unsaved changes. Save or reset to discard.</span>
        </div>
      )}

      <div className="flex gap-6 flex-col md:flex-row">
        <div className="md:w-48 shrink-0">
          <nav className="flex flex-row md:flex-col gap-1 overflow-x-auto md:overflow-x-visible">
            {TABS.map((tab) => (
              <button
                key={tab.key}
                onClick={() => handleTabClick(tab.key)}
                className={cn(
                  'flex items-center gap-2.5 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all whitespace-nowrap cursor-pointer',
                  activeTab === tab.key
                    ? 'bg-primary/10 text-primary'
                    : 'text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface',
                )}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="flex-1 min-w-0">
          <Card className="p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
                {TABS.find((t) => t.key === activeTab)?.icon}
              </div>
              <div>
                <h3 className="text-sm font-semibold text-on-surface">{TABS.find((t) => t.key === activeTab)?.label}</h3>
                <p className="text-xs text-outline">
                  {activeTab === 'general' && 'Federation strategy, model, and communication settings'}
                  {activeTab === 'training' && 'Training hyperparameters and optimization settings'}
                  {activeTab === 'federated' && 'Federated learning aggregation and client settings'}
                  {activeTab === 'prototype' && 'Prototype dimension and representation settings'}
                  {activeTab === 'knowledge_transfer' && 'Cross-modal knowledge transfer configuration'}
                  {activeTab === 'personalization' && 'Client personalization configuration'}
                  {activeTab === 'logging' && 'System logging and monitoring configuration'}
                  {activeTab === 'about' && 'System version and configuration summary'}
                </p>
              </div>
            </div>
            {renderTabContent()}
          </Card>
        </div>
      </div>

      {pendingTab && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
          <div className="bg-surface-container border border-outline-variant rounded-2xl shadow-2xl p-6 max-w-sm w-full mx-4">
            <h3 className="text-sm font-bold text-on-surface mb-2">Unsaved Changes</h3>
            <p className="text-sm text-outline mb-5">You have unsaved changes. Switching tabs will discard them. Continue?</p>
            <div className="flex items-center justify-end gap-3">
              <button onClick={cancelTabChange} className="px-4 py-2 rounded-xl bg-surface-container-high text-sm font-medium text-on-surface hover:bg-surface-container-highest transition-colors cursor-pointer">Cancel</button>
              <button onClick={confirmTabChange} className="px-4 py-2 rounded-xl bg-rose-500/10 text-rose-400 text-sm font-semibold hover:bg-rose-500/20 transition-colors cursor-pointer">Discard</button>
            </div>
          </div>
        </div>
      )}

      {toast && (
        <div className={cn(
          'fixed bottom-6 right-6 z-50 px-5 py-3 rounded-xl shadow-2xl text-sm font-medium transition-all animate-slide-in',
          toast.type === 'success' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border border-rose-500/20',
        )}>
          {toast.message}
        </div>
      )}
    </div>
  );
};
