import { useConnectionStatus } from './hooks';
import type { ConnectionStatus } from './types';

interface StatusIndicatorProps {
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

const STATUS_CONFIG: Record<ConnectionStatus, { color: string; pulse: boolean; label: string }> = {
  connected: { color: 'bg-emerald-400', pulse: true, label: 'Connected' },
  disconnected: { color: 'bg-rose-400', pulse: false, label: 'Disconnected' },
  reconnecting: { color: 'bg-amber-400', pulse: true, label: 'Reconnecting' },
  connecting: { color: 'bg-amber-400', pulse: true, label: 'Connecting' },
  polling: { color: 'bg-sky-400', pulse: true, label: 'Polling' },
  offline: { color: 'bg-rose-400', pulse: false, label: 'Offline' },
};

const sizeMap = { sm: 'w-2 h-2', md: 'w-2.5 h-2.5', lg: 'w-3 h-3' };

export const ConnectionStatusIndicator = ({ showLabel = true, size = 'sm' }: StatusIndicatorProps) => {
  const { connectionStatus, currentTransportLabel } = useConnectionStatus();
  const cfg = STATUS_CONFIG[connectionStatus];

  return (
    <div className="flex items-center gap-2" title={`${cfg.label} · ${currentTransportLabel}`}>
      <span className={`relative inline-flex ${sizeMap[size]}`}>
        <span className={`absolute inset-0 rounded-full ${cfg.color} ${cfg.pulse ? 'animate-ping opacity-40' : ''}`} />
        <span className={`relative inline-block rounded-full ${sizeMap[size]} ${cfg.color}`} />
      </span>
      {showLabel && (
        <span className="text-[10px] font-mono font-semibold uppercase tracking-wider text-outline">
          {cfg.label}
        </span>
      )}
    </div>
  );
};

export const TransportLabel = () => {
  const { currentTransportLabel } = useConnectionStatus();
  return (
    <span className="text-[10px] font-mono text-outline uppercase tracking-wider">
      {currentTransportLabel}
    </span>
  );
};
