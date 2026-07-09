import { ShieldOff } from 'lucide-react';
import { Card } from '../ui/Card';

export const AuthBanner = () => (
  <Card className="mb-6 p-4 border-amber-500/20 bg-amber-500/5" variant="outline">
    <div className="flex items-center gap-3">
      <div className="w-9 h-9 rounded-xl bg-amber-500/10 flex items-center justify-center shrink-0">
        <ShieldOff size={18} className="text-amber-400" />
      </div>
      <div>
        <p className="text-sm font-semibold text-amber-400">Authentication Disabled</p>
        <p className="text-xs text-outline mt-0.5">
          Authentication is not enabled by the backend. All features are accessible without login.
        </p>
      </div>
    </div>
  </Card>
);
