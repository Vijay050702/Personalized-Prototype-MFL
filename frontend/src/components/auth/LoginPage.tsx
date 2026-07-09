import { ShieldOff } from 'lucide-react';
import { Card } from '../ui/Card';

export const LoginPage = () => (
  <div className="flex items-center justify-center min-h-[80vh]">
    <Card className="p-10 max-w-md w-full text-center">
      <div className="w-16 h-16 rounded-2xl bg-amber-500/10 flex items-center justify-center mx-auto mb-5">
        <ShieldOff size={32} className="text-amber-400" />
      </div>
      <h2 className="text-xl font-display font-bold text-on-surface mb-2">Authentication Unavailable</h2>
      <p className="text-sm text-outline leading-relaxed">
        Authentication is not enabled by the backend. A login page will be available once
        the backend authentication service is configured and deployed.
      </p>
      <div className="mt-6 p-4 rounded-xl bg-surface-container-high">
        <p className="text-xs text-outline font-mono">
          Status: Backend auth endpoints not detected
        </p>
      </div>
    </Card>
  </div>
);
