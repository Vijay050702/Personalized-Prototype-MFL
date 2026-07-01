import React, { useEffect, useState } from 'react';
import { getDatasets } from '../services/datasetService';
import { Dataset } from '../types';
import { Card } from '../components/ui/Card';
import { Database, ShieldCheck, FileText, Globe, Plus } from 'lucide-react';

export const Datasets = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDatasets().then(data => {
      setDatasets(data);
      setLoading(false);
    });
  }, []);

  if (loading) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">Distributed Data Inventory</h1>
          <p className="text-sm text-outline">Manage partitioned data sources across the federated network</p>
        </div>
        <button className="px-4 py-2 bg-primary text-on-primary rounded-lg text-sm font-bold flex items-center gap-2 hover:opacity-90 transition-opacity">
          <Plus size={18} />
          Register Data Source
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {datasets.map((ds) => (
          <Card key={ds.id} className="p-6 group cursor-pointer hover:border-primary/50 transition-all">
            <div className="flex items-start justify-between mb-4">
              <div className="w-12 h-12 rounded-2xl bg-surface-container-high flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-on-primary transition-colors">
                <Database size={24} />
              </div>
              <div className={`px-2 py-1 rounded-md text-[10px] font-bold uppercase tracking-widest border ${
                ds.privacyLevel === 'high' ? 'border-emerald-500/30 text-emerald-400 bg-emerald-400/5' : 
                ds.privacyLevel === 'medium' ? 'border-amber-500/30 text-amber-400 bg-amber-400/5' : 
                'border-outline-variant text-outline'
              }`}>
                {ds.privacyLevel} Privacy
              </div>
            </div>
            
            <h3 className="text-lg font-bold text-on-surface mb-1">{ds.name}</h3>
            <p className="text-xs text-outline mb-6">Last consolidated on {ds.lastUpdated}</p>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-surface-container-low rounded-xl border border-outline-variant/30">
                <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Total Samples</p>
                <p className="text-xl font-display font-bold text-on-surface">{(ds.samples / 1000000).toFixed(1)}M</p>
              </div>
              <div className="p-3 bg-surface-container-low rounded-xl border border-outline-variant/30">
                <p className="text-[10px] uppercase font-bold tracking-wider text-outline mb-1">Feature Vector</p>
                <p className="text-xl font-display font-bold text-on-surface">{ds.features}</p>
              </div>
            </div>

            <div className="mt-6 pt-6 border-t border-outline-variant/50 flex items-center justify-between text-xs text-outline font-medium">
              <div className="flex items-center gap-2">
                <Globe size={14} />
                <span>Source: {ds.source}</span>
              </div>
              <div className="flex items-center gap-4">
                 <div className="flex items-center gap-1">
                   <ShieldCheck size={14} className="text-emerald-400" />
                   <span>DP Verified</span>
                 </div>
                 <div className="flex items-center gap-1">
                   <FileText size={14} />
                   <span>Schema V2</span>
                 </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      <Card className="p-8 bg-gradient-to-br from-surface-container to-surface-container-low flex flex-col md:flex-row items-center gap-8 border-primary/20">
        <div className="flex-1">
          <h2 className="text-xl font-bold text-on-surface mb-2">Secure Multi-Party Computation (SMPC)</h2>
          <p className="text-sm text-on-surface-variant max-w-xl">
            Enable advanced cryptographic protocols to allow nodes to perform joint training without exposing raw underlying data features. Our SMPC layer ensures zero-knowledge proof verification at every round.
          </p>
          <div className="mt-6 flex gap-4">
             <button className="px-4 py-2 border border-outline-variant rounded-lg text-xs font-bold hover:bg-surface-container-high transition-colors">Configure Protocols</button>
             <button className="px-4 py-2 text-primary text-xs font-bold hover:underline">View Compliance Logs</button>
          </div>
        </div>
        <div className="w-48 h-48 rounded-full border-4 border-dashed border-outline-variant/30 flex items-center justify-center relative animate-[spin_20s_linear_infinite]">
          <ShieldCheck size={64} className="text-primary/20 absolute rotate-0" />
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-32 h-32 rounded-full bg-primary/10 blur-2xl" />
          </div>
        </div>
      </Card>
    </div>
  );
};
