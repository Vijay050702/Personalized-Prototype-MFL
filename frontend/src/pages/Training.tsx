import React, { useEffect, useState } from 'react';
import { getTrainingHistory, getConvergenceData } from '../services/trainingService';
import { TrainingRound } from '../types';
import { Card } from '../components/ui/Card';
import { StatusBadge } from '../components/ui/StatusBadge';
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip,
  Legend
} from 'recharts';
import { Play, RotateCcw, Activity, Target, Zap } from 'lucide-react';

export const Training = () => {
  const [history, setHistory] = useState<TrainingRound[]>([]);
  const [convergence, setConvergence] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getTrainingHistory(), getConvergenceData()]).then(([h, c]) => {
      setHistory(h);
      setConvergence(c);
      setLoading(false);
    });
  }, []);

  if (loading) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">Model Optimization Engine</h1>
          <p className="text-sm text-outline">Distributed SGD convergence and global model versioning</p>
        </div>
        <div className="flex items-center gap-3">
          <button className="px-4 py-2 border border-outline-variant rounded-lg text-sm font-medium hover:bg-surface-container-high transition-colors flex items-center gap-2">
            <RotateCcw size={16} />
            Rollback
          </button>
          <button className="px-6 py-2 bg-primary text-on-primary rounded-lg text-sm font-bold flex items-center gap-2 hover:opacity-90 transition-opacity shadow-lg shadow-primary/20">
            <Play size={16} fill="currentColor" />
            Start Next Round
          </button>
        </div>
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
                <p className="text-xs text-outline">Accuracy vs. Loss over global rounds</p>
              </div>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={convergence}>
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
                />
                <YAxis 
                  yAxisId="right"
                  orientation="right"
                  axisLine={false} 
                  tickLine={false} 
                  tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
                  label={{ value: 'Loss Index', angle: 90, position: 'insideRight', fill: 'var(--color-outline)', fontSize: 10 }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'var(--color-surface-container-highest)', 
                    borderColor: 'var(--color-outline-variant)',
                    borderRadius: '12px',
                    fontSize: '12px'
                  }}
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
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="loss" 
                  stroke="var(--color-secondary)" 
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <div className="space-y-4">
           <Card className="p-6 bg-surface-container-lowest">
             <div className="flex items-center justify-between mb-4">
               <span className="text-[10px] uppercase font-bold tracking-[0.2em] text-outline">Current Model</span>
               <div className="flex items-center gap-1.5">
                 <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                 <span className="text-[10px] text-emerald-400 font-bold uppercase">Stable</span>
               </div>
             </div>
             <h4 className="text-xl font-display font-bold text-on-surface mb-1">EfficientNet-B0 v2.1</h4>
             <p className="text-xs text-outline mb-6">Quantized INT8 • PyTorch 2.2</p>
             
             <div className="space-y-4">
               <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                 <span className="text-xs text-outline">Global Accuracy</span>
                 <span className="text-sm font-mono font-bold text-on-surface">94.21%</span>
               </div>
               <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                 <span className="text-xs text-outline">Avg. Loss</span>
                 <span className="text-sm font-mono font-bold text-on-surface">0.0452</span>
               </div>
               <div className="flex items-center justify-between py-2 border-b border-outline-variant/30">
                 <span className="text-xs text-outline">Model Size</span>
                 <span className="text-sm font-mono font-bold text-on-surface">4.8 MB</span>
               </div>
               <div className="flex items-center justify-between py-2">
                 <span className="text-xs text-outline">Consolidation Strategy</span>
                 <span className="text-sm font-mono font-bold text-on-surface">FedAvg</span>
               </div>
             </div>
           </Card>

           <Card className="p-5 bg-primary/5 border-primary/20">
             <div className="flex items-center gap-3 mb-3">
               <Zap size={18} className="text-primary" />
               <h5 className="text-sm font-bold text-on-surface">Auto-Hyperparameter Tuning</h5>
             </div>
             <p className="text-[11px] text-on-surface-variant leading-relaxed">
               System is currently exploring optimal learning rates using Bayesian optimization across edge nodes.
             </p>
             <div className="mt-4 flex items-center justify-between">
                <span className="text-[10px] font-bold text-primary">Active Search...</span>
                <div className="flex gap-0.5">
                   {Array.from({ length: 12 }).map((_, i) => (
                     <div key={i} className="w-1 h-3 bg-primary/20 rounded-full animate-bounce" style={{ animationDelay: `${i * 0.1}s` }} />
                   ))}
                </div>
             </div>
           </Card>
        </div>
      </div>

      <Card className="overflow-hidden">
        <div className="px-6 py-4 border-b border-outline-variant flex items-center justify-between bg-surface-container-low/50">
          <h3 className="text-sm font-bold text-on-surface">Round Execution History</h3>
          <button className="text-[10px] uppercase font-bold tracking-widest text-outline hover:text-on-surface">Export CSV</button>
        </div>
        <div className="overflow-x-auto">
           <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-surface-container-low/30 text-[10px] uppercase tracking-[0.15em] font-bold text-outline">
                  <th className="px-6 py-4 font-bold border-b border-outline-variant">Round #</th>
                  <th className="px-6 py-4 font-bold border-b border-outline-variant">Status</th>
                  <th className="px-6 py-4 font-bold border-b border-outline-variant">Start Time</th>
                  <th className="px-6 py-4 font-bold border-b border-outline-variant">Duration</th>
                  <th className="px-6 py-4 font-bold border-b border-outline-variant">Nodes</th>
                  <th className="px-6 py-4 font-bold border-b border-outline-variant text-right">Accuracy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/30 text-xs">
                {history.map((round) => (
                  <tr key={round.id} className="hover:bg-surface-container-high/20 transition-colors">
                    <td className="px-6 py-4 font-mono font-bold text-on-surface">R-{round.round}</td>
                    <td className="px-6 py-4"><StatusBadge status={round.status} /></td>
                    <td className="px-6 py-4 text-outline">{round.startTime}</td>
                    <td className="px-6 py-4 text-outline">{round.duration}</td>
                    <td className="px-6 py-4 font-mono text-on-surface-variant">{round.participants}</td>
                    <td className="px-6 py-4 text-right font-mono font-bold text-primary">{(round.accuracy * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
           </table>
        </div>
      </Card>
    </div>
  );
};
