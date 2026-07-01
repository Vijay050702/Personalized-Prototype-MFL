import React, { useEffect, useState } from 'react';
import { StatCard } from '../components/ui/StatCard';
import { Card } from '../components/ui/Card';
import { StatusBadge } from '../components/ui/StatusBadge';
import { getDashboardStats, getRecentActivity } from '../services/dashboardService';
import { SystemMetric, ActivityLog } from '../types';
import { BarChart3, Clock, AlertCircle, CheckCircle2, Info } from 'lucide-react';
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip,
  BarChart,
  Bar
} from 'recharts';

const chartData = [
  { time: '00:00', load: 32, latency: 120 },
  { time: '04:00', load: 45, latency: 110 },
  { time: '08:00', load: 68, latency: 145 },
  { time: '12:00', load: 82, latency: 160 },
  { time: '16:00', load: 74, latency: 155 },
  { time: '20:00', load: 52, latency: 130 },
  { time: '23:59', load: 38, latency: 115 },
];

export const Dashboard = () => {
  const [stats, setStats] = useState<SystemMetric[]>([]);
  const [activity, setActivity] = useState<ActivityLog[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [s, a] = await Promise.all([getDashboardStats(), getRecentActivity()]);
        setStats(s);
        setActivity(a);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <StatCard 
            key={stat.label} 
            label={stat.label} 
            value={stat.value} 
            change={stat.change} 
            trend={stat.trend} 
          />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2 p-6 h-[400px] flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-sm font-semibold text-on-surface">System Performance</h3>
              <p className="text-xs text-outline">Real-time aggregate load and latency metrics</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-primary" />
                <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Load</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-secondary" />
                <span className="text-[10px] uppercase font-bold tracking-wider text-outline">Latency</span>
              </div>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorLoad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--color-primary)" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="var(--color-primary)" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorLatency" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--color-secondary)" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="var(--color-secondary)" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis 
                  dataKey="time" 
                  axisLine={false} 
                  tickLine={false} 
                  tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
                  dy={10}
                />
                <YAxis 
                  axisLine={false} 
                  tickLine={false} 
                  tick={{ fill: 'var(--color-outline)', fontSize: 10 }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'var(--color-surface-container-highest)', 
                    borderColor: 'var(--color-outline-variant)',
                    borderRadius: '12px',
                    fontSize: '12px',
                    color: 'var(--color-on-surface)'
                  }}
                  itemStyle={{ color: 'var(--color-on-surface)' }}
                />
                <Area 
                  type="monotone" 
                  dataKey="load" 
                  stroke="var(--color-primary)" 
                  fillOpacity={1} 
                  fill="url(#colorLoad)" 
                  strokeWidth={2}
                />
                <Area 
                  type="monotone" 
                  dataKey="latency" 
                  stroke="var(--color-secondary)" 
                  fillOpacity={1} 
                  fill="url(#colorLatency)" 
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-sm font-semibold text-on-surface">Recent Activity</h3>
            <button className="text-[10px] uppercase font-bold tracking-widest text-primary hover:underline">View All</button>
          </div>
          <div className="space-y-6">
            {activity.map((log) => (
              <div key={log.id} className="flex gap-4 group">
                <div className="relative flex flex-col items-center">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                    log.type === 'success' ? 'bg-emerald-500/10 text-emerald-500' :
                    log.type === 'warning' ? 'bg-amber-500/10 text-amber-500' :
                    log.type === 'error' ? 'bg-rose-500/10 text-rose-500' :
                    'bg-primary/10 text-primary'
                  }`}>
                    {log.type === 'success' && <CheckCircle2 size={16} />}
                    {log.type === 'warning' && <AlertCircle size={16} />}
                    {log.type === 'error' && <AlertCircle size={16} />}
                    {log.type === 'info' && <Info size={16} />}
                  </div>
                  <div className="w-px flex-1 bg-outline-variant/30 my-2 group-last:hidden" />
                </div>
                <div className="flex-1 min-w-0 pt-0.5">
                  <p className="text-xs font-medium text-on-surface leading-normal">{log.message}</p>
                  <p className="text-[10px] text-outline mt-1 font-mono">{log.timestamp}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-primary">
              <BarChart3 size={20} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-on-surface">Privacy Score Distribution</h3>
              <p className="text-xs text-outline">Anonymization effectiveness across clusters</p>
            </div>
          </div>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={[
                { name: 'Cluster A', score: 98 },
                { name: 'Cluster B', score: 92 },
                { name: 'Cluster C', score: 85 },
                { name: 'Cluster D', score: 96 },
                { name: 'Cluster E', score: 89 },
              ]}>
                <XAxis dataKey="name" hide />
                <Tooltip 
                   contentStyle={{ 
                    backgroundColor: 'var(--color-surface-container-highest)', 
                    borderColor: 'var(--color-outline-variant)',
                    borderRadius: '12px',
                    fontSize: '12px'
                  }}
                />
                <Bar dataKey="score" fill="var(--color-primary)" radius={[4, 4, 0, 0]} barSize={32} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6 relative overflow-hidden group">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center text-secondary">
              <Clock size={20} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-on-surface">Uptime & Reliability</h3>
              <p className="text-xs text-outline">Current system stability index</p>
            </div>
          </div>
          <div className="flex items-baseline gap-2 mb-4">
            <span className="text-4xl font-display font-bold text-on-surface">99.98%</span>
            <span className="text-xs text-emerald-400 font-medium">+0.02% from last week</span>
          </div>
          <div className="flex gap-1 h-8 items-end">
            {Array.from({ length: 40 }).map((_, i) => (
              <div 
                key={i} 
                className={`flex-1 rounded-sm transition-all duration-500 hover:scale-y-110 cursor-help ${
                  i === 15 ? 'bg-amber-400/50' : i === 32 ? 'bg-rose-400/50' : 'bg-secondary/40'
                }`}
                style={{ height: `${Math.random() * 60 + 40}%` }}
                title={`Day ${i}: ${i === 15 ? 'Warning' : i === 32 ? 'Interrupted' : 'Normal'}`}
              />
            ))}
          </div>
          <p className="text-[10px] text-outline mt-4 uppercase tracking-widest font-bold">Past 40 Training Cycles</p>
        </Card>
      </div>
    </div>
  );
};
