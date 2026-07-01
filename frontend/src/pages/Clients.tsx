import React, { useEffect, useState } from 'react';
import { getClients } from '../services/clientService';
import { Client } from '../types';
import { Card } from '../components/ui/Card';
import { StatusBadge } from '../components/ui/StatusBadge';
import { Search, Filter, MoreVertical, MapPin, Cpu, Smartphone } from 'lucide-react';

export const Clients = () => {
  const [clients, setClients] = useState<Client[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getClients().then(data => {
      setClients(data);
      setLoading(false);
    });
  }, []);

  if (loading) return null;

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-on-surface tracking-tight">Active Node Fleet</h1>
          <p className="text-sm text-outline">Manage and monitor distributed edge intelligence nodes</p>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-4 py-2 bg-surface-container-high border border-outline-variant rounded-lg text-sm font-medium text-on-surface hover:bg-surface-container-highest transition-colors flex items-center gap-2">
            <Filter size={16} />
            Filter
          </button>
          <button className="px-4 py-2 bg-primary text-on-primary rounded-lg text-sm font-bold hover:opacity-90 transition-opacity">
            Provision New Node
          </button>
        </div>
      </div>

      <Card className="overflow-hidden">
        <div className="p-4 border-b border-outline-variant flex items-center justify-between bg-surface-container-low/50">
          <div className="relative">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-outline" />
            <input 
              type="text" 
              placeholder="Search by ID, location, or hardware..." 
              className="pl-10 pr-4 py-1.5 bg-surface-container-lowest border border-outline-variant rounded-lg text-sm w-80 focus:outline-none focus:border-primary transition-all"
            />
          </div>
          <div className="text-xs text-outline font-medium">
            Showing {clients.length} of 1,284 nodes
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-surface-container-low/30 text-[10px] uppercase tracking-[0.15em] font-bold text-outline">
                <th className="px-6 py-4 font-bold border-b border-outline-variant">Node Identity</th>
                <th className="px-6 py-4 font-bold border-b border-outline-variant">Status</th>
                <th className="px-6 py-4 font-bold border-b border-outline-variant">Hardware & OS</th>
                <th className="px-6 py-4 font-bold border-b border-outline-variant">Location</th>
                <th className="px-6 py-4 font-bold border-b border-outline-variant text-right">Model Accuracy</th>
                <th className="px-6 py-4 font-bold border-b border-outline-variant w-10"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/30">
              {clients.map((client) => (
                <tr key={client.id} className="hover:bg-surface-container-high/20 transition-colors group">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-surface-container-highest flex items-center justify-center text-outline group-hover:text-primary transition-colors">
                        {client.deviceInfo.includes('iPhone') ? <Smartphone size={16} /> : <Cpu size={16} />}
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-on-surface">{client.name}</p>
                        <p className="text-[10px] font-mono text-outline uppercase">{client.id}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex flex-col gap-1">
                      <StatusBadge status={client.status} className="w-fit" />
                      <span className="text-[10px] text-outline italic">Seen: {client.lastSeen}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <p className="text-xs text-on-surface-variant font-medium">{client.deviceInfo}</p>
                    <div className="w-32 h-1 bg-surface-container-highest rounded-full mt-2 overflow-hidden">
                      <div 
                        className="h-full bg-primary rounded-full transition-all duration-1000" 
                        style={{ width: `${client.progress}%` }} 
                      />
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-1.5 text-xs text-outline">
                      <MapPin size={12} />
                      <span>{client.location}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span className="font-mono text-sm font-bold text-primary">{(client.accuracy * 100).toFixed(1)}%</span>
                  </td>
                  <td className="px-6 py-4">
                    <button className="p-1 text-outline hover:text-on-surface transition-colors">
                      <MoreVertical size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="p-4 border-t border-outline-variant bg-surface-container-low/20 flex items-center justify-between">
          <div className="flex items-center gap-1">
             <button className="p-2 text-outline hover:text-on-surface disabled:opacity-30" disabled>Previous</button>
             <button className="w-8 h-8 rounded-lg bg-primary text-on-primary text-xs font-bold">1</button>
             <button className="w-8 h-8 rounded-lg text-outline hover:bg-surface-container-high text-xs font-bold">2</button>
             <button className="w-8 h-8 rounded-lg text-outline hover:bg-surface-container-high text-xs font-bold">3</button>
             <span className="text-outline mx-1">...</span>
             <button className="p-2 text-outline hover:text-on-surface">Next</button>
          </div>
          <p className="text-xs text-outline">Page 1 of 12</p>
        </div>
      </Card>
    </div>
  );
};
