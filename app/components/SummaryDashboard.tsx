import React, { useMemo } from 'react';
import {
    X, TrendingDown, TrendingUp, Cpu, Activity,
    CheckCircle2, AlertCircle, RefreshCw, Layers,
} from 'lucide-react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer,
} from 'recharts';
import { NetworkNode, NetworkConnection, SimulationState } from './types';

// ── Param estimation (same logic as ExportPanel, kept local to avoid coupling)
function estimateLayerParams(node: NetworkNode, prevNode: NetworkNode | undefined): number {
    switch (node.type) {
        case 'dense': {
            const inp = (prevNode?.config?.neurons ?? prevNode?.config?.units ?? Number(prevNode?.config?.inputShape ?? 0)) || 128;
            return inp * (node.config.neurons ?? 128) + (node.config.neurons ?? 128);
        }
        case 'conv2d': {
            const k = node.config.kernelSize ?? 3;
            return k * k * 1 * (node.config.filters ?? 32) + (node.config.filters ?? 32);
        }
        case 'lstm': {
            const u = node.config.units ?? 64;
            return 4 * ((128 + u) * u + u);
        }
        case 'embedding':
            return (node.config.vocabSize ?? 10000) * (node.config.embedDim ?? 64);
        default:
            return 0;
    }
}

function formatParams(p: number): string {
    if (p >= 1_000_000) return `${(p / 1_000_000).toFixed(2)}M`;
    if (p >= 1_000) return `${(p / 1_000).toFixed(1)}K`;
    return `${p}`;
}

// ── Gauge component ────────────────────────────────────────────────────────
function Gauge({ value, max, label, color, icon, format }:
    { value: number; max: number; label: string; color: string; icon: React.ReactNode; format: (v: number) => string }) {
    const pct = Math.min(1, value / max);
    const r = 30;
    const circ = 2 * Math.PI * r;
    const dash = circ * pct;

    return (
        <div className="flex flex-col items-center gap-1">
            <div className="relative w-[72px] h-[72px]">
                <svg viewBox="0 0 72 72" className="w-full h-full -rotate-90">
                    <circle cx="36" cy="36" r={r} fill="none" stroke="currentColor" strokeWidth="5"
                        className="opacity-10" style={{ color }} />
                    <circle cx="36" cy="36" r={r} fill="none" stroke={color} strokeWidth="5"
                        strokeDasharray={`${dash} ${circ - dash}`} strokeLinecap="round"
                        style={{ transition: 'stroke-dasharray 0.6s ease' }} />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <div className="text-[7px] mb-0.5" style={{ color }}>{icon}</div>
                    <div className="text-[14px] font-bold tabular-nums leading-none" style={{ color }}>
                        {format(value)}
                    </div>
                </div>
            </div>
            <div className="text-[9.5px] font-semibold text-slate-500 dark:text-slate-400">{label}</div>
        </div>
    );
}

// ── Mini sparkline tooltip ─────────────────────────────────────────────────
function MiniTooltip({ active, payload, label }: { active?: boolean; payload?: { value: number; name: string; color: string }[]; label?: number }) {
    if (!active || !payload?.length) return null;
    return (
        <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1.5 shadow-md text-[9px]">
            <div className="text-slate-400 mb-0.5">Epoch {label}</div>
            {payload.map(p => (
                <div key={p.name} className="flex items-center gap-1">
                    <div className="w-1.5 h-1.5 rounded-full" style={{ background: p.color }} />
                    <span className="capitalize text-slate-500">{p.name}:</span>
                    <span className="font-mono font-bold" style={{ color: p.color }}>
                        {p.name === 'accuracy' ? `${(p.value * 100).toFixed(1)}%` : p.value.toFixed(3)}
                    </span>
                </div>
            ))}
        </div>
    );
}

// ── Node type badge colors ─────────────────────────────────────────────────
const TYPE_BADGE: Record<string, { bg: string; text: string }> = {
    input: { bg: '#EFF6FF', text: '#1D4ED8' },
    dense: { bg: '#EEF2FF', text: '#4338CA' },
    activation: { bg: '#F5F3FF', text: '#6D28D9' },
    dropout: { bg: '#FFF7ED', text: '#C2410C' },
    output: { bg: '#F0FDFA', text: '#0F766E' },
    conv2d: { bg: '#ECFEFF', text: '#0E7490' },
    flatten: { bg: '#F8FAFC', text: '#475569' },
    lstm: { bg: '#FDF2F8', text: '#9D174D' },
    batchnorm: { bg: '#F7FEE7', text: '#3F6212' },
    embedding: { bg: '#FFFBEB', text: '#92400E' },
};

// ── Props ──────────────────────────────────────────────────────────────────
interface SummaryDashboardProps {
    isOpen: boolean;
    onClose: () => void;
    nodes: NetworkNode[];
    connections: NetworkConnection[];
    simulation: SimulationState;
    isDark?: boolean;
}

// ── Component ──────────────────────────────────────────────────────────────
export function SummaryDashboard({
    isOpen, onClose, nodes, connections, simulation, isDark,
}: SummaryDashboardProps) {
    const sorted = useMemo(() => [...nodes].sort((a, b) => a.x - b.x), [nodes]);

    const layerParams = useMemo(() =>
        sorted.map((n, i) => ({
            node: n,
            params: estimateLayerParams(n, sorted[i - 1]),
        })), [sorted]);

    const totalParams = useMemo(() => layerParams.reduce((s, l) => s + l.params, 0), [layerParams]);
    const hasHistory = simulation.history.length > 1;
    const isValid = nodes.some(n => n.type === 'input') && nodes.some(n => n.type === 'output') && connections.length > 0;

    if (!isOpen) return null;

    const bg = isDark ? '#1E293B' : '#FFFFFF';
    const border = isDark ? '#334155' : '#E2E8F0';
    const text = isDark ? '#E2E8F0' : '#1E293B';
    const sub = isDark ? '#94A3B8' : '#64748B';
    const card = isDark ? '#0F172A' : '#F8FAFC';
    const cardBorder = isDark ? '#1E293B' : '#E2E8F0';

    return (
        <div
            role="dialog" aria-label="Summary Dashboard" aria-modal="true"
            className="absolute top-4 left-4 z-[45] nf-animate-in w-[300px] max-h-[calc(100vh-140px)] flex flex-col rounded-2xl overflow-hidden"
            style={{
                background: bg, border: `1px solid ${border}`,
                boxShadow: '0 24px 64px rgba(0,0,0,0.16), 0 4px 16px rgba(0,0,0,0.08)',
            }}
        >
            {/* Header */}
            <div className="flex items-center gap-2.5 px-4 py-3 flex-shrink-0"
                style={{ background: 'linear-gradient(135deg, #0F766E, #0D9488)', borderBottom: '1px solid rgba(255,255,255,0.12)' }}>
                <div className="w-7 h-7 rounded-lg bg-white/20 flex items-center justify-center flex-shrink-0 shadow-inner">
                    <Activity size={14} className="text-white" />
                </div>
                <div className="flex-1">
                    <div className="text-[13px] font-bold text-white">Summary Dashboard</div>
                    <div className="text-[10px] text-teal-200">Live metrics · Architecture overview</div>
                </div>
                <button onClick={onClose} aria-label="Close summary dashboard"
                    className="w-6 h-6 rounded-lg bg-white/15 hover:bg-white/25 flex items-center justify-center text-white/80 hover:text-white transition-all">
                    <X size={12} />
                </button>
            </div>

            <div className="flex-1 overflow-y-auto">

                {/* ── Gauges ─────────────────────────────────────────────────────── */}
                <div className="px-4 pt-4 pb-3">
                    <div className="flex items-center justify-around">
                        <Gauge
                            value={simulation.accuracy}
                            max={1}
                            label="Accuracy"
                            color="#10B981"
                            icon={<TrendingUp size={9} />}
                            format={v => `${(v * 100).toFixed(0)}%`}
                        />
                        <div className="h-16 w-px" style={{ background: border }} />
                        <Gauge
                            value={Math.max(0, 1 - simulation.loss)}
                            max={1}
                            label="Loss Score"
                            color="#EF4444"
                            icon={<TrendingDown size={9} />}
                            format={() => simulation.loss.toFixed(3)}
                        />
                        <div className="h-16 w-px" style={{ background: border }} />
                        <div className="flex flex-col items-center gap-1">
                            <div className="w-[72px] h-[72px] rounded-full flex items-center justify-center"
                                style={{ background: isDark ? '#1E3A5F' : '#EFF6FF', border: `2px solid ${isDark ? '#2563EB44' : '#BFDBFE'}` }}>
                                <div className="text-center">
                                    <Cpu size={13} className="text-blue-500 mx-auto mb-0.5" />
                                    <div className="text-[12px] font-bold tabular-nums text-blue-600 dark:text-blue-400">{formatParams(totalParams)}</div>
                                </div>
                            </div>
                            <div className="text-[9.5px] font-semibold" style={{ color: sub }}>Params</div>
                        </div>
                    </div>
                </div>

                {/* Status strip */}
                <div className="px-4 mb-3">
                    <div className="flex items-center gap-2 rounded-xl px-3 py-2"
                        style={{ background: isValid ? (isDark ? '#052e16' : '#F0FDF4') : (isDark ? '#2D1717' : '#FEF2F2'), border: `1px solid ${isValid ? (isDark ? '#16a34a44' : '#BBF7D0') : (isDark ? '#dc262644' : '#FECACA')}` }}>
                        {isValid
                            ? <CheckCircle2 size={12} className="text-emerald-500 flex-shrink-0" />
                            : <AlertCircle size={12} className="text-red-500 flex-shrink-0" />}
                        <span className="text-[10.5px] font-medium" style={{ color: isValid ? '#15803D' : '#DC2626' }}>
                            {isValid ? 'Network valid · TF export ready' : 'Missing input/output or connections'}
                        </span>
                    </div>
                </div>

                {/* Live metrics text row */}
                <div className="px-4 mb-3 grid grid-cols-3 gap-2">
                    {[
                        { label: 'Epoch', value: `${simulation.epoch} / ${simulation.totalEpochs}`, color: text },
                        { label: 'Loss', value: simulation.loss.toFixed(4), color: '#EF4444' },
                        { label: 'Accuracy', value: `${(simulation.accuracy * 100).toFixed(1)}%`, color: '#10B981' },
                    ].map(m => (
                        <div key={m.label} className="rounded-xl p-2 text-center" style={{ background: card, border: `1px solid ${cardBorder}` }}>
                            <div className="text-[9px] mb-0.5" style={{ color: sub }}>{m.label}</div>
                            <div className="text-[12px] font-bold tabular-nums font-mono" style={{ color: m.color }}>{m.value}</div>
                        </div>
                    ))}
                </div>

                {/* ── History chart ─────────────────────────────────────────────── */}
                {hasHistory && (
                    <div className="px-4 mb-3">
                        <div className="text-[10px] font-semibold uppercase tracking-wider mb-2" style={{ color: sub }}>
                            Training Curves
                        </div>
                        <div className="rounded-xl overflow-hidden p-2" style={{ background: card, border: `1px solid ${cardBorder}` }}>
                            <ResponsiveContainer width="100%" height={100}>
                                <LineChart data={simulation.history} margin={{ top: 4, right: 4, bottom: 4, left: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1E293B' : '#E2E8F0'} />
                                    <XAxis dataKey="epoch" tick={{ fontSize: 8, fill: sub }} tickLine={false} axisLine={false} />
                                    <YAxis yAxisId="l" domain={[0, 1]} tick={{ fontSize: 8, fill: sub }} tickLine={false} axisLine={false} width={22} />
                                    <Tooltip content={<MiniTooltip />} />
                                    <Line yAxisId="l" type="monotone" dataKey="loss" stroke="#EF4444" strokeWidth={1.5} dot={false} name="loss" isAnimationActive={false} />
                                    <Line yAxisId="l" type="monotone" dataKey="accuracy" stroke="#10B981" strokeWidth={1.5} dot={false} name="accuracy" isAnimationActive={false} />
                                </LineChart>
                            </ResponsiveContainer>
                            <div className="flex items-center justify-center gap-4 mt-1">
                                <div className="flex items-center gap-1"><div className="w-3 h-0.5 bg-red-400 rounded" /><span className="text-[8px]" style={{ color: sub }}>Loss ↓</span></div>
                                <div className="flex items-center gap-1"><div className="w-3 h-0.5 bg-emerald-400 rounded" /><span className="text-[8px]" style={{ color: sub }}>Accuracy ↑</span></div>
                            </div>
                        </div>
                    </div>
                )}

                {/* ── Architecture overview ──────────────────────────────────────── */}
                <div className="px-4 pb-4">
                    <div className="flex items-center gap-1.5 mb-2">
                        <Layers size={10} style={{ color: sub }} />
                        <div className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: sub }}>
                            Architecture
                        </div>
                        <div className="ml-auto text-[9.5px] tabular-nums" style={{ color: sub }}>{nodes.length} layers</div>
                    </div>
                    <div className="space-y-1">
                        {sorted.map((n, i) => {
                            const badge = TYPE_BADGE[n.type] ?? TYPE_BADGE.input;
                            const params = layerParams[i]?.params ?? 0;
                            return (
                                <div key={n.id} className="flex items-center gap-2 rounded-lg px-2.5 py-1.5"
                                    style={{ background: card, border: `1px solid ${cardBorder}` }}>
                                    <div className="text-[8.5px] font-mono w-3.5 text-center" style={{ color: sub }}>{i + 1}</div>
                                    <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded-md flex-shrink-0"
                                        style={{ background: isDark ? badge.text + '22' : badge.bg, color: isDark ? badge.text : badge.text }}>
                                        {n.type}
                                    </span>
                                    <div className="flex-1 min-w-0">
                                        <div className="text-[10px] font-medium truncate" style={{ color: text }}>{n.label}</div>
                                    </div>
                                    {params > 0 && (
                                        <div className="text-[8.5px] font-mono flex-shrink-0" style={{ color: sub }}>
                                            {formatParams(params)}
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                        {!nodes.length && (
                            <div className="text-center py-4" style={{ color: sub }}>
                                <RefreshCw size={18} className="mx-auto mb-1 opacity-30" />
                                <p className="text-[11px]">No layers yet</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
