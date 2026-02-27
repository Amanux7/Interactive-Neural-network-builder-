import React, { useState, useMemo } from 'react';
import {
  Download, X, Copy, CheckCheck, FileJson,
  Layers, Brain, Code2, Info, ExternalLink,
} from 'lucide-react';
import { NetworkNode, NetworkConnection } from './types';
import { SimulationState } from './types';

// ── TF.js layer descriptor builder ──────────────────────────────────────────
function toTFLayer(node: NetworkNode): object | null {
  switch (node.type) {
    case 'dense':
      return {
        class_name: 'Dense',
        name: node.label.toLowerCase().replace(/\s+/g, '_'),
        config: {
          units: node.config.neurons ?? 128,
          activation: 'linear',
          use_bias: true,
        },
      };
    case 'activation':
      return {
        class_name: 'Activation',
        name: node.label.toLowerCase().replace(/\s+/g, '_'),
        config: { activation: (node.config.activationFn ?? 'relu').toLowerCase() },
      };
    case 'dropout':
      return {
        class_name: 'Dropout',
        name: node.label.toLowerCase().replace(/\s+/g, '_'),
        config: { rate: node.config.dropoutRate ?? 0.5 },
      };
    case 'conv2d':
      return {
        class_name: 'Conv2D',
        name: node.label.toLowerCase().replace(/\s+/g, '_'),
        config: {
          filters: node.config.filters ?? 32,
          kernel_size: [node.config.kernelSize ?? 3, node.config.kernelSize ?? 3],
          activation: 'relu',
          padding: 'same',
        },
      };
    case 'flatten':
      return { class_name: 'Flatten', name: 'flatten', config: {} };
    case 'batchnorm':
      return { class_name: 'BatchNormalization', name: node.label.toLowerCase().replace(/\s+/g, '_'), config: {} };
    case 'lstm':
      return {
        class_name: 'LSTM',
        name: node.label.toLowerCase().replace(/\s+/g, '_'),
        config: { units: node.config.units ?? 64, return_sequences: false },
      };
    case 'embedding':
      return {
        class_name: 'Embedding',
        name: node.label.toLowerCase().replace(/\s+/g, '_'),
        config: {
          input_dim: node.config.vocabSize ?? 10000,
          output_dim: node.config.embedDim ?? 64,
        },
      };
    case 'input':
    case 'output':
    default:
      return null;
  }
}

// ── Parameter count estimation ───────────────────────────────────────────────
function estimateParams(nodes: NetworkNode[]): number {
  let total = 0;
  const sorted = [...nodes].sort((a, b) => a.x - b.x);
  for (let i = 0; i < sorted.length; i++) {
    const n = sorted[i];
    const prevN = sorted[i - 1];
    if (n.type === 'dense') {
      const inp = prevN?.config?.neurons ?? prevN?.config?.units ?? Number(prevN?.config?.inputShape) ?? 128;
      total += (inp || 128) * (n.config.neurons ?? 128) + (n.config.neurons ?? 128);
    } else if (n.type === 'conv2d') {
      const cin = 1;
      total += (n.config.kernelSize ?? 3) ** 2 * cin * (n.config.filters ?? 32) + (n.config.filters ?? 32);
    } else if (n.type === 'lstm') {
      const units = n.config.units ?? 64;
      total += 4 * ((128 + units) * units + units);
    } else if (n.type === 'embedding') {
      total += (n.config.vocabSize ?? 10000) * (n.config.embedDim ?? 64);
    }
  }
  return total;
}

function formatParams(p: number): string {
  if (p >= 1_000_000) return `${(p / 1_000_000).toFixed(1)}M`;
  if (p >= 1_000) return `${(p / 1_000).toFixed(1)}K`;
  return `${p}`;
}

// ── Props ────────────────────────────────────────────────────────────────────
interface ExportPanelProps {
  isOpen: boolean;
  onClose: () => void;
  nodes: NetworkNode[];
  connections: NetworkConnection[];
  simulation: SimulationState;
  isDark?: boolean;
}

// ── Component ────────────────────────────────────────────────────────────────
export function ExportPanel({ isOpen, onClose, nodes, connections, simulation, isDark }: ExportPanelProps) {
  const [copied, setCopied] = useState(false);
  const [modelName, setModelName] = useState('my_neural_net');

  const exportData = useMemo(() => {
    const tfLayers = nodes
      .filter(n => n.type !== 'input' && n.type !== 'output')
      .map(n => toTFLayer(n))
      .filter(Boolean);

    const inputNode = nodes.find(n => n.type === 'input');
    const outputNode = nodes.find(n => n.type === 'output');

    return {
      format: 'neuralforge-v2',
      name: modelName,
      exported_at: new Date().toISOString(),
      meta: {
        tool: 'NeuralForge — Visual Network Builder',
        research_note: 'Export compatible with TensorFlow.js tf.Sequential model reconstruction.',
        total_params: estimateParams(nodes),
      },
      architecture: {
        input_shape: inputNode?.config?.inputShape ?? 'unknown',
        output_shape: outputNode?.config?.outputShape ?? 'unknown',
        nodes: nodes.map(n => ({ id: n.id, type: n.type, label: n.label, config: n.config })),
        connections: connections.map(c => ({
          id: c.id, from: c.fromId, to: c.toId,
          weight: c.weight ?? 0.5, active: c.active !== false,
        })),
      },
      training_history: simulation.history.length > 0 ? {
        epochs_run: simulation.epoch,
        final_loss: simulation.loss,
        final_accuracy: simulation.accuracy,
        history: simulation.history,
      } : null,
      tensorflowjs: {
        comment: 'Use tfjs.layers.* to reconstruct. Input layer and output layer configs are in architecture.input_shape / output_shape.',
        layers: tfLayers,
        reconstruction_snippet: `// TensorFlow.js reconstruction example:\n// const model = tf.sequential();\n// layers.forEach(l => model.add(tf.layers[camelCase(l.class_name)](l.config)));\n// model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });`,
      },
    };
  }, [nodes, connections, simulation, modelName]);

  const jsonString = useMemo(() => JSON.stringify(exportData, null, 2), [exportData]);

  const handleDownload = () => {
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${modelName.replace(/\s+/g, '_')}.neuralforge.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(jsonString);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { }
  };

  if (!isOpen) return null;

  const paramCount = estimateParams(nodes);
  const tfLayerCount = (exportData.tensorflowjs.layers as object[]).length;
  const bg = isDark ? '#1E293B' : '#FFFFFF';
  const border = isDark ? '#334155' : '#E2E8F0';
  const text = isDark ? '#E2E8F0' : '#1E293B';
  const sub = isDark ? '#94A3B8' : '#64748B';
  const card = isDark ? '#0F172A' : '#F8FAFC';

  return (
    <div
      role="dialog" aria-label="Export Model" aria-modal="true"
      className="absolute top-4 right-4 z-50 nf-animate-in w-[320px] max-h-[calc(100vh-140px)] flex flex-col rounded-2xl overflow-hidden"
      style={{
        background: bg, border: `1px solid ${border}`,
        boxShadow: '0 24px 64px rgba(0,0,0,0.16), 0 4px 16px rgba(0,0,0,0.08)',
      }}
    >
      {/* Header */}
      <div className="flex items-center gap-2.5 px-4 py-3 flex-shrink-0"
        style={{ background: 'linear-gradient(135deg, #1E40AF, #3B82F6)', borderBottom: `1px solid rgba(255,255,255,0.12)` }}>
        <div className="w-7 h-7 rounded-lg bg-white/20 flex items-center justify-center flex-shrink-0 shadow-inner">
          <FileJson size={14} className="text-white" />
        </div>
        <div className="flex-1">
          <div className="text-[13px] font-bold text-white">Export Model</div>
          <div className="text-[10px] text-blue-200">Download JSON · TF.js compatible</div>
        </div>
        <button onClick={onClose} aria-label="Close export panel"
          className="w-6 h-6 rounded-lg bg-white/15 hover:bg-white/25 flex items-center justify-center text-white/80 hover:text-white transition-all">
          <X size={12} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Stats strip */}
        <div className="px-4 pt-4 grid grid-cols-3 gap-2 mb-4">
          {[
            { label: 'Layers', value: nodes.length, icon: <Layers size={10} />, color: '#3B82F6' },
            { label: 'TF Layers', value: tfLayerCount, icon: <Brain size={10} />, color: '#8B5CF6' },
            { label: 'Params', value: formatParams(paramCount), icon: <Code2 size={10} />, color: '#10B981' },
          ].map(s => (
            <div key={s.label} className="rounded-xl p-2 text-center" style={{ background: card, border: `1px solid ${border}` }}>
              <div className="flex items-center justify-center gap-1 mb-1" style={{ color: s.color }}>
                {s.icon}
              </div>
              <div className="text-[15px] font-bold tabular-nums" style={{ color: text }}>{s.value}</div>
              <div className="text-[9px]" style={{ color: sub }}>{s.label}</div>
            </div>
          ))}
        </div>

        {/* Model name */}
        <div className="px-4 mb-4">
          <label className="text-[10px] font-semibold uppercase tracking-wider block mb-1.5" style={{ color: sub }}>
            Model Name
          </label>
          <input
            value={modelName}
            onChange={e => setModelName(e.target.value)}
            className="w-full px-3 py-2 rounded-xl text-[12px] font-mono border focus:outline-none transition-all"
            style={{ background: card, border: `1px solid ${border}`, color: text }}
            placeholder="my_neural_net"
          />
        </div>

        {/* TF.js note */}
        <div className="mx-4 mb-4 flex items-start gap-2 rounded-xl p-3"
          style={{ background: isDark ? '#1E3A5F' : '#EFF6FF', border: `1px solid ${isDark ? '#2563EB44' : '#BFDBFE'}` }}>
          <Info size={12} className="text-blue-500 flex-shrink-0 mt-0.5" />
          <p className="text-[10.5px] leading-relaxed" style={{ color: isDark ? '#93C5FD' : '#1D4ED8' }}>
            The exported JSON includes a <strong>tensorflowjs</strong> key with layer descriptors and a reconstruction snippet for use with <code>tf.sequential()</code>.
            <a href="https://www.tensorflow.org/js" target="_blank" rel="noreferrer"
              className="flex items-center gap-0.5 mt-1 font-semibold opacity-75 hover:opacity-100 transition-opacity">
              TF.js docs <ExternalLink size={9} />
            </a>
          </p>
        </div>

        {/* JSON preview */}
        <div className="px-4 mb-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-1.5" style={{ color: sub }}>
            JSON Preview
          </div>
          <div className="rounded-xl overflow-hidden" style={{ border: `1px solid ${border}` }}>
            <pre className="text-[9px] font-mono p-3 overflow-auto max-h-[160px] leading-relaxed"
              style={{ background: isDark ? '#0F172A' : '#F1F5F9', color: isDark ? '#94A3B8' : '#475569' }}>
              {jsonString.slice(0, 800)}{jsonString.length > 800 ? '\n  ... (truncated in preview)' : ''}
            </pre>
          </div>
        </div>

        {/* Training history note */}
        {simulation.history.length > 0 && (
          <div className="mx-4 mb-4 flex items-start gap-2 rounded-xl p-3"
            style={{ background: isDark ? '#052e16' : '#F0FDF4', border: `1px solid ${isDark ? '#16a34a44' : '#BBF7D0'}` }}>
            <CheckCheck size={12} className="text-emerald-500 flex-shrink-0 mt-0.5" />
            <p className="text-[10.5px]" style={{ color: isDark ? '#86EFAC' : '#15803D' }}>
              Training history included: <strong>{simulation.epoch} epochs</strong>,
              final accuracy <strong>{(simulation.accuracy * 100).toFixed(1)}%</strong>,
              loss <strong>{simulation.loss.toFixed(4)}</strong>.
            </p>
          </div>
        )}
      </div>

      {/* Footer buttons */}
      <div className="px-4 py-3 flex gap-2 flex-shrink-0" style={{ borderTop: `1px solid ${border}`, background: card }}>
        <button
          onClick={handleCopy}
          className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl text-[11px] font-semibold border transition-all"
          style={{ background: card, border: `1px solid ${border}`, color: sub }}>
          {copied ? <><CheckCheck size={12} className="text-emerald-500" /> Copied!</> : <><Copy size={12} /> Copy JSON</>}
        </button>
        <button
          onClick={handleDownload}
          className="flex-1 flex items-center justify-center gap-2 py-2 rounded-xl text-[12px] font-bold text-white transition-all active:scale-[0.98] nf-bevel"
          style={{ background: 'linear-gradient(135deg, #1E40AF, #3B82F6)', boxShadow: '0 4px 14px rgba(30,64,175,0.35)' }}>
          <Download size={13} />
          Download JSON
        </button>
      </div>
    </div>
  );
}
