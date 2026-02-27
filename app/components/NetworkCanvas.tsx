import React, {
  useRef, useState, useCallback, useEffect, useMemo,
} from 'react';
import { Trash2, Settings, Copy, Power, PowerOff, Edit3 } from 'lucide-react';
import {
  NetworkNode, NetworkConnection, Particle,
  NODE_DIMS, NODE_STYLE, NodeType, NodeConfig,
} from './types';
import {
  ActivationBadge, ActivationSwitcher, getActivationDef,
} from './ActivationSwitcher';

// ─── Constants ────────────────────────────────────────────────────────────────
const GRID = 20;
const MIN_NODE_W = 120;
const MIN_NODE_H = 56;
const OVERLAP_PADDING = 8;
const ALIGN_SNAP_DIST = 7;
const WEIGHT_MIN = -2;
const WEIGHT_MAX = 2;
const WEIGHT_DEFAULT = 0.5;

// ─── Weight visual helpers ────────────────────────────────────────────────────
interface WeightVisual {
  stroke: string;       // wire colour
  width: number;        // stroke-width
  dashArray: string;    // SVG dash pattern, '' = solid
  opacity: number;
  labelColor: string;   // weight badge text/border colour
  arrowId: string;      // marker id
  glowFilter: boolean;
}

function getWeightVisual(
  weight: number,
  isActive: boolean,
  isSimulating: boolean,
  isSelected: boolean,
  isHovered: boolean,
): WeightVisual {
  const w = weight;
  const abs = Math.abs(w);

  // Colour: blue positive → grey zero → red negative
  let stroke: string;
  let labelColor: string;
  if (abs < 0.08) {
    stroke = '#94A3B8'; labelColor = '#64748B';
  } else if (w > 0) {
    if (w < 0.35) { stroke = '#93C5FD'; labelColor = '#2563EB'; }
    else if (w < 0.65) { stroke = '#3B82F6'; labelColor = '#1D4ED8'; }
    else if (w < 0.85) { stroke = '#1E40AF'; labelColor = '#1E3A8A'; }
    else { stroke = '#172554'; labelColor = '#172554'; }
  } else {
    if (-w < 0.35) { stroke = '#FDE68A'; labelColor = '#D97706'; }
    else if (-w < 0.65) { stroke = '#F59E0B'; labelColor = '#B45309'; }
    else if (-w < 0.85) { stroke = '#D97706'; labelColor = '#92400E'; }
    else { stroke = '#B45309'; labelColor = '#78350F'; }
  }

  if (isSelected) { stroke = '#F59E0B'; labelColor = '#92400E'; }
  if (isHovered && !isSelected) { stroke = stroke; }

  // Width: proportional to |weight|, range [1.5, 4.5]
  const width = isSelected ? 3 : Math.max(1.5, Math.min(4.5, 1.5 + abs * 3));

  // Dash: solid only when simulating AND active
  const isDashed = !isActive || !isSimulating;
  const gapRatio = isActive ? 1 : 1.5;
  const dashLen = Math.max(4, 6 + abs * 4);
  const dashArray = isDashed ? `${dashLen},${dashLen * gapRatio * 0.6}` : '';

  const opacity = isActive ? 1 : 0.4;

  const arrowId = isSelected
    ? 'arrow-sel'
    : isSimulating && isActive ? 'arrow-active' : 'arrow-idle';

  const glowFilter = isSimulating && isActive;

  return { stroke, width, dashArray, opacity, labelColor, arrowId, glowFilter };
}

// ─── Bezier helpers ───────────────────────────────────────────────────────────
function bezierPt(t: number, x1: number, y1: number, cx1: number, cy1: number, cx2: number, cy2: number, x2: number, y2: number) {
  const m = 1 - t;
  return {
    x: m * m * m * x1 + 3 * m * m * t * cx1 + 3 * m * t * t * cx2 + t * t * t * x2,
    y: m * m * m * y1 + 3 * m * m * t * cy1 + 3 * m * t * t * cy2 + t * t * t * y2,
  };
}

function bezierTangent(t: number, x1: number, y1: number, cx1: number, cy1: number, cx2: number, cy2: number, x2: number, y2: number) {
  const m = 1 - t;
  const tx = 3 * (m * m * (cx1 - x1) + 2 * m * t * (cx2 - cx1) + t * t * (x2 - cx2));
  const ty = 3 * (m * m * (cy1 - y1) + 2 * m * t * (cy2 - cy1) + t * t * (y2 - cy2));
  const len = Math.sqrt(tx * tx + ty * ty) || 1;
  return { tx: tx / len, ty: ty / len };
}

// Normal (perpendicular, pointing "up" in canvas)
function bezierNormal(t: number, ...args: Parameters<typeof bezierTangent>) {
  const { tx, ty } = bezierTangent(t, ...args);
  return { nx: -ty, ny: tx };
}

// ─── Misc helpers ───────��─────────────────────────────────────────────────────
const snapToGrid = (v: number) => Math.round(v / GRID) * GRID;

const TYPE_LABEL: Record<NodeType, string> = {
  input: 'IN', dense: 'FC', activation: 'fn', dropout: 'DO',
  output: 'OUT', conv2d: '2D', flatten: 'FT', lstm: 'RNN',
  batchnorm: 'BN', embedding: 'EMB',
};

function getNodeSubtitle(node: NetworkNode): string {
  const c = node.config;
  if (node.type === 'input') return `shape: [${c.inputShape || '?'}]`;
  if (node.type === 'dense') return `${c.neurons || 64} neurons`;
  if (node.type === 'activation') return c.activationFn || 'ReLU';
  if (node.type === 'dropout') return `rate: ${c.dropoutRate || 0.5}`;
  if (node.type === 'output') return `classes: ${c.outputShape || '10'}`;
  if (node.type === 'conv2d') return `${c.filters || 32} × ${c.kernelSize || 3}×${c.kernelSize || 3}`;
  if (node.type === 'flatten') return 'reshape to 1D';
  if (node.type === 'lstm') return `${c.units || 64} units`;
  if (node.type === 'batchnorm') return 'normalize';
  if (node.type === 'embedding') return `${c.vocabSize || 10000} vocab`;
  return '';
}

function getNodeDims(node: NetworkNode) {
  const base = NODE_DIMS[node.type];
  return { w: node.w ?? base.w, h: node.h ?? base.h };
}

// ��── Overlap & snap-guide helpers ─────────────────────────────────────────────
function getOverlappingIds(id: string, x: number, y: number, w: number, h: number, nodes: NetworkNode[]): Set<string> {
  const r = new Set<string>();
  for (const n of nodes) {
    if (n.id === id) continue;
    const { w: nw, h: nh } = getNodeDims(n);
    if (x < n.x + nw + OVERLAP_PADDING && x + w + OVERLAP_PADDING > n.x &&
      y < n.y + nh + OVERLAP_PADDING && y + h + OVERLAP_PADDING > n.y) r.add(n.id);
  }
  return r;
}

interface SnapGuide { dir: 'h' | 'v'; pos: number; from: number; to: number; }

function computeSnapGuides(id: string, x: number, y: number, w: number, h: number, nodes: NetworkNode[]): SnapGuide[] {
  const gs: SnapGuide[] = [];
  const [mL, mR, mCX] = [x, x + w, x + w / 2];
  const [mT, mB, mCY] = [y, y + h, y + h / 2];
  for (const n of nodes) {
    if (n.id === id) continue;
    const { w: nw, h: nh } = getNodeDims(n);
    const [nL, nR, nCX] = [n.x, n.x + nw, n.x + nw / 2];
    const [nT, nB, nCY] = [n.y, n.y + nh, n.y + nh / 2];
    for (const [mv, ov] of [[mT, nT], [mT, nCY], [mT, nB], [mCY, nT], [mCY, nCY], [mCY, nB], [mB, nT], [mB, nCY], [mB, nB]] as [number, number][]) {
      if (Math.abs(mv - ov) < ALIGN_SNAP_DIST)
        gs.push({ dir: 'h', pos: ov, from: Math.min(mL, nL) - 24, to: Math.max(mR, nR) + 24 });
    }
    for (const [mv, ov] of [[mL, nL], [mL, nCX], [mL, nR], [mCX, nL], [mCX, nCX], [mCX, nR], [mR, nL], [mR, nCX], [mR, nR]] as [number, number][]) {
      if (Math.abs(mv - ov) < ALIGN_SNAP_DIST)
        gs.push({ dir: 'v', pos: ov, from: Math.min(mT, nT) - 24, to: Math.max(mB, nB) + 24 });
    }
  }
  return gs.filter((g, i, a) => a.findIndex(o => o.dir === g.dir && Math.abs(o.pos - g.pos) < 2) === i);
}

// ─── Resize handles ───────────────────────────────────────────────────────────
type RHandle = 'tl' | 't' | 'tr' | 'r' | 'br' | 'b' | 'bl' | 'l';
const RESIZE_HANDLES: { id: RHandle; cursor: string; getPos: (w: number, h: number) => { left: number; top: number } }[] = [
  { id: 'tl', cursor: 'nw-resize', getPos: (w, h) => ({ left: 0, top: 0 }) },
  { id: 't', cursor: 'n-resize', getPos: (w, h) => ({ left: w / 2, top: 0 }) },
  { id: 'tr', cursor: 'ne-resize', getPos: (w, h) => ({ left: w, top: 0 }) },
  { id: 'r', cursor: 'e-resize', getPos: (w, h) => ({ left: w, top: h / 2 }) },
  { id: 'br', cursor: 'se-resize', getPos: (w, h) => ({ left: w, top: h }) },
  { id: 'b', cursor: 's-resize', getPos: (w, h) => ({ left: w / 2, top: h }) },
  { id: 'bl', cursor: 'sw-resize', getPos: (w, h) => ({ left: 0, top: h }) },
  { id: 'l', cursor: 'w-resize', getPos: (w, h) => ({ left: 0, top: h / 2 }) },
];

// ─── Canvas node component ────────────────────────────────────────────────────
interface CanvasNodeProps {
  node: NetworkNode;
  isSelected: boolean; isDragging: boolean; isConnectingFrom: boolean;
  isConnecting: boolean; isOverlapping: boolean; isActive: boolean;
  propActivation?: number;   // 0-1 activation value from forward pass
  inPropWave?: boolean;      // currently in the animated wave-front
  inPropDone?: boolean;      // already propagated (dim settled glow)
  onPointerDown: (e: React.PointerEvent, id: string) => void;
  onResizeStart: (e: React.PointerEvent, id: string, h: RHandle) => void;
  onOutputPortClick: (e: React.MouseEvent, id: string) => void;
  onInputPortClick: (e: React.MouseEvent, id: string) => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onDuplicate: (id: string) => void;
  onConfigChange: (id: string, cfg: Partial<NodeConfig>) => void;
}

function CanvasNodeComponent({
  node, isSelected, isDragging, isConnectingFrom, isConnecting, isOverlapping, isActive,
  propActivation, inPropWave, inPropDone,
  onPointerDown, onResizeStart, onOutputPortClick, onInputPortClick,
  onSelect, onDelete, onDuplicate, onConfigChange,
}: CanvasNodeProps) {
  const style = NODE_STYLE[node.type];
  const { w, h } = getNodeDims(node);
  const [showConfig, setShowConfig] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  // ── Activation switcher hover logic ──────────────────────────────────────
  const [showSwitcher, setShowSwitcher] = useState(false);
  const leaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isActivationNode = node.type === 'activation';
  const currentFn = node.config.activationFn ?? 'ReLU';
  const actDef = isActivationNode ? getActivationDef(currentFn) : null;

  const handleNodeMouseEnter = useCallback(() => {
    if (leaveTimerRef.current) clearTimeout(leaveTimerRef.current);
    setIsHovered(true);
    if (isActivationNode && !isDragging && !isConnecting && !isConnectingFrom) {
      setShowSwitcher(true);
    }
  }, [isActivationNode, isDragging, isConnecting, isConnectingFrom]);

  const handleNodeMouseLeave = useCallback(() => {
    leaveTimerRef.current = setTimeout(() => {
      setIsHovered(false);
      setShowSwitcher(false);
    }, 120); // grace period so popup stays when moving into it
  }, []);

  const handleSwitcherMouseEnter = useCallback(() => {
    if (leaveTimerRef.current) clearTimeout(leaveTimerRef.current);
  }, []);

  const handleSwitcherMouseLeave = useCallback(() => {
    leaveTimerRef.current = setTimeout(() => {
      setIsHovered(false);
      setShowSwitcher(false);
    }, 80);
  }, []);

  // Hide switcher when dragging starts
  useEffect(() => {
    if (isDragging) { setShowSwitcher(false); setIsHovered(false); }
  }, [isDragging]);

  useEffect(() => () => { if (leaveTimerRef.current) clearTimeout(leaveTimerRef.current); }, []);

  const handleActivationSelect = useCallback((fn: string) => {
    onConfigChange(node.id, { activationFn: fn });
    // Update label to match
    // (NeuralNetworkBuilder's handleConfigChange covers config; label update via onConfigChange extension)
  }, [node.id, onConfigChange]);

  // For activation nodes, use activation-specific glow color
  const glowDot = isActivationNode && actDef ? actDef.color : style.dot;
  const glowColor = isActivationNode && actDef
    ? `rgba(${parseInt(actDef.color.slice(1, 3), 16)},${parseInt(actDef.color.slice(3, 5), 16)},${parseInt(actDef.color.slice(5, 7), 16)},0.35)`
    : style.glowColor;

  const boxShadow = useMemo(() => {
    if (isOverlapping) return '0 0 0 2.5px #EF4444, 0 0 16px rgba(239,68,68,0.35)';
    if (isDragging) return `0 24px 48px rgba(30,64,175,0.22), 0 8px 16px rgba(0,0,0,0.15), 0 0 0 2px ${glowColor.replace('0.35', '0.5')}, inset 0 2px 4px rgba(255,255,255,0.6)`;
    if (inPropWave) return `0 0 0 3px ${glowDot}, 0 0 28px ${glowDot}88, 0 8px 24px ${glowDot}33, inset 0 2px 4px rgba(255,255,255,0.6)`;
    if (inPropDone && propActivation !== undefined && propActivation >= 0.5)
      return `0 0 0 2px ${glowDot}88, 0 0 16px ${glowDot}44, inset 0 2px 4px rgba(255,255,255,0.6)`;
    if (isConnectingFrom) return '0 0 0 3px rgba(30,64,175,0.8), 0 0 20px rgba(30,64,175,0.3), inset 0 2px 4px rgba(255,255,255,0.6)';
    if (isSelected) return `0 0 0 2px ${isActivationNode ? glowDot : '#F59E0B'}, 0 0 0 5px ${isActivationNode ? glowColor.replace('0.35', '0.15') : 'rgba(245,158,11,0.25)'}, 0 4px 16px rgba(0,0,0,0.1), inset 0 2px 4px rgba(255,255,255,0.6)`;
    if (showSwitcher) return `0 0 0 2.5px ${glowDot}99, 0 8px 28px ${glowColor.replace('0.35', '0.18')}, 0 2px 8px rgba(0,0,0,0.08), inset 0 2px 4px rgba(255,255,255,0.6)`;
    if (isHovered) return `0 0 0 2px ${glowColor}, 0 8px 24px ${glowColor.replace('0.35', '0.15')}, 0 2px 8px rgba(0,0,0,0.08), inset 0 2px 4px rgba(255,255,255,0.5)`;
    // Default bevel format
    return '0 2px 6px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.04), inset 0 2px 4px rgba(255,255,255,0.6), inset 0 -2px 4px rgba(0,0,0,0.04)';
  }, [isDragging, isSelected, isHovered, showSwitcher, isOverlapping, isConnectingFrom, glowDot, glowColor, isActivationNode]);

  return (
    <div
      style={{
        position: 'absolute', left: 0, top: 0, width: w, height: h,
        transform: `translate(${node.x}px,${node.y}px) ${isDragging ? 'scale(1.04)' : 'scale(1)'}`,
        transformOrigin: 'center center',
        zIndex: isDragging ? 1000 : isSelected ? 30 : 10,
        transition: isDragging
          ? 'transform 0s, box-shadow 0.1s'
          : 'transform 0.07s cubic-bezier(0.2,0,0.2,1), box-shadow 0.15s ease',
        willChange: isDragging ? 'transform' : 'auto',
        opacity: isDragging ? 0.93 : 1,
      }}
      className="group"
      onMouseEnter={handleNodeMouseEnter}
      onMouseLeave={handleNodeMouseLeave}
    >
      {/* Input port */}
      <div
        className={`absolute left-0 top-1/2 -translate-x-1/2 -translate-y-1/2 z-40 w-3.5 h-3.5 rounded-full border-2 bg-white transition-all duration-150
          ${isConnecting ? 'border-amber-500 hover:bg-amber-500 hover:scale-150 cursor-crosshair shadow-[0_0_8px_rgba(245,158,11,0.6)]'
            : 'border-slate-300 hover:border-blue-700 hover:bg-blue-50 hover:scale-125 cursor-crosshair'}`}
        onClick={e => onInputPortClick(e, node.id)}
        title="Input port — click to complete connection"
      />

      {/* Body */}
      <div
        className={`absolute inset-0 rounded-xl border-2 select-none overflow-visible ${isActivationNode ? '' : style.bg} ${isActivationNode ? '' : style.border} ${style.text}`}
        style={{
          cursor: isDragging ? 'grabbing' : 'grab',
          boxShadow,
          // Activation nodes get dynamic background + border based on selected fn
          ...(isActivationNode && actDef ? {
            background: `linear-gradient(145deg, ${actDef.color}0D 0%, ${actDef.color}06 100%)`,
            borderColor: showSwitcher || isHovered || isSelected ? `${actDef.color}88` : `${actDef.color}44`,
          } : {}),
          transition: isDragging ? 'box-shadow 0.1s' : 'box-shadow 0.2s ease, border-color 0.2s ease, background 0.25s ease',
        }}
        onPointerDown={e => { onSelect(node.id); onPointerDown(e, node.id); }}
      >
        {isActive && <div className="absolute inset-0 rounded-xl animate-pulse" style={{ background: glowColor.replace('0.35', '0.08') }} />}

        {/* Grip dots */}
        <div className="absolute top-1.5 left-1.5 z-20 flex flex-col gap-[3px] opacity-0 group-hover:opacity-40 transition-opacity pointer-events-none">
          {[0, 1, 2].map(r => (
            <div key={r} className="flex gap-[3px]">
              {[0, 1].map(c => <div key={c} className="w-[3px] h-[3px] rounded-full bg-current opacity-70" />)}
            </div>
          ))}
        </div>

        {/* Content */}
        <div className="flex flex-col items-center justify-center h-full px-4 relative z-10 gap-0.5">
          {isActivationNode ? (
            /* Activation node: show animated curve badge + colored label */
            <>
              <div
                className="flex items-center justify-center rounded-lg px-2 py-1"
                style={{
                  background: actDef ? `${actDef.color}18` : style.dot + '22',
                  border: `1.5px solid ${actDef ? actDef.color + '40' : style.dot + '40'}`,
                  transition: 'background 0.25s ease, border-color 0.25s ease',
                }}
              >
                <ActivationBadge fn={currentFn} size={34} />
              </div>
              <div
                className="text-[11px] font-bold truncate max-w-full text-center mt-0.5"
                style={{
                  color: actDef?.dotColor ?? style.dot,
                  transition: 'color 0.2s ease',
                }}
              >
                {currentFn}
              </div>
              <div className="text-[9px] opacity-50 truncate max-w-full text-center font-mono">
                {actDef?.formula.replace('f(x) = ', '') ?? currentFn}
              </div>
            </>
          ) : (
            /* All other nodes: standard badge */
            <>
              <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white text-[10px] font-bold shadow-sm"
                style={{ background: `linear-gradient(135deg,${style.dot},${style.dot}cc)` }}>
                {TYPE_LABEL[node.type]}
              </div>
              <div className="text-[11px] font-semibold truncate max-w-full text-center mt-0.5">{node.label}</div>
              <div className="text-[10px] opacity-50 truncate max-w-full text-center">{getNodeSubtitle(node)}</div>
            </>
          )}
        </div>

        {/* ── Forward propagation overlays ────────────────────────────────── */}
        {propActivation !== undefined && (
          <>
            {/* Bottom fill bar showing activation magnitude */}
            <div className="absolute bottom-0 left-0 right-0 h-1.5 bg-black/5 rounded-b-xl overflow-hidden pointer-events-none z-20">
              <div
                className="h-full transition-all duration-500"
                style={{
                  width: `${propActivation * 100}%`,
                  background: propActivation >= 0.5
                    ? `linear-gradient(90deg, ${style.dot}88, ${style.dot})`
                    : 'linear-gradient(90deg, #94A3B866, #94A3B8)',
                  borderRadius: '0 4px 4px 0',
                }}
              />
            </div>
            {/* Radial inner glow */}
            <div
              className="absolute inset-0 rounded-xl pointer-events-none transition-all duration-400"
              style={{
                background: inPropWave
                  ? `radial-gradient(ellipse at center, ${style.dot}44 0%, transparent 75%)`
                  : inPropDone && propActivation >= 0.5
                    ? `radial-gradient(ellipse at center, ${style.dot}1A 0%, transparent 70%)`
                    : 'none',
                zIndex: 1,
              }}
            />
            {/* Wave-front ping ring */}
            {inPropWave && (
              <div
                className="absolute inset-0 rounded-xl pointer-events-none animate-ping opacity-25"
                style={{ boxShadow: `inset 0 0 0 2px ${style.dot}`, background: `${style.dot}0A` }}
              />
            )}
          </>
        )}

        {/* Action toolbar */}
        {isSelected && (
          <div className="absolute -top-9 left-1/2 -translate-x-1/2 flex gap-1 z-50" onPointerDown={e => e.stopPropagation()}>
            <button onClick={e => { e.stopPropagation(); setShowConfig(v => !v); }}
              className={`w-7 h-7 rounded-lg border flex items-center justify-center shadow-sm transition-all duration-100
                ${showConfig ? 'bg-blue-500 border-blue-500 text-white' : 'bg-white border-slate-200 text-slate-500 hover:bg-blue-50 hover:border-blue-300 hover:text-blue-600'}`}
              title="Configure">
              <Settings size={11} />
            </button>
            <button onClick={e => { e.stopPropagation(); onDuplicate(node.id); }}
              className="w-7 h-7 rounded-lg border border-slate-200 bg-white flex items-center justify-center shadow-sm text-slate-500 hover:bg-indigo-50 hover:border-indigo-300 hover:text-indigo-600 transition-all duration-100" title="Duplicate">
              <Copy size={11} />
            </button>
            <button onClick={e => { e.stopPropagation(); onDelete(node.id); }}
              className="w-7 h-7 rounded-lg border border-slate-200 bg-white flex items-center justify-center shadow-sm text-slate-500 hover:bg-red-50 hover:border-red-300 hover:text-red-500 transition-all duration-100" title="Delete">
              <Trash2 size={11} />
            </button>
          </div>
        )}

        {/* Config panel */}
        {showConfig && isSelected && (
          <div className="absolute top-full mt-3 left-1/2 -translate-x-1/2 z-[60] bg-white rounded-xl shadow-2xl border border-slate-200 p-4 w-52"
            onPointerDown={e => e.stopPropagation()} onClick={e => e.stopPropagation()}
            style={{ boxShadow: '0 20px 40px rgba(0,0,0,0.12),0 0 0 1px rgba(0,0,0,0.05)' }}>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-5 h-5 rounded-md flex items-center justify-center text-white text-[9px] font-bold"
                style={{ background: `linear-gradient(135deg,${style.dot},${style.dot}cc)` }}>{TYPE_LABEL[node.type]}</div>
              <span className="text-[12px] font-semibold text-slate-700">{node.label}</span>
            </div>
            <div className="space-y-3">
              {node.type === 'dense' && (
                <label className="flex flex-col gap-1.5">
                  <span className="text-[10px] font-medium text-slate-500">NEURONS</span>
                  <input type="number" min={1} max={2048} value={node.config.neurons ?? 64}
                    onChange={e => onConfigChange(node.id, { neurons: +e.target.value })}
                    className="text-[12px] px-2.5 py-1.5 border border-slate-200 rounded-lg w-full focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all" />
                </label>
              )}
              {node.type === 'activation' && (
                <label className="flex flex-col gap-1.5">
                  <span className="text-[10px] font-medium text-slate-500">FUNCTION</span>
                  <select value={node.config.activationFn || 'ReLU'} onChange={e => onConfigChange(node.id, { activationFn: e.target.value })}
                    className="text-[12px] px-2.5 py-1.5 border border-slate-200 rounded-lg w-full focus:outline-none focus:border-violet-400 focus:ring-2 focus:ring-violet-100 transition-all bg-white">
                    {['ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'GELU', 'LeakyReLU', 'Swish'].map(fn => <option key={fn} value={fn}>{fn}</option>)}
                  </select>
                </label>
              )}
              {node.type === 'dropout' && (
                <label className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-medium text-slate-500">RATE</span>
                    <span className="text-[11px] font-semibold text-orange-600">{node.config.dropoutRate ?? 0.5}</span>
                  </div>
                  <input type="range" min={0.05} max={0.95} step={0.05} value={node.config.dropoutRate ?? 0.5}
                    onChange={e => onConfigChange(node.id, { dropoutRate: +e.target.value })} className="w-full accent-orange-400" />
                </label>
              )}
              {node.type === 'conv2d' && (
                <div className="space-y-2">
                  <label className="flex flex-col gap-1.5">
                    <span className="text-[10px] font-medium text-slate-500">FILTERS</span>
                    <input type="number" min={1} max={512} value={node.config.filters ?? 32}
                      onChange={e => onConfigChange(node.id, { filters: +e.target.value })}
                      className="text-[12px] px-2.5 py-1.5 border border-slate-200 rounded-lg w-full focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-100 transition-all" />
                  </label>
                  <label className="flex flex-col gap-1.5">
                    <span className="text-[10px] font-medium text-slate-500">KERNEL</span>
                    <input type="number" min={1} max={11} value={node.config.kernelSize ?? 3}
                      onChange={e => onConfigChange(node.id, { kernelSize: +e.target.value })}
                      className="text-[12px] px-2.5 py-1.5 border border-slate-200 rounded-lg w-full focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-100 transition-all" />
                  </label>
                </div>
              )}
              {node.type === 'lstm' && (
                <label className="flex flex-col gap-1.5">
                  <span className="text-[10px] font-medium text-slate-500">UNITS</span>
                  <input type="number" min={1} max={1024} value={node.config.units ?? 64}
                    onChange={e => onConfigChange(node.id, { units: +e.target.value })}
                    className="text-[12px] px-2.5 py-1.5 border border-slate-200 rounded-lg w-full focus:outline-none focus:border-pink-400 focus:ring-2 focus:ring-pink-100 transition-all" />
                </label>
              )}
              {node.type === 'input' && (
                <label className="flex flex-col gap-1.5">
                  <span className="text-[10px] font-medium text-slate-500">INPUT SHAPE</span>
                  <input type="text" value={node.config.inputShape ?? '784'}
                    onChange={e => onConfigChange(node.id, { inputShape: e.target.value })}
                    className="text-[12px] px-2.5 py-1.5 border border-slate-200 rounded-lg w-full focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all" />
                </label>
              )}
              {node.type === 'output' && (
                <label className="flex flex-col gap-1.5">
                  <span className="text-[10px] font-medium text-slate-500">OUTPUT CLASSES</span>
                  <input type="text" value={node.config.outputShape ?? '10'}
                    onChange={e => onConfigChange(node.id, { outputShape: e.target.value })}
                    className="text-[12px] px-2.5 py-1.5 border border-slate-200 rounded-lg w-full focus:outline-none focus:border-teal-400 focus:ring-2 focus:ring-teal-100 transition-all" />
                </label>
              )}
              {(node.type === 'flatten' || node.type === 'batchnorm' || node.type === 'embedding') && (
                <div className="text-[11px] text-slate-400 italic py-1">No configurable parameters</div>
              )}
            </div>
            <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-400">
              <span>{w}×{h}px</span>
              <span>grid {Math.round(node.x / GRID)},{Math.round(node.y / GRID)}</span>
            </div>
            <button onClick={() => setShowConfig(false)}
              className="mt-2 w-full py-1.5 text-[11px] bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg transition-colors text-slate-600">
              Done
            </button>
          </div>
        )}
      </div>

      {/* Activation switcher popup — appears below the node on hover */}
      {isActivationNode && (
        <div
          onMouseEnter={handleSwitcherMouseEnter}
          onMouseLeave={handleSwitcherMouseLeave}
        >
          <ActivationSwitcher
            currentFn={currentFn}
            onSelect={handleActivationSelect}
            visible={showSwitcher && !isDragging}
          />
        </div>
      )}

      {/* Output port */}
      <div
        className={`absolute right-0 top-1/2 translate-x-1/2 -translate-y-1/2 z-40 w-3.5 h-3.5 rounded-full border-2 bg-white transition-all duration-150 cursor-crosshair
          ${isConnectingFrom
            ? 'border-amber-500 bg-amber-500 scale-150 shadow-[0_0_10px_rgba(245,158,11,0.8)]'
            : 'border-slate-300 hover:border-teal-600 hover:bg-teal-50 hover:scale-150 hover:shadow-[0_0_8px_rgba(13,148,136,0.5)]'}`}
        onClick={e => onOutputPortClick(e, node.id)}
        title="Output port — click to start connection"
      />

      {/* Resize handles */}
      {isSelected && !isDragging && RESIZE_HANDLES.map(rh => {
        const pos = rh.getPos(w, h);
        return (
          <div key={rh.id}
            style={{ position: 'absolute', left: pos.left, top: pos.top, transform: 'translate(-50%,-50%)', cursor: rh.cursor, zIndex: 50 }}
            className="w-3 h-3 bg-white border-2 border-blue-500 rounded-sm shadow-sm hover:bg-blue-50 hover:scale-125 transition-transform"
            onPointerDown={e => { e.stopPropagation(); onResizeStart(e, node.id, rh.id); }} />
        );
      })}

      {isDragging && (
        <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-[9px] bg-blue-600 text-white px-1.5 py-0.5 rounded-md font-mono whitespace-nowrap shadow-sm" style={{ pointerEvents: 'none' }}>
          {node.x},{node.y}
        </div>
      )}
    </div>
  );
}

// ─── Multiplayer cursors ──────────────────────────────────────────────────────
const MOCK_CURSORS = [
  { id: 'alice', name: 'Alice', color: '#EF4444', x: 640, y: 180 },
  { id: 'bob', name: 'Bob', color: '#10B981', x: 1020, y: 360 },
];

// ─── Validation toast ──────────────────────────────────────────────────────────
interface ToastMsg { id: number; text: string; kind: 'error' | 'warn' }

// ─── Main canvas props ────────────────────────────────────────────────────────
interface NetworkCanvasProps {
  nodes: NetworkNode[];
  connections: NetworkConnection[];
  particles: Particle[];
  selectedNodeId: string | null;
  connectingFrom: string | null;
  isSimulating: boolean;
  onNodesChange: (n: NetworkNode[]) => void;
  onConnectionCreate: (fromId: string, toId: string) => void;
  onConnectionDelete: (id: string) => void;
  onConnectionWeightChange: (id: string, weight: number) => void;
  onConnectionActiveToggle: (id: string) => void;
  onSelectNode: (id: string | null) => void;
  onStartConnecting: (id: string) => void;
  onCancelConnecting: () => void;
  onDeleteNode: (id: string) => void;
  onDuplicateNode: (id: string) => void;
  onConfigChange: (id: string, cfg: Partial<NodeConfig>) => void;
  onDropNewNode: (type: NodeType, label: string, cfg: NodeConfig, x: number, y: number) => void;
  onNodeDragStart: () => void;
  onResizeNodeStart: () => void;
  // ── Forward propagation visualization ──
  propNodeActivations?: Record<string, number>;
  propWaveNodeIds?: string[];
  propDoneNodeIds?: string[];
  propWaveConnIds?: string[];
  propDoneConnIds?: string[];
  // ── Training mode ──
  trainPhase?: 'idle' | 'forward' | 'backprop' | 'update';
  trainActiveConnIds?: string[];
  trainWeightDeltas?: Record<string, number>;
  trainGradients?: Record<string, number>;
}

export function NetworkCanvas({
  nodes, connections, particles,
  selectedNodeId, connectingFrom, isSimulating,
  onNodesChange, onConnectionCreate, onConnectionDelete,
  onConnectionWeightChange, onConnectionActiveToggle,
  onSelectNode, onStartConnecting, onCancelConnecting,
  onDeleteNode, onDuplicateNode, onConfigChange, onDropNewNode,
  onNodeDragStart, onResizeNodeStart,
  propNodeActivations, propWaveNodeIds, propDoneNodeIds,
  propWaveConnIds, propDoneConnIds,
  trainPhase, trainActiveConnIds, trainWeightDeltas, trainGradients,
}: NetworkCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);

  // View state
  const [pan, setPan] = useState({ x: 40, y: 40 });
  const [zoom, setZoom] = useState(1);
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef<{ cx: number; cy: number; px: number; py: number } | null>(null);

  // Figma panning state
  const [isHandMode, setIsHandMode] = useState(false);
  const velocityRef = useRef({ x: 0, y: 0 });
  const lastMousePosRef = useRef({ x: 0, y: 0, time: 0 });
  const inertiaFrameRef = useRef<number | null>(null);

  // Keyboard spacebar listener for Hand Mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat) {
        const tag = (e.target as HTMLElement).tagName;
        if (tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
          e.preventDefault();
          setIsHandMode(true);
        }
      }
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        setIsHandMode(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      if (inertiaFrameRef.current) cancelAnimationFrame(inertiaFrameRef.current);
    };
  }, []);

  // Node drag/resize
  const [dragState, setDragState] = useState<{ nodeId: string; offsetX: number; offsetY: number } | null>(null);
  const [resizeState, setResizeState] = useState<{
    nodeId: string; handle: RHandle; startCX: number; startCY: number;
    startX: number; startY: number; startW: number; startH: number;
  } | null>(null);

  // Interaction state
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isDragOver, setIsDragOver] = useState(false);
  const [dropPreviewPos, setDropPreviewPos] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [overlappingIds, setOverlappingIds] = useState<Set<string>>(new Set());
  const [snapGuides, setSnapGuides] = useState<SnapGuide[]>([]);

  // Connection state
  const [selectedConnId, setSelectedConnId] = useState<string | null>(null);
  const [hoveredConnId, setHoveredConnId] = useState<string | null>(null);
  const [editingConnId, setEditingConnId] = useState<string | null>(null);
  const [editWeightVal, setEditWeightVal] = useState('');
  const [weightInputErr, setWeightInputErr] = useState(false);

  // Validation toasts
  const [toasts, setToasts] = useState<ToastMsg[]>([]);
  const toastIdRef = useRef(0);

  const showToast = useCallback((text: string, kind: ToastMsg['kind'] = 'error') => {
    const id = ++toastIdRef.current;
    setToasts(t => [...t.slice(-2), { id, text, kind }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 2500);
  }, []);

  // Coord helpers
  const toCanvasCoords = useCallback((cx: number, cy: number) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: (cx - rect.left - pan.x) / zoom, y: (cy - rect.top - pan.y) / zoom };
  }, [pan, zoom]);

  // Screen position of a canvas-space point (for floating input overlay)
  const canvasPtToDiv = useCallback((cx: number, cy: number) => ({
    x: cx * zoom + pan.x,
    y: cy * zoom + pan.y,
  }), [pan, zoom]);

  // Node drag start
  const handleNodePointerDown = useCallback((e: React.PointerEvent, nodeId: string) => {
    if (connectingFrom || resizeState || isHandMode) return;
    e.stopPropagation();
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;
    const coords = toCanvasCoords(e.clientX, e.clientY);
    onNodeDragStart();
    setDragState({ nodeId, offsetX: coords.x - node.x, offsetY: coords.y - node.y });
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }, [nodes, connectingFrom, resizeState, isHandMode, toCanvasCoords, onNodeDragStart]);

  // Resize start
  const handleResizeStart = useCallback((e: React.PointerEvent, nodeId: string, handle: RHandle) => {
    e.stopPropagation();
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;
    const { w, h } = getNodeDims(node);
    onResizeNodeStart();
    setResizeState({ nodeId, handle, startCX: e.clientX, startCY: e.clientY, startX: node.x, startY: node.y, startW: w, startH: h });
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }, [nodes, onResizeNodeStart]);

  // Pointer move
  const handleCanvasPointerMove = useCallback((e: React.PointerEvent) => {
    const coords = toCanvasCoords(e.clientX, e.clientY);
    setMousePos(coords);

    if (dragState) {
      const snX = snapToGrid(coords.x - dragState.offsetX);
      const snY = snapToGrid(coords.y - dragState.offsetY);
      const node = nodes.find(n => n.id === dragState.nodeId);
      if (!node) return;
      const { w, h } = getNodeDims(node);
      setOverlappingIds(getOverlappingIds(dragState.nodeId, snX, snY, w, h, nodes));
      setSnapGuides(computeSnapGuides(dragState.nodeId, snX, snY, w, h, nodes));
      onNodesChange(nodes.map(n => n.id === dragState.nodeId ? { ...n, x: snX, y: snY } : n));
      return;
    }

    if (resizeState) {
      const dx = (e.clientX - resizeState.startCX) / zoom;
      const dy = (e.clientY - resizeState.startCY) / zoom;
      const { handle, startX, startY, startW, startH } = resizeState;
      let [nx, ny, nw, nh] = [startX, startY, startW, startH];
      if (handle.includes('r')) nw = Math.max(MIN_NODE_W, snapToGrid(startW + dx));
      if (handle.includes('b')) nh = Math.max(MIN_NODE_H, snapToGrid(startH + dy));
      if (handle.includes('l')) { const px = snapToGrid(startX + dx); nx = Math.min(px, startX + startW - MIN_NODE_W); nw = startX + startW - nx; }
      if (handle.includes('t')) { const py = snapToGrid(startY + dy); ny = Math.min(py, startY + startH - MIN_NODE_H); nh = startY + startH - ny; }
      onNodesChange(nodes.map(n => n.id === resizeState.nodeId ? { ...n, x: nx, y: ny, w: nw, h: nh } : n));
      return;
    }

    if (isPanning && panStartRef.current) {
      const now = performance.now();
      const dt = now - lastMousePosRef.current.time;
      if (dt > 0 && dt < 100) {
        velocityRef.current = {
          x: (e.clientX - lastMousePosRef.current.x) / Math.max(1, dt / 16.6),
          y: (e.clientY - lastMousePosRef.current.y) / Math.max(1, dt / 16.6),
        };
      }
      lastMousePosRef.current = { x: e.clientX, y: e.clientY, time: now };

      const newPanX = panStartRef.current.px + (e.clientX - panStartRef.current.cx);
      const newPanY = panStartRef.current.py + (e.clientY - panStartRef.current.cy);
      setPan({
        x: Math.max(-10000, Math.min(10000, newPanX)),
        y: Math.max(-10000, Math.min(10000, newPanY))
      });
      return;
    }
  }, [dragState, resizeState, nodes, onNodesChange, toCanvasCoords, isPanning, zoom]);

  // Pointer up
  const handleCanvasPointerUp = useCallback((e: React.PointerEvent) => {
    if (dragState) { setOverlappingIds(new Set()); setSnapGuides([]); setDragState(null); }
    if (resizeState) setResizeState(null);
    if (isPanning) {
      setIsPanning(false);
      panStartRef.current = null;
      try { (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId); } catch { }

      let lastTime = performance.now();
      const tick = (time: number) => {
        const dt = time - lastTime;
        lastTime = time;
        const friction = Math.pow(0.85, dt / 16.6); // 15% decay per 60fps frame
        velocityRef.current.x *= friction;
        velocityRef.current.y *= friction;

        if (Math.abs(velocityRef.current.x) > 0.1 || Math.abs(velocityRef.current.y) > 0.1) {
          setPan(p => ({
            x: Math.max(-10000, Math.min(10000, p.x + velocityRef.current.x * (dt / 16.6))),
            y: Math.max(-10000, Math.min(10000, p.y + velocityRef.current.y * (dt / 16.6)))
          }));
          inertiaFrameRef.current = requestAnimationFrame(tick);
        }
      };
      if (Math.abs(velocityRef.current.x) > 1 || Math.abs(velocityRef.current.y) > 1) {
        inertiaFrameRef.current = requestAnimationFrame(tick);
      }
    }
  }, [dragState, resizeState, isPanning]);

  // Canvas click (deselect / close weight editor)
  const handleCanvasClick = useCallback(() => {
    if (editingConnId) { setEditingConnId(null); return; }
    if (connectingFrom) { onCancelConnecting(); return; }
    onSelectNode(null);
    setSelectedConnId(null);
  }, [editingConnId, connectingFrom, onCancelConnecting, onSelectNode]);

  // Hand mode / Middle-mouse / alt pan
  const handleCanvasPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button === 1 || (e.button === 0 && e.altKey) || (e.button === 0 && isHandMode)) {
      e.preventDefault();
      setIsPanning(true);
      panStartRef.current = { cx: e.clientX, cy: e.clientY, px: pan.x, py: pan.y };
      lastMousePosRef.current = { x: e.clientX, y: e.clientY, time: performance.now() };
      velocityRef.current = { x: 0, y: 0 };
      if (inertiaFrameRef.current) cancelAnimationFrame(inertiaFrameRef.current);
      try { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); } catch { }
    }
  }, [pan, isHandMode]);

  // Wheel zoom (mouse-centred)
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const nz = Math.min(2.5, Math.max(0.25, zoom * factor));
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    setPan(p => ({ x: mx - (mx - p.x) * (nz / zoom), y: my - (my - p.y) * (nz / zoom) }));
    setZoom(nz);
  }, [zoom]);

  // HTML5 drag from sidebar
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    setIsDragOver(true);
    try {
      const raw = e.dataTransfer.getData('application/json');
      if (raw) {
        const data = JSON.parse(raw);
        const dims = NODE_DIMS[data.type as NodeType];
        const c = toCanvasCoords(e.clientX, e.clientY);
        setDropPreviewPos({ x: snapToGrid(c.x - dims.w / 2), y: snapToGrid(c.y - dims.h / 2), w: dims.w, h: dims.h });
      }
    } catch { }
  }, [toCanvasCoords]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    if (!canvasRef.current?.contains(e.relatedTarget as Node)) {
      setIsDragOver(false); setDropPreviewPos(null);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false); setDropPreviewPos(null);
    try {
      const data = JSON.parse(e.dataTransfer.getData('application/json'));
      const dims = NODE_DIMS[data.type as NodeType];
      const c = toCanvasCoords(e.clientX, e.clientY);
      onDropNewNode(data.type, data.label, data.config, snapToGrid(c.x - dims.w / 2), snapToGrid(c.y - dims.h / 2));
    } catch { }
  }, [toCanvasCoords, onDropNewNode]);

  // Port clicks (connection creation with validation)
  const handleOutputPortClick = useCallback((e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation();
    connectingFrom === nodeId ? onCancelConnecting() : onStartConnecting(nodeId);
  }, [connectingFrom, onCancelConnecting, onStartConnecting]);

  const handleInputPortClick = useCallback((e: React.MouseEvent, toId: string) => {
    e.stopPropagation();
    if (!connectingFrom) return;

    // Validation
    if (connectingFrom === toId) {
      showToast('Self-loops are not allowed — a layer cannot connect to itself.', 'error');
      onCancelConnecting();
      return;
    }
    if (connections.some(c => c.fromId === connectingFrom && c.toId === toId)) {
      showToast('This connection already exists.', 'warn');
      onCancelConnecting();
      return;
    }
    if (connections.some(c => c.fromId === toId && c.toId === connectingFrom)) {
      showToast('Reverse connection already exists — cycles require explicit loop nodes.', 'warn');
      // Still allow it — just warn
    }
    onConnectionCreate(connectingFrom, toId);
    onCancelConnecting();
  }, [connectingFrom, connections, onConnectionCreate, onCancelConnecting, showToast]);

  // Connection coords
  const getConnCoords = useCallback((fromId: string, toId: string) => {
    const from = nodes.find(n => n.id === fromId);
    const to = nodes.find(n => n.id === toId);
    if (!from || !to) return null;
    const fd = getNodeDims(from), td = getNodeDims(to);
    const x1 = from.x + fd.w, y1 = from.y + fd.h / 2;
    const x2 = to.x, y2 = to.y + td.h / 2;
    const cp = Math.max(60, Math.abs(x2 - x1) * 0.45);
    return { x1, y1, x2, y2, cp1x: x1 + cp, cp1y: y1, cp2x: x2 - cp, cp2y: y2 };
  }, [nodes]);

  // Weight editing helpers
  const commitWeightEdit = useCallback(() => {
    if (!editingConnId) return;
    const raw = parseFloat(editWeightVal);
    if (isNaN(raw)) { setEditingConnId(null); return; }
    const clamped = Math.max(WEIGHT_MIN, Math.min(WEIGHT_MAX, raw));
    onConnectionWeightChange(editingConnId, Math.round(clamped * 100) / 100);
    setEditingConnId(null);
    setWeightInputErr(false);
  }, [editingConnId, editWeightVal, onConnectionWeightChange]);

  const startWeightEdit = useCallback((connId: string) => {
    const conn = connections.find(c => c.id === connId);
    if (!conn) return;
    setEditingConnId(connId);
    setEditWeightVal(String(conn.weight ?? WEIGHT_DEFAULT));
    setWeightInputErr(false);
  }, [connections]);

  // Compute midpoint for weight label (with perpendicular offset)
  const getConnMid = useCallback((connId: string) => {
    const conn = connections.find(c => c.id === connId);
    if (!conn) return null;
    const c = getConnCoords(conn.fromId, conn.toId);
    if (!c) return null;
    const mid = bezierPt(0.5, c.x1, c.y1, c.cp1x, c.cp1y, c.cp2x, c.cp2y, c.x2, c.y2);
    const { nx, ny } = bezierNormal(0.5, c.x1, c.y1, c.cp1x, c.cp1y, c.cp2x, c.cp2y, c.x2, c.y2);
    // offset 18px along normal, biased upward
    const offsetDir = ny < 0 ? 1 : -1;
    return { x: mid.x + nx * 18 * offsetDir, y: mid.y + ny * 18 * offsetDir };
  }, [connections, getConnCoords]);

  const CVSW = 3200, CVSH = 2200;

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div
      ref={canvasRef}
      className={`relative flex-1 h-full overflow-hidden select-none ${isDragOver ? 'bg-[#EFF6FF]' : ''}
        ${connectingFrom ? 'cursor-crosshair' : isPanning ? 'cursor-grabbing' : isHandMode ? 'cursor-grab' : ''}`}
      style={{ transition: 'background-color 0.15s' }}
      onPointerMove={handleCanvasPointerMove}
      onPointerUp={handleCanvasPointerUp}
      onPointerDown={handleCanvasPointerDown}
      onClick={handleCanvasClick}
      onWheel={handleWheel}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* ── World container (pan + zoom) ────────────────────────────────────── */}
      <div style={{
        transform: `translate(${pan.x}px,${pan.y}px) scale(${zoom})`, transformOrigin: '0 0',
        width: CVSW, height: CVSH, position: 'absolute', top: 0, left: 0
      }}>

        {/* ── SVG layer: grid, connections, guides ─────────────────────────── */}
        <svg width={CVSW} height={CVSH} style={{ position: 'absolute', top: 0, left: 0, overflow: 'visible' }}>
          <defs>
            <pattern id="minor-dots" x="0" y="0" width={GRID} height={GRID} patternUnits="userSpaceOnUse">
              <circle cx={1} cy={1} r={0.8} fill="#C7D7F0" opacity={0.6} />
            </pattern>
            <pattern id="major-dots" x="0" y="0" width={GRID * 5} height={GRID * 5} patternUnits="userSpaceOnUse">
              <rect width={GRID * 5} height={GRID * 5} fill="url(#minor-dots)" />
              <circle cx={1} cy={1} r={1.4} fill="#A8C0E8" opacity={0.8} />
            </pattern>

            {/* Forward arrow markers */}
            {([
              { id: 'arrow-idle', fill: '#93C5FD' },
              { id: 'arrow-active', fill: '#3B82F6' },
              { id: 'arrow-sel', fill: '#F59E0B' },
              { id: 'arrow-hover', fill: '#60A5FA' },
              { id: 'arrow-neg', fill: '#F87171' },
              { id: 'arrow-pend', fill: '#94A3B8' },
              { id: 'arrow-train-fwd', fill: '#3B82F6' },
              { id: 'arrow-train-upd', fill: '#10B981' },
            ] as const).map(({ id, fill }) => (
              <marker key={id} id={id} markerWidth="9" markerHeight="9" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,7 L9,3.5 z" fill={fill} />
              </marker>
            ))}
            {/* Backprop arrow (points LEFT — markerStart usage) */}
            <marker id="arrow-backprop" markerWidth="9" markerHeight="9" refX="1" refY="3.5" orient="auto" markerUnits="strokeWidth">
              <path d="M9,0 L9,7 L0,3.5 z" fill="#F97316" />
            </marker>

            {/* Glow filter for active connections */}
            <filter id="conn-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <filter id="conn-glow-strong" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>

            <pattern id="drop-hatch" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="8" stroke="rgba(59,130,246,0.2)" strokeWidth="4" />
            </pattern>
          </defs>

          {/* Grid */}
          <rect width={CVSW} height={CVSH} fill="url(#major-dots)" />

          {/* Snap alignment guides */}
          {snapGuides.map((g, i) =>
            g.dir === 'h'
              ? <line key={i} x1={g.from} y1={g.pos} x2={g.to} y2={g.pos} stroke="#3B82F6" strokeWidth="1.5" strokeDasharray="5,3" opacity={0.7} style={{ pointerEvents: 'none' }} />
              : <line key={i} x1={g.pos} y1={g.from} x2={g.pos} y2={g.to} stroke="#EF4444" strokeWidth="1.5" strokeDasharray="5,3" opacity={0.7} style={{ pointerEvents: 'none' }} />
          )}

          {/* Drop preview ghost */}
          {isDragOver && dropPreviewPos && (
            <g style={{ pointerEvents: 'none' }}>
              <rect x={dropPreviewPos.x} y={dropPreviewPos.y} width={dropPreviewPos.w} height={dropPreviewPos.h}
                rx={12} ry={12} fill="url(#drop-hatch)" stroke="rgba(59,130,246,0.5)" strokeWidth="2" strokeDasharray="6,3" />
              <line x1={dropPreviewPos.x} y1={dropPreviewPos.y - 16} x2={dropPreviewPos.x} y2={dropPreviewPos.y + dropPreviewPos.h + 16} stroke="rgba(59,130,246,0.3)" strokeWidth="1" strokeDasharray="3,3" />
              <line x1={dropPreviewPos.x - 16} y1={dropPreviewPos.y} x2={dropPreviewPos.x + dropPreviewPos.w + 16} y2={dropPreviewPos.y} stroke="rgba(59,130,246,0.3)" strokeWidth="1" strokeDasharray="3,3" />
            </g>
          )}

          {/* ── Connections ─────────────────────────────────────────────────── */}
          {connections.map(conn => {
            const c = getConnCoords(conn.fromId, conn.toId);
            if (!c) return null;

            const weight = conn.weight ?? WEIGHT_DEFAULT;
            const isActive = conn.active !== false;
            const isSel = conn.id === selectedConnId;
            const isHov = conn.id === hoveredConnId;
            const vis = getWeightVisual(weight, isActive, isSimulating, isSel, isHov);

            const d = `M ${c.x1} ${c.y1} C ${c.cp1x} ${c.cp1y} ${c.cp2x} ${c.cp2y} ${c.x2} ${c.y2}`;

            // Midpoint for weight label
            const mid = bezierPt(0.5, c.x1, c.y1, c.cp1x, c.cp1y, c.cp2x, c.cp2y, c.x2, c.y2);
            const { nx, ny } = bezierNormal(0.5, c.x1, c.y1, c.cp1x, c.cp1y, c.cp2x, c.cp2y, c.x2, c.y2);
            const offsetDir = ny < 0 ? 1 : -1;
            const lx = mid.x + nx * 18 * offsetDir;
            const ly = mid.y + ny * 18 * offsetDir;

            const isEditingThis = editingConnId === conn.id;
            const weightStr = isEditingThis ? editWeightVal : (weight >= 0 ? '+' : '') + weight.toFixed(2);
            const labelW = 52;

            return (
              <g key={conn.id}
                style={{ opacity: vis.opacity }}
                onMouseEnter={() => setHoveredConnId(conn.id)}
                onMouseLeave={() => setHoveredConnId(null)}
              >
                {/* Wide hit-area for click selection */}
                <path d={d} fill="none" stroke="transparent" strokeWidth="20"
                  style={{ cursor: 'pointer', pointerEvents: 'stroke' }}
                  onClick={e => { e.stopPropagation(); setSelectedConnId(isSel ? null : conn.id); setEditingConnId(null); }}
                />

                {/* Hover glow halo */}
                {(isHov || isSel) && (
                  <path d={d} fill="none"
                    stroke={isSel ? 'rgba(245,158,11,0.25)' : 'rgba(59,130,246,0.2)'}
                    strokeWidth={vis.width + 8}
                    style={{ pointerEvents: 'none' }}
                  />
                )}

                {/* Main wire */}
                <path d={d} fill="none"
                  stroke={vis.stroke}
                  strokeWidth={vis.width}
                  strokeDasharray={vis.dashArray}
                  strokeLinecap="round"
                  markerEnd={`url(#${vis.arrowId})`}
                  filter={vis.glowFilter ? `url(#conn-glow${Math.abs(weight) > 0.7 ? '-strong' : ''})` : undefined}
                  style={{ pointerEvents: 'none', transition: 'stroke 0.2s, stroke-width 0.2s' }}
                />

                {/* Direction tick marks (3 evenly spaced small notches) — only when hovered/selected */}
                {(isHov || isSel) && [0.25, 0.5, 0.75].map(t => {
                  const pt = bezierPt(t, c.x1, c.y1, c.cp1x, c.cp1y, c.cp2x, c.cp2y, c.x2, c.y2);
                  const { tx, ty } = bezierTangent(t, c.x1, c.y1, c.cp1x, c.cp1y, c.cp2x, c.cp2y, c.x2, c.y2);
                  return (
                    <line key={t}
                      x1={pt.x - ty * 4} y1={pt.y + tx * 4}
                      x2={pt.x + ty * 4} y2={pt.y - tx * 4}
                      stroke={vis.stroke} strokeWidth={1.5} opacity={0.6}
                      style={{ pointerEvents: 'none' }}
                    />
                  );
                })}

                {/* Particles (simulation) */}
                {isActive && particles.filter(p => p.connectionId === conn.id).map(p => {
                  const pt = bezierPt(p.progress, c.x1, c.y1, c.cp1x, c.cp1y, c.cp2x, c.cp2y, c.x2, c.y2);
                  const pColor = weight < 0 ? '#FCA5A5' : '#BFDBFE';
                  const pCore = weight < 0 ? '#EF4444' : '#3B82F6';
                  return (
                    <g key={p.id} style={{ pointerEvents: 'none' }}>
                      <circle cx={pt.x} cy={pt.y} r={5} fill={pColor} opacity={0.4} />
                      <circle cx={pt.x} cy={pt.y} r={2.5} fill={pCore} />
                      <circle cx={pt.x} cy={pt.y} r={1.2} fill="white" opacity={0.7} />
                    </g>
                  );
                })}

                {/* ── Weight label badge ─────────────────────────────────────── */}
                {!isEditingThis && (
                  <g transform={`translate(${lx},${ly})`}
                    style={{ cursor: 'pointer', pointerEvents: 'all' }}
                    onClick={e => { e.stopPropagation(); setSelectedConnId(conn.id); startWeightEdit(conn.id); }}
                  >
                    {/* Badge background */}
                    <rect x={-labelW / 2} y={-11} width={labelW} height={22} rx={11}
                      fill="white"
                      stroke={isSel ? '#F59E0B' : isHov ? vis.labelColor : '#E2E8F0'}
                      strokeWidth={isSel || isHov ? 2 : 1.5}
                      style={{ filter: isSel ? 'drop-shadow(0 2px 6px rgba(245,158,11,0.3))' : isHov ? `drop-shadow(0 2px 6px ${vis.stroke}44)` : 'drop-shadow(0 1px 3px rgba(0,0,0,0.1))' }}
                    />
                    {/* Inactive indicator stripe */}
                    {!isActive && (
                      <line x1={-labelW / 2 + 8} y1={0} x2={labelW / 2 - 8} y2={0}
                        stroke="#EF4444" strokeWidth={1.5} opacity={0.5} strokeDasharray="3,2" />
                    )}
                    {/* Weight text */}
                    <text textAnchor="middle" dy={4} fontSize={10} fontWeight={700}
                      fontFamily="ui-monospace, monospace"
                      fill={!isActive ? '#94A3B8' : vis.labelColor}>
                      {weightStr}
                    </text>
                    {/* Edit pen icon dot */}
                    <circle cx={labelW / 2 - 6} cy={-6} r={3}
                      fill={isSel || isHov ? vis.labelColor : '#CBD5E1'}
                      opacity={isHov || isSel ? 1 : 0}
                      style={{ transition: 'opacity 0.15s' }}
                    />
                  </g>
                )}
              </g>
            );
          })}

          {/* ── Pending connection (dashed preview) ──────────────────────────── */}
          {(() => {
            if (!connectingFrom) return null;
            const from = nodes.find(n => n.id === connectingFrom);
            if (!from) return null;
            const fd = getNodeDims(from);
            const x1 = from.x + fd.w, y1 = from.y + fd.h / 2;
            const cp = Math.max(40, Math.abs(mousePos.x - x1) * 0.4);
            const d = `M ${x1} ${y1} C ${x1 + cp} ${y1} ${mousePos.x - cp} ${mousePos.y} ${mousePos.x} ${mousePos.y}`;
            return (
              <g style={{ pointerEvents: 'none' }}>
                <path d={d} fill="none" stroke="#94A3B8" strokeWidth={2.5}
                  strokeDasharray="8,5" markerEnd="url(#arrow-pend)" opacity={0.8} />
                <circle cx={mousePos.x} cy={mousePos.y} r={5}
                  fill="none" stroke="#3B82F6" strokeWidth={2} strokeDasharray="3,2" opacity={0.7} />
              </g>
            );
          })()}

          {/* ── Forward propagation: connection wave overlays ─────────────────
               Rendered on top of normal wires so they are always visible     */}
          {(propWaveConnIds || propDoneConnIds) && connections.map(conn => {
            const inWave = propWaveConnIds?.includes(conn.id);
            const inDone = propDoneConnIds?.includes(conn.id);
            if (!inWave && !inDone) return null;
            const c = getConnCoords(conn.fromId, conn.toId);
            if (!c) return null;
            const d = `M ${c.x1} ${c.y1} C ${c.cp1x} ${c.cp1y} ${c.cp2x} ${c.cp2y} ${c.x2} ${c.y2}`;
            const signal = propNodeActivations?.[conn.fromId] ?? 0.5;
            // Colour based on connection weight + signal
            const isNeg = (conn.weight ?? 0.5) < 0;
            const waveColor = isNeg ? '#EF4444' : '#3B82F6';
            const doneColor = isNeg ? '#FCA5A5' : '#93C5FD';
            return (
              <g key={`fp-${conn.id}`} style={{ pointerEvents: 'none' }}>
                {/* Glow halo */}
                {inWave && (
                  <path d={d} fill="none"
                    stroke={waveColor} strokeWidth={10} strokeOpacity={0.18}
                    filter="url(#conn-glow)" />
                )}
                {/* Main overlay path */}
                <path d={d} fill="none"
                  stroke={inWave ? waveColor : doneColor}
                  strokeWidth={inWave ? Math.max(2, 2 + signal * 3) : 2}
                  strokeOpacity={inWave ? 0.95 : 0.45}
                  strokeLinecap="round"
                  strokeDasharray={inDone && !inWave ? '4,4' : undefined}
                  markerEnd={inWave ? `url(#arrow-active)` : undefined}
                  filter={inWave ? 'url(#conn-glow)' : undefined}
                />
                {/* Signal value badge on wave-front */}
                {inWave && (() => {
                  const mid = bezierPt(0.5, c.x1, c.y1, c.cp1x, c.cp1y, c.cp2x, c.cp2y, c.x2, c.y2);
                  return (
                    <g transform={`translate(${mid.x},${mid.y - 14})`}>
                      <rect x={-18} y={-9} width={36} height={18} rx={9}
                        fill="white" stroke={waveColor} strokeWidth={1.5}
                        style={{ filter: `drop-shadow(0 2px 6px ${waveColor}44)` }} />
                      <text textAnchor="middle" dy={4} fontSize={9} fontWeight={700}
                        fontFamily="ui-monospace,monospace" fill={waveColor}>
                        {signal.toFixed(2)}
                      </text>
                    </g>
                  );
                })()}
              </g>
            );
          })}
          {/* ── Training mode overlays ──────────────────────────────────────
               Forward pass (blue), backprop (orange←), weight update (green)  */}
          {trainPhase && trainPhase !== 'idle' && trainActiveConnIds && connections.map(conn => {
            if (!trainActiveConnIds.includes(conn.id)) return null;
            const c = getConnCoords(conn.fromId, conn.toId);
            if (!c) return null;
            const d = `M ${c.x1} ${c.y1} C ${c.cp1x} ${c.cp1y} ${c.cp2x} ${c.cp2y} ${c.x2} ${c.y2}`;
            const grad = trainGradients?.[conn.id] ?? 0.5;
            const normG = Math.max(0.1, Math.min(1, grad / (Math.max(...Object.values(trainGradients ?? {}), 1e-9))));

            if (trainPhase === 'forward') {
              return (
                <g key={`tr-f-${conn.id}`} style={{ pointerEvents: 'none' }}>
                  <path d={d} fill="none" stroke="#3B82F6" strokeWidth={2 + normG * 3}
                    strokeOpacity={0.7 + normG * 0.25} strokeLinecap="round"
                    markerEnd="url(#arrow-train-fwd)"
                    filter="url(#conn-glow)" />
                </g>
              );
            }

            if (trainPhase === 'backprop') {
              // Orange reversed arrows — markerStart points LEFT
              return (
                <g key={`tr-b-${conn.id}`} style={{ pointerEvents: 'none' }}>
                  {/* Glow halo */}
                  <path d={d} fill="none" stroke="#F97316" strokeWidth={normG * 12} strokeOpacity={0.15}
                    filter="url(#conn-glow)" />
                  {/* Main orange reverse arrow */}
                  <path d={d} fill="none" stroke="#F97316" strokeWidth={2 + normG * 2.5}
                    strokeOpacity={0.85} strokeLinecap="round"
                    markerStart="url(#arrow-backprop)"
                    strokeDasharray={`${6 + normG * 4},${3}`} />
                  {/* Gradient magnitude label at midpoint */}
                  {normG > 0.4 && (() => {
                    const mid = bezierPt(0.5, c.x1, c.y1, c.cp1x, c.cp1y, c.cp2x, c.cp2y, c.x2, c.y2);
                    return (
                      <g transform={`translate(${mid.x},${mid.y + 16})`}>
                        <rect x={-18} y={-9} width={36} height={18} rx={9}
                          fill="#FFF7ED" stroke="#F97316" strokeWidth={1.5}
                          style={{ filter: 'drop-shadow(0 2px 4px rgba(249,115,22,0.3))' }} />
                        <text textAnchor="middle" dy={4} fontSize={9} fontWeight={700}
                          fontFamily="ui-monospace,monospace" fill="#EA580C">
                          ∇{normG.toFixed(2)}
                        </text>
                      </g>
                    );
                  })()}
                </g>
              );
            }

            if (trainPhase === 'update') {
              const delta = trainWeightDeltas?.[conn.id] ?? 0;
              if (Math.abs(delta) < 0.0005) return null;
              const isPos = delta > 0;
              const color = isPos ? '#10B981' : '#EF4444';
              const mid = bezierPt(0.5, c.x1, c.y1, c.cp1x, c.cp1y, c.cp2x, c.cp2y, c.x2, c.y2);
              const { nx, ny } = bezierNormal(0.5, c.x1, c.y1, c.cp1x, c.cp1y, c.cp2x, c.cp2y, c.x2, c.y2);
              const offsetDir = ny < 0 ? 1 : -1;
              const lx = mid.x + nx * 18 * offsetDir;
              const ly = mid.y + ny * 18 * offsetDir;
              return (
                <g key={`tr-u-${conn.id}`} style={{ pointerEvents: 'none' }}>
                  {/* Wire flash */}
                  <path d={d} fill="none" stroke={color} strokeWidth={3.5}
                    strokeOpacity={0.6} markerEnd="url(#arrow-train-upd)"
                    filter="url(#conn-glow)" />
                  {/* Delta badge */}
                  <g transform={`translate(${lx},${ly})`}>
                    <rect x={-26} y={-11} width={52} height={22} rx={11}
                      fill={color} opacity={0.92}
                      style={{ filter: `drop-shadow(0 2px 8px ${color}66)` }} />
                    <text textAnchor="middle" dy={4} fontSize={10} fontWeight={700}
                      fontFamily="ui-monospace,monospace" fill="white">
                      {isPos ? '+' : ''}{delta.toFixed(4)}
                    </text>
                  </g>
                </g>
              );
            }

            return null;
          })}
        </svg>

        {/* ── Canvas nodes ─────────────────────────────────────────────────── */}
        {nodes.map(node => (
          <CanvasNodeComponent
            key={node.id}
            node={node}
            isSelected={node.id === selectedNodeId}
            isDragging={dragState?.nodeId === node.id}
            isConnectingFrom={node.id === connectingFrom}
            isConnecting={!!connectingFrom && node.id !== connectingFrom}
            isOverlapping={overlappingIds.has(node.id)}
            isActive={isSimulating}
            propActivation={propNodeActivations?.[node.id]}
            inPropWave={propWaveNodeIds?.includes(node.id)}
            inPropDone={propDoneNodeIds?.includes(node.id)}
            onPointerDown={handleNodePointerDown}
            onResizeStart={handleResizeStart}
            onOutputPortClick={handleOutputPortClick}
            onInputPortClick={handleInputPortClick}
            onSelect={onSelectNode}
            onDelete={onDeleteNode}
            onDuplicate={onDuplicateNode}
            onConfigChange={onConfigChange}
          />
        ))}

        {/* Multiplayer cursors */}
        {MOCK_CURSORS.map(cur => (
          <div key={cur.id} style={{ position: 'absolute', left: cur.x, top: cur.y, pointerEvents: 'none', zIndex: 200 }}>
            <svg width="20" height="20" viewBox="0 0 20 20" style={{ filter: 'drop-shadow(0 1px 3px rgba(0,0,0,0.25))' }}>
              <path d="M0 0 L0 14 L4 10 L7 16 L9 15 L6 9 L11 9 Z" fill={cur.color} stroke="white" strokeWidth="1.2" />
            </svg>
            <div className="mt-0.5 ml-3.5 px-2 py-0.5 rounded-full text-white text-[10px] font-semibold shadow-sm whitespace-nowrap"
              style={{ backgroundColor: cur.color }}>{cur.name}</div>
          </div>
        ))}
      </div>

      {/* ── Weight edit floating input ─────────────────────────────────────────
          Positioned in canvas-div space using the computed midpoint coords     */}
      {editingConnId && (() => {
        const mid = getConnMid(editingConnId);
        if (!mid) return null;
        const sp = canvasPtToDiv(mid.x, mid.y);
        const conn = connections.find(c => c.id === editingConnId);
        const vis = conn ? getWeightVisual(conn.weight ?? WEIGHT_DEFAULT, conn.active !== false, isSimulating, true, false) : null;
        return (
          <div
            style={{ position: 'absolute', left: sp.x, top: sp.y, transform: 'translate(-50%,-50%)', zIndex: 200, pointerEvents: 'all' }}
            onPointerDown={e => e.stopPropagation()}
            onClick={e => e.stopPropagation()}
          >
            <div className="bg-white rounded-2xl shadow-2xl border-2 p-3 w-44"
              style={{
                borderColor: weightInputErr ? '#EF4444' : vis?.stroke ?? '#3B82F6',
                boxShadow: `0 8px 32px ${vis?.stroke ?? '#3B82F6'}33, 0 0 0 3px ${vis?.stroke ?? '#3B82F6'}22`
              }}>

              <div className="flex items-center gap-1.5 mb-2">
                <Edit3 size={11} className="text-slate-400 flex-shrink-0" />
                <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide">Synaptic Weight</span>
              </div>

              <input
                type="number" step="0.01" min={WEIGHT_MIN} max={WEIGHT_MAX}
                value={editWeightVal}
                autoFocus
                onChange={e => {
                  setEditWeightVal(e.target.value);
                  const v = parseFloat(e.target.value);
                  setWeightInputErr(!isNaN(v) && (v < WEIGHT_MIN || v > WEIGHT_MAX));
                }}
                onKeyDown={e => {
                  if (e.key === 'Enter') commitWeightEdit();
                  if (e.key === 'Escape') { setEditingConnId(null); setWeightInputErr(false); }
                  e.stopPropagation();
                }}
                onBlur={commitWeightEdit}
                className="w-full px-3 py-1.5 text-[13px] font-mono font-bold rounded-lg border focus:outline-none focus:ring-2 transition-all text-center"
                style={{
                  borderColor: weightInputErr ? '#EF4444' : '#E2E8F0',
                  color: parseFloat(editWeightVal) < 0 ? '#DC2626' : '#1D4ED8',
                  focusRing: 'none'
                }}
              />

              {/* Weight range mini-slider for quick adjustment */}
              <input type="range" min={WEIGHT_MIN} max={WEIGHT_MAX} step="0.05"
                value={parseFloat(editWeightVal) || 0}
                onChange={e => { setEditWeightVal(e.target.value); setWeightInputErr(false); }}
                className="w-full mt-2 accent-blue-500"
                style={{ accentColor: vis?.stroke ?? '#3B82F6' }}
              />

              <div className="flex justify-between text-[9px] text-slate-400 mt-0.5 font-mono">
                <span>−2.0</span>
                <span className="text-[9px]" style={{ color: parseFloat(editWeightVal) < 0 ? '#EF4444' : '#3B82F6' }}>
                  {parseFloat(editWeightVal) >= 0 ? '+' : ''}{(parseFloat(editWeightVal) || 0).toFixed(2)}
                </span>
                <span>+2.0</span>
              </div>

              {weightInputErr && (
                <div className="mt-1.5 text-[10px] text-red-500 flex items-center gap-1">
                  <span>⚠</span> Range: −2.0 to +2.0
                </div>
              )}

              <div className="mt-2 flex gap-1">
                <button onClick={commitWeightEdit}
                  className="flex-1 py-1 text-[10px] bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium">
                  Apply ↵
                </button>
                <button onClick={() => { setEditingConnId(null); setWeightInputErr(false); }}
                  className="px-2 py-1 text-[10px] bg-slate-100 text-slate-500 rounded-lg hover:bg-slate-200 transition-colors">
                  ✕
                </button>
              </div>
            </div>
          </div>
        );
      })()}

      {/* ── Connection selected toolbar ────────────────────────────────────────
           Shown at top-centre when a connection is selected                   */}
      {selectedConnId && !editingConnId && (() => {
        const conn = connections.find(c => c.id === selectedConnId);
        if (!conn) return null;
        const weight = conn.weight ?? WEIGHT_DEFAULT;
        const isActive = conn.active !== false;
        const vis = getWeightVisual(weight, isActive, isSimulating, true, false);
        return (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 bg-white rounded-2xl px-3 py-2 shadow-xl border border-slate-200"
            style={{ boxShadow: '0 8px 24px rgba(0,0,0,0.10),0 0 0 1px rgba(0,0,0,0.04)' }}>

            {/* Weight swatch */}
            <div className="flex items-center gap-1.5 pr-2 border-r border-slate-100">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: vis.stroke }} />
              <span className="text-[11px] font-mono font-bold" style={{ color: vis.labelColor }}>
                {weight >= 0 ? '+' : ''}{weight.toFixed(2)}
              </span>
              <button
                onClick={e => { e.stopPropagation(); startWeightEdit(selectedConnId); }}
                className="w-5 h-5 rounded-md flex items-center justify-center text-slate-400 hover:text-blue-500 hover:bg-blue-50 transition-all"
                title="Edit weight">
                <Edit3 size={10} />
              </button>
            </div>

            {/* Active toggle */}
            <button
              onClick={e => { e.stopPropagation(); onConnectionActiveToggle(selectedConnId); }}
              className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-[11px] font-medium transition-all ${isActive ? 'bg-green-50 text-green-700 hover:bg-green-100 border border-green-200'
                : 'bg-slate-100 text-slate-500 hover:bg-slate-200 border border-slate-200'
                }`}
              title={isActive ? 'Disable connection' : 'Enable connection'}
            >
              {isActive ? <Power size={10} /> : <PowerOff size={10} />}
              {isActive ? 'Active' : 'Inactive'}
            </button>

            {/* Delete */}
            <button
              onClick={e => { e.stopPropagation(); onConnectionDelete(selectedConnId); setSelectedConnId(null); }}
              className="flex items-center gap-1 px-2.5 py-1 bg-red-50 text-red-500 hover:bg-red-100 rounded-lg text-[11px] font-medium transition-colors border border-red-200"
            >
              <Trash2 size={10} /> Delete
            </button>
          </div>
        );
      })()}

      {/* ── Connecting mode banner ─────────────────────────────────────────── */}
      {connectingFrom && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2.5 bg-blue-600 text-white rounded-xl px-5 py-2.5 shadow-lg text-[12px] font-medium">
          <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
          Click any input port (◉ left side) to connect
          <kbd className="bg-blue-500 px-1.5 py-0.5 rounded text-[10px] ml-1">Esc</kbd> cancel
        </div>
      )}

      {/* ── Drop zone border ───────────────────────────────────────────────── */}
      {isDragOver && (
        <div className="absolute inset-2 border-2 border-dashed border-blue-400 rounded-2xl pointer-events-none z-50 flex items-center justify-center">
          <div className="bg-white/95 backdrop-blur-sm rounded-2xl px-6 py-3 text-blue-600 text-[14px] font-semibold shadow-lg border border-blue-200 flex items-center gap-2">
            <span className="text-xl">＋</span> Drop to add layer
          </div>
        </div>
      )}

      {/* ── Overlap warning ────────────────────────────────────────────────── */}
      {overlappingIds.size > 0 && dragState && (
        <div className="absolute top-4 right-4 z-50 flex items-center gap-2 bg-red-50 border border-red-200 text-red-600 rounded-xl px-3 py-2 shadow-sm text-[11px] font-medium">
          ⚠ Overlapping {overlappingIds.size} node{overlappingIds.size > 1 ? 's' : ''}
        </div>
      )}

      {/* ── Validation toasts ──────────────────────────────────────────────── */}
      <div className="absolute bottom-16 left-1/2 -translate-x-1/2 z-50 flex flex-col items-center gap-2 pointer-events-none">
        {toasts.map(t => (
          <div key={t.id} className={`px-4 py-2.5 rounded-xl shadow-lg text-[12px] font-medium flex items-center gap-2 backdrop-blur-sm border
            ${t.kind === 'error'
              ? 'bg-red-50/95 text-red-700 border-red-200'
              : 'bg-amber-50/95 text-amber-700 border-amber-200'
            }`}
            style={{ animation: 'fadeInUp 0.2s ease' }}>
            <span>{t.kind === 'error' ? '⛔' : '⚠️'}</span>
            {t.text}
          </div>
        ))}
      </div>

      {/* ── Zoom controls ─────────────────────────────────────────────────── */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-1 z-50">
        {[
          { label: '+', action: () => setZoom(v => Math.min(2.5, v * 1.2)), title: 'Zoom in' },
          { label: '−', action: () => setZoom(v => Math.max(0.25, v / 1.2)), title: 'Zoom out' },
        ].map(({ label, action, title }) => (
          <button key={label} onClick={action} title={title}
            className="w-8 h-8 bg-white border border-slate-200 rounded-lg text-slate-600 hover:bg-slate-50 shadow-sm text-lg flex items-center justify-center transition-all hover:shadow">
            {label}
          </button>
        ))}
        <div className="bg-white border border-slate-200 rounded-lg px-1 py-1 text-[11px] text-slate-500 shadow-sm text-center font-mono">{Math.round(zoom * 100)}%</div>
        <button onClick={() => { setZoom(1); setPan({ x: 40, y: 40 }); }}
          className="w-8 h-8 bg-white border border-slate-200 rounded-lg text-slate-500 hover:bg-slate-50 shadow-sm text-[10px] flex items-center justify-center" title="Reset view">⌖</button>
        <button onClick={() => {
          if (nodes.length === 0) return;
          const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
          setPan({ x: 80 - Math.min(...xs) * zoom, y: 80 - Math.min(...ys) * zoom });
        }} className="w-8 h-8 bg-white border border-slate-200 rounded-lg text-slate-500 hover:bg-blue-50 hover:border-blue-200 shadow-sm text-[10px] flex items-center justify-center" title="Fit to nodes">⊞</button>
      </div>

      {/* ── Empty-canvas hint ─────────────────────────────────────────────── */}
      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center space-y-4">
            <div className="text-8xl opacity-10 select-none">🧠</div>
            <div className="text-slate-400 text-[15px] font-medium">Drag layers from the sidebar to get started</div>
            <div className="text-slate-300 text-[12px]">Or pick a template from the header</div>
          </div>
        </div>
      )}

      {/* ── Hand Mode Indicator ────────────────────────────────────────────── */}
      <div className={`absolute bottom-6 left-1/2 -translate-x-1/2 bg-slate-800 text-white px-3 py-1.5 rounded-full text-[11px] font-medium shadow-lg pointer-events-none transition-all duration-300 z-50 ${isHandMode ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}`}>
        👋 Hand Mode activated · Drag to pan
      </div>
    </div>
  );
}
