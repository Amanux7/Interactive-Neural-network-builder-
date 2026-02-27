import React, { useState, useRef, useCallback } from 'react';
import {
  Search, Layers, Zap, Shield, BarChart3,
  ChevronDown, ChevronRight, X,
} from 'lucide-react';
import { COMPONENT_LIBRARY, NODE_STYLE, NODE_DIMS, ComponentTemplate, NodeType } from './types';

const NODE_ICONS: Record<NodeType, string> = {
  input: 'IN', dense: 'FC', activation: 'fn', dropout: 'DO',
  output: 'OUT', conv2d: '2D', flatten: 'FT', lstm: 'RNN',
  batchnorm: 'BN', embedding: 'EM',
};

const CATEGORY_META: Record<string, { icon: React.ReactNode; desc: string }> = {
  Layers:         { icon: <Layers size={11} />,      desc: 'Core building blocks' },
  Activations:    { icon: <Zap size={11} />,          desc: 'Non-linear functions' },
  Regularization: { icon: <Shield size={11} />,       desc: 'Prevent overfitting' },
  Sequence:       { icon: <BarChart3 size={11} />,    desc: 'Temporal / recurrent' },
};

// ── Draggable Sidebar Card ────────────────────────────────────────────────────
function SidebarItem({ template }: { template: ComponentTemplate }) {
  const style = NODE_STYLE[template.type];
  const dims  = NODE_DIMS[template.type];
  const [isDragging, setIsDragging] = useState(false);
  const ghostRef = useRef<HTMLDivElement | null>(null);

  const handleDragStart = useCallback((e: React.DragEvent) => {
    setIsDragging(true);
    // Rich ghost image
    const ghost = document.createElement('div');
    ghost.style.cssText = `
      position:fixed; top:-9999px; left:-9999px;
      width:${dims.w}px; height:${dims.h}px;
      background:white;
      border:2.5px solid ${style.borderHex};
      border-radius:14px;
      box-shadow:0 24px 48px rgba(59,130,246,0.3),0 0 0 4px rgba(59,130,246,0.12);
      display:flex; flex-direction:column; align-items:center; justify-content:center; gap:5px;
      font-family:'Inter',system-ui,sans-serif;
      pointer-events:none; opacity:0.96;
    `;
    ghost.innerHTML = `
      <div style="width:30px;height:30px;border-radius:9px;background:linear-gradient(135deg,${style.dot},${style.dot}cc);display:flex;align-items:center;justify-content:center;color:white;font-size:10px;font-weight:700">${NODE_ICONS[template.type]}</div>
      <div style="font-size:11px;font-weight:600;color:#1E293B">${template.label}</div>
    `;
    document.body.appendChild(ghost);
    ghostRef.current = ghost;
    e.dataTransfer.setDragImage(ghost, dims.w / 2, dims.h / 2);
    setTimeout(() => {
      if (ghostRef.current) { document.body.removeChild(ghostRef.current); ghostRef.current = null; }
    }, 100);
    e.dataTransfer.setData('application/json', JSON.stringify({
      type: template.type, label: template.label, config: template.defaultConfig,
    }));
    e.dataTransfer.effectAllowed = 'copy';
  }, [style, dims, template]);

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onDragEnd={() => setIsDragging(false)}
      role="listitem"
      aria-label={`${template.label} — ${template.description}. Drag to add.`}
      tabIndex={0}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') e.preventDefault(); }}
      className={`
        group relative flex items-center gap-3 px-3 py-2.5
        rounded-xl border-2 cursor-grab active:cursor-grabbing
        select-none outline-none
        transition-all duration-150
        focus-visible:ring-2 focus-visible:ring-blue-400/60
        ${style.bg} ${style.border} ${style.text}
        ${isDragging ? 'opacity-35 scale-95' : 'opacity-100 hover:scale-[1.015]'}
      `}
      style={{
        boxShadow: isDragging ? 'none' : '0 1px 3px rgba(0,0,0,0.05)',
        transition: 'all 0.15s cubic-bezier(0.2,0,0.2,1)',
      }}
      onMouseEnter={e => {
        if (!isDragging) (e.currentTarget as HTMLElement).style.boxShadow =
          `0 0 0 3px ${style.glowColor}, 0 6px 20px ${style.glowColor.replace('0.35','0.12')}`;
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.boxShadow = '0 1px 3px rgba(0,0,0,0.05)';
      }}
    >
      {/* Grip dots */}
      <div className="flex flex-col gap-[3px] opacity-20 group-hover:opacity-50 transition-opacity flex-shrink-0" aria-hidden="true">
        {[0,1,2].map(r => (
          <div key={r} className="flex gap-[3px]">
            {[0,1].map(c => <div key={c} className="w-[3px] h-[3px] rounded-full bg-current" />)}
          </div>
        ))}
      </div>

      {/* Badge */}
      <div
        className="w-7 h-7 rounded-lg flex items-center justify-center text-white text-[10px] font-bold flex-shrink-0 shadow-sm nf-bevel"
        style={{ background: `linear-gradient(135deg, ${style.dot}, ${style.dot}bb)` }}
        aria-hidden="true"
      >
        {NODE_ICONS[template.type]}
      </div>

      {/* Text */}
      <div className="flex-1 min-w-0">
        <div className="text-[12px] font-semibold truncate leading-tight">{template.label}</div>
        <div className="text-[10px] opacity-50 truncate mt-0.5 leading-tight">{template.description}</div>
      </div>

      {/* Size chip */}
      <div className="text-[8.5px] opacity-25 font-mono flex-shrink-0 tabular-nums">{dims.w}×{dims.h}</div>
    </div>
  );
}

// ── Main Sidebar ──────────────────────────────────────────────────────────────
interface SidebarProps {
  isMobileOpen?: boolean;
  onMobileClose?: () => void;
}

export function Sidebar({ isMobileOpen = true, onMobileClose }: SidebarProps) {
  const [search, setSearch] = useState('');
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const searchRef = useRef<HTMLInputElement>(null);
  const categories = ['Layers', 'Activations', 'Regularization', 'Sequence'] as const;

  const filtered = COMPONENT_LIBRARY.filter(c =>
    c.label.toLowerCase().includes(search.toLowerCase()) ||
    c.description.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <aside
      role="complementary"
      aria-label="Component library"
      className={`
        flex flex-col w-full h-full
        bg-white dark:bg-slate-900
        border-r border-slate-200/80 dark:border-slate-700/60
        overflow-hidden
        transition-transform duration-300
      `}
    >
      {/* ── Header ── */}
      <div className="px-4 pt-4 pb-3 border-b border-slate-100 dark:border-slate-700/60 flex-shrink-0">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-sm nf-bevel flex-shrink-0">
            <Layers size={12} className="text-white" aria-hidden="true" />
          </div>
          <h2 className="text-[13px] font-semibold text-slate-700 dark:text-slate-200 flex-1">
            Components
          </h2>
          <span
            aria-label={`${COMPONENT_LIBRARY.length} components available`}
            className="text-[10px] bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 px-1.5 py-0.5 rounded-full border border-blue-100 dark:border-blue-800/60 font-semibold tabular-nums"
          >
            {COMPONENT_LIBRARY.length}
          </span>
          {/* Mobile close */}
          {onMobileClose && (
            <button
              onClick={onMobileClose}
              className="ml-1 w-6 h-6 rounded-lg flex items-center justify-center text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
              aria-label="Close component library"
            >
              <X size={13} />
            </button>
          )}
        </div>

        {/* Search */}
        <div className="relative">
          <Search
            size={12}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400 dark:text-slate-500 pointer-events-none"
            aria-hidden="true"
          />
          <input
            ref={searchRef}
            type="search"
            placeholder="Search components…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            aria-label="Search components"
            className={`
              w-full pl-8 pr-8 py-2 text-[12px]
              bg-slate-50 dark:bg-slate-800
              border border-slate-200 dark:border-slate-600/60
              text-slate-700 dark:text-slate-200
              placeholder:text-slate-400 dark:placeholder:text-slate-500
              rounded-xl outline-none
              focus:border-blue-400 dark:focus:border-blue-500
              focus:bg-white dark:focus:bg-slate-750
              focus:ring-2 focus:ring-blue-100 dark:focus:ring-blue-900/50
              transition-all duration-150
            `}
          />
          {search && (
            <button
              onClick={() => { setSearch(''); searchRef.current?.focus(); }}
              aria-label="Clear search"
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
            >
              <X size={12} />
            </button>
          )}
        </div>
      </div>

      {/* ── Component list ── */}
      <div
        role="list"
        aria-label="Available components"
        className="flex-1 overflow-y-auto px-3 py-3 space-y-0.5 overscroll-contain"
      >
        {search ? (
          <div className="space-y-1.5">
            {filtered.length > 0
              ? filtered.map((t, i) => <SidebarItem key={`${t.type}-${i}`} template={t} />)
              : (
                <div role="status" className="text-center py-12">
                  <div className="text-2xl mb-2">🔍</div>
                  <p className="text-[12px] text-slate-400 dark:text-slate-500">
                    No results for "<strong>{search}</strong>"
                  </p>
                  <button
                    onClick={() => setSearch('')}
                    className="mt-2 text-[11px] text-blue-500 hover:underline"
                  >
                    Clear search
                  </button>
                </div>
              )
            }
          </div>
        ) : (
          categories.map(category => {
            const items  = COMPONENT_LIBRARY.filter(c => c.category === category);
            const meta   = CATEGORY_META[category];
            const isOpen = !collapsed[category];
            return (
              <div key={category} className="mb-1">
                <button
                  onClick={() => setCollapsed(p => ({ ...p, [category]: !p[category] }))}
                  aria-expanded={isOpen}
                  aria-controls={`cat-${category}`}
                  className={`
                    flex items-center gap-2 w-full px-2.5 py-2 rounded-xl
                    text-[11px] font-semibold uppercase tracking-wider
                    text-slate-500 dark:text-slate-400
                    hover:text-slate-700 dark:hover:text-slate-200
                    hover:bg-slate-50 dark:hover:bg-slate-800/60
                    transition-all duration-150 outline-none
                    focus-visible:ring-2 focus-visible:ring-blue-400/60
                  `}
                >
                  <span className="flex items-center text-slate-400 dark:text-slate-500" aria-hidden="true">
                    {meta.icon}
                  </span>
                  <span className="flex-1 text-left">{category}</span>
                  <span className="text-[9.5px] bg-slate-100 dark:bg-slate-700/70 text-slate-500 dark:text-slate-400 px-1.5 py-0.5 rounded-full font-normal tabular-nums">
                    {items.length}
                  </span>
                  <span className="text-slate-300 dark:text-slate-600" aria-hidden="true">
                    {isOpen ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
                  </span>
                </button>

                {isOpen && (
                  <div
                    id={`cat-${category}`}
                    className="space-y-1.5 mt-1 pl-0.5"
                  >
                    {items.map((t, i) => <SidebarItem key={`${t.type}-${i}`} template={t} />)}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* ── Footer tips ── */}
      <div className="px-4 py-3 border-t border-slate-100 dark:border-slate-700/60 bg-gradient-to-b from-white dark:from-slate-900 to-slate-50/80 dark:to-slate-900/80 flex-shrink-0">
        <p className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-2">
          Quick Guide
        </p>
        <div className="space-y-1.5 text-[10px] text-slate-400 dark:text-slate-500">
          {[
            { icon: '⬇', color: 'bg-blue-100 dark:bg-blue-900/40 text-blue-500 dark:text-blue-400',  text: 'Drag components onto the canvas' },
            { icon: '○', color: 'bg-green-100 dark:bg-green-900/40 text-green-500 dark:text-green-400', text: 'Right port → left port to connect' },
            { icon: '⌦', color: 'bg-slate-100 dark:bg-slate-700/50 text-slate-500 dark:text-slate-400', text: 'Select + Delete to remove a node' },
          ].map(({ icon, color, text }) => (
            <div key={text} className="flex items-center gap-2">
              <span className={`w-4 h-4 rounded-md ${color} flex items-center justify-center text-[9px] flex-shrink-0`} aria-hidden="true">
                {icon}
              </span>
              <span>{text}</span>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
