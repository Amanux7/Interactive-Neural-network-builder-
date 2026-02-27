import React, { memo } from 'react';
import {
  Play, Pause, Square, SkipForward, Zap, TrendingDown,
  Target, Activity, ChevronUp, ChevronDown, GraduationCap,
} from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, Tooltip } from 'recharts';
import { SimulationState } from './types';

interface SimulationBarProps {
  simulation: SimulationState;
  onStart: () => void;
  onPause: () => void;
  onStop: () => void;
  onStep: () => void;
  onSpeedChange: (speed: number) => void;
  onEpochsChange: (epochs: number) => void;
  nodeCount: number;
  connectionCount: number;
  onRunForwardPass: () => void;
  fwdPropOpen: boolean;
  fwdIsRunning: boolean;
  fwdIsComplete: boolean;
  onOpenTraining: () => void;
  trainOpen: boolean;
  trainIsRunning: boolean;
  trainIsComplete: boolean;
}

// ── Metric badge ──────────────────────────────────────────────────────────────
const MetricBadge = memo(function MetricBadge({
  label, value, colorClass, icon, ariaLabel,
}: {
  label: string;
  value: string;
  colorClass: string;
  icon: React.ReactNode;
  ariaLabel: string;
}) {
  return (
    <div
      role="status"
      aria-label={ariaLabel}
      className={`
        flex items-center gap-2 px-3 py-2 rounded-xl
        bg-white dark:bg-slate-800
        border dark:border-slate-700/60 ${colorClass}
        shadow-[var(--shadow-xs)]
        min-w-[108px] flex-shrink-0
      `}
    >
      <div aria-hidden="true">{icon}</div>
      <div>
        <div className="text-[9.5px] text-slate-400 dark:text-slate-500 uppercase tracking-wider font-medium leading-none mb-0.5">
          {label}
        </div>
        <div className="text-[13.5px] font-semibold text-slate-700 dark:text-slate-200 tabular-nums leading-none">
          {value}
        </div>
      </div>
    </div>
  );
});

// ── Divider ───────────────────────────────────────────────────────────────────
function Divider() {
  return <div className="h-10 w-px bg-slate-100 dark:bg-slate-700/60 flex-shrink-0 hidden sm:block" aria-hidden="true" />;
}

// ── Main bar ──────────────────────────────────────────────────────────────────
export const SimulationBar = memo(function SimulationBar({
  simulation, onStart, onPause, onStop, onStep, onSpeedChange, onEpochsChange,
  nodeCount, connectionCount,
  onRunForwardPass, fwdPropOpen, fwdIsRunning, fwdIsComplete,
  onOpenTraining, trainOpen, trainIsRunning, trainIsComplete,
}: SimulationBarProps) {
  const { isRunning, isPaused, epoch, totalEpochs, loss, accuracy, speed, history } = simulation;
  const progress   = totalEpochs > 0 ? (epoch / totalEpochs) * 100 : 0;
  const isComplete = epoch >= totalEpochs && totalEpochs > 0;
  const speedOptions = [0.5, 1, 2, 5, 10];

  const statusText  = isRunning && !isPaused ? 'Training'
    : isComplete ? 'Complete' : isPaused ? 'Paused' : 'Idle';
  const statusColor = isRunning && !isPaused
    ? 'bg-green-400 animate-pulse'
    : isComplete ? 'bg-teal-400' : 'bg-slate-300 dark:bg-slate-600';

  return (
    <div
      role="toolbar"
      aria-label="Simulation controls"
      className="
        bg-white dark:bg-slate-900
        border-t border-slate-200 dark:border-slate-700/60
        shadow-[0_-2px_12px_rgba(0,0,0,0.06)] dark:shadow-[0_-2px_12px_rgba(0,0,0,0.25)]
        flex-shrink-0
      "
    >
      {/* Scrollable inner row */}
      <div className="h-[88px] flex items-center px-3 gap-3 overflow-x-auto overscroll-x-contain">

        {/* ── Transport controls ── */}
        <div className="flex items-center gap-1.5 flex-shrink-0" role="group" aria-label="Playback controls">
          <button
            onClick={onStop}
            disabled={!isRunning && epoch === 0}
            aria-label="Stop training"
            title="Stop (resets epoch counter)"
            className="
              w-9 h-9 rounded-xl border border-slate-200 dark:border-slate-700
              flex items-center justify-center
              bg-white dark:bg-slate-800
              hover:bg-red-50 dark:hover:bg-red-950/30
              hover:border-red-300 dark:hover:border-red-700
              disabled:opacity-35 disabled:cursor-not-allowed
              transition-all duration-150 group
              shadow-[var(--shadow-xs)] nf-bevel
            "
          >
            <Square size={14} className="text-slate-500 dark:text-slate-400 group-hover:text-red-500 transition-colors" />
          </button>

          {isRunning && !isPaused ? (
            <button
              onClick={onPause}
              aria-label="Pause training"
              title="Pause"
              className="
                w-9 h-9 rounded-xl border border-amber-300 dark:border-amber-600
                bg-amber-50 dark:bg-amber-950/30
                flex items-center justify-center
                hover:bg-amber-100 dark:hover:bg-amber-950/50
                transition-all shadow-[var(--shadow-xs)] nf-bevel
              "
            >
              <Pause size={14} className="text-amber-600 dark:text-amber-400" />
            </button>
          ) : (
            <button
              onClick={onStart}
              disabled={nodeCount === 0 || isComplete}
              aria-label={isPaused ? 'Resume training' : 'Start training'}
              title={isPaused ? 'Resume' : 'Start training'}
              className="
                w-9 h-9 rounded-xl border border-blue-400 dark:border-blue-500
                bg-blue-500 hover:bg-blue-600
                flex items-center justify-center
                disabled:opacity-35 disabled:cursor-not-allowed
                transition-all shadow-sm shadow-blue-200 dark:shadow-blue-900/30 nf-bevel
              "
            >
              <Play size={14} className="text-white ml-0.5" />
            </button>
          )}

          <button
            onClick={onStep}
            disabled={isRunning && !isPaused}
            aria-label="Step one epoch"
            title="Step one epoch"
            className="
              w-9 h-9 rounded-xl border border-slate-200 dark:border-slate-700
              bg-white dark:bg-slate-800
              flex items-center justify-center
              hover:bg-indigo-50 dark:hover:bg-indigo-950/30
              hover:border-indigo-300 dark:hover:border-indigo-700
              disabled:opacity-35 disabled:cursor-not-allowed
              transition-all group shadow-[var(--shadow-xs)] nf-bevel
            "
          >
            <SkipForward size={14} className="text-slate-500 dark:text-slate-400 group-hover:text-indigo-500 transition-colors" />
          </button>
        </div>

        {/* ── Forward Pass button ── */}
        <button
          onClick={onRunForwardPass}
          aria-label="Open forward propagation visualiser"
          aria-pressed={fwdPropOpen}
          title="Forward propagation"
          className={`
            flex items-center gap-2 px-3.5 py-2 rounded-xl border
            text-[11.5px] font-semibold flex-shrink-0
            transition-all duration-150 nf-bevel
            ${fwdPropOpen
              ? 'bg-gradient-to-r from-blue-500 to-indigo-600 border-blue-500 text-white shadow-md shadow-blue-300/40 dark:shadow-blue-900/40'
              : fwdIsComplete
                ? 'bg-teal-50 dark:bg-teal-950/30 border-teal-300 dark:border-teal-700 text-teal-700 dark:text-teal-400 hover:bg-teal-100'
                : `bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700
                   text-slate-600 dark:text-slate-300
                   hover:bg-blue-50 dark:hover:bg-blue-950/30
                   hover:border-blue-300 dark:hover:border-blue-700 hover:text-blue-700 dark:hover:text-blue-400`
            }
          `}
        >
          <Zap
            size={13}
            aria-hidden="true"
            className={fwdIsRunning ? 'animate-pulse' : ''}
            style={{ fill: fwdPropOpen ? 'white' : fwdIsComplete ? '#14B8A6' : 'none' }}
          />
          <span className="hidden sm:inline">
            {fwdIsRunning ? 'Running…' : fwdIsComplete ? 'View Results' : 'Forward Pass'}
          </span>
          <span className="sm:hidden">Fwd</span>
          {fwdPropOpen && <span className="w-1.5 h-1.5 rounded-full bg-white/70 animate-pulse" aria-hidden="true" />}
        </button>

        {/* ── Train Model button ── */}
        <button
          onClick={onOpenTraining}
          aria-label="Open training mode"
          aria-pressed={trainOpen}
          title="Train model with gradient descent"
          className={`
            flex items-center gap-2 px-3.5 py-2 rounded-xl border
            text-[11.5px] font-semibold flex-shrink-0
            transition-all duration-150 nf-bevel
            ${trainOpen
              ? 'bg-gradient-to-r from-emerald-500 to-teal-600 border-emerald-500 text-white shadow-md shadow-emerald-300/40 dark:shadow-emerald-900/40'
              : trainIsComplete
                ? 'bg-emerald-50 dark:bg-emerald-950/30 border-emerald-300 dark:border-emerald-700 text-emerald-700 dark:text-emerald-400 hover:bg-emerald-100'
                : trainIsRunning
                  ? 'bg-emerald-50 dark:bg-emerald-950/30 border-emerald-300 dark:border-emerald-700 text-emerald-700 dark:text-emerald-400'
                  : `bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700
                     text-slate-600 dark:text-slate-300
                     hover:bg-emerald-50 dark:hover:bg-emerald-950/30
                     hover:border-emerald-300 dark:hover:border-emerald-700 hover:text-emerald-700 dark:hover:text-emerald-400`
            }
          `}
        >
          <GraduationCap size={13} aria-hidden="true" className={trainIsRunning ? 'animate-pulse' : ''} />
          <span className="hidden sm:inline">
            {trainIsRunning ? 'Training…' : trainIsComplete ? 'Trained ✓' : 'Train Model'}
          </span>
          <span className="sm:hidden">Train</span>
          {trainOpen && <span className="w-1.5 h-1.5 rounded-full bg-white/70 animate-pulse" aria-hidden="true" />}
        </button>

        <Divider />

        {/* ── Epoch progress ── */}
        <div className="flex-shrink-0 min-w-[150px]">
          <div className="flex items-center justify-between mb-1.5">
            <span
              role="status"
              aria-live="polite"
              className="text-[10.5px] text-slate-500 dark:text-slate-400 font-medium"
            >
              {isComplete ? 'Complete ✓' : isRunning ? 'Training…' : epoch > 0 ? 'Paused' : 'Ready'}
            </span>
            <span className="text-[10.5px] text-slate-700 dark:text-slate-300 font-semibold tabular-nums">
              {epoch} <span className="text-slate-400 dark:text-slate-500 font-normal">/ {totalEpochs}</span>
            </span>
          </div>

          {/* Progress track */}
          <div
            role="progressbar"
            aria-valuenow={epoch}
            aria-valuemin={0}
            aria-valuemax={totalEpochs}
            aria-label="Training progress"
            className="w-full h-2 bg-slate-100 dark:bg-slate-700/70 rounded-full overflow-hidden"
          >
            <div
              className={`h-full rounded-full transition-all duration-500 ${isComplete ? 'bg-teal-500' : 'bg-gradient-to-r from-blue-500 to-indigo-500'}`}
              style={{ width: `${progress}%` }}
            />
          </div>

          {/* Epoch stepper */}
          <div className="flex items-center gap-1 mt-1.5">
            <span className="text-[9.5px] text-slate-400 dark:text-slate-500">Epochs:</span>
            <button
              onClick={() => onEpochsChange(Math.max(10, totalEpochs - 10))}
              aria-label="Decrease total epochs"
              className="text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
            >
              <ChevronDown size={11} />
            </button>
            <span className="text-[10.5px] text-slate-600 dark:text-slate-300 font-semibold tabular-nums min-w-[28px] text-center">
              {totalEpochs}
            </span>
            <button
              onClick={() => onEpochsChange(Math.min(500, totalEpochs + 10))}
              aria-label="Increase total epochs"
              className="text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
            >
              <ChevronUp size={11} />
            </button>
          </div>
        </div>

        {/* ── Speed selector ── */}
        <div className="flex-shrink-0">
          <div className="text-[9.5px] text-slate-400 dark:text-slate-500 mb-1.5 uppercase tracking-wider font-medium">
            Speed
          </div>
          <div
            className="flex items-center gap-1"
            role="group"
            aria-label="Training speed"
          >
            {speedOptions.map(s => (
              <button
                key={s}
                onClick={() => onSpeedChange(s)}
                aria-label={`${s}× speed`}
                aria-pressed={speed === s}
                className={`
                  px-2 py-1 rounded-lg text-[10.5px] font-semibold tabular-nums
                  transition-all duration-100 nf-bevel
                  ${speed === s
                    ? 'bg-blue-500 text-white shadow-sm shadow-blue-300/40 dark:shadow-blue-900/40'
                    : 'bg-slate-100 dark:bg-slate-700/70 text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                  }
                `}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>

        <Divider />

        {/* ── Metrics ── */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <MetricBadge
            label="Loss"
            value={loss.toFixed(4)}
            colorClass="border-red-100 dark:border-red-900/50"
            icon={<TrendingDown size={14} className="text-red-400 dark:text-red-500" />}
            ariaLabel={`Current loss: ${loss.toFixed(4)}`}
          />
          <MetricBadge
            label="Accuracy"
            value={`${(accuracy * 100).toFixed(1)}%`}
            colorClass="border-teal-100 dark:border-teal-900/50"
            icon={<Target size={14} className="text-teal-500 dark:text-teal-400" />}
            ariaLabel={`Current accuracy: ${(accuracy * 100).toFixed(1)} percent`}
          />
        </div>

        {/* ── Mini loss chart ── */}
        <div className="flex-shrink-0 w-[148px] hidden md:block">
          <div className="text-[9.5px] text-slate-400 dark:text-slate-500 mb-1 uppercase tracking-wider flex items-center gap-1 font-medium">
            <Activity size={10} aria-hidden="true" />
            Loss curve
          </div>
          {history.length > 2 ? (
            <ResponsiveContainer width="100%" height={44}>
              <LineChart data={history} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
                <Line type="monotone" dataKey="loss"     stroke="#EF4444" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="accuracy" stroke="#14B8A6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Tooltip
                  contentStyle={{
                    fontSize: 10, padding: '3px 8px',
                    borderRadius: 8, border: '1px solid #E2E8F0',
                    background: 'rgba(255,255,255,0.96)',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                  }}
                  formatter={(v: number, name: string) => [
                    name === 'loss' ? v.toFixed(4) : `${(v*100).toFixed(1)}%`,
                    name === 'loss' ? 'Loss' : 'Acc',
                  ]}
                  labelFormatter={() => ''}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-11 flex items-center justify-center text-[9.5px] text-slate-300 dark:text-slate-600 italic">
              Train to see curve
            </div>
          )}
        </div>

        <Divider />

        {/* ── Model stats ── */}
        <div className="flex-shrink-0 space-y-1 hidden sm:block">
          <div className="text-[9.5px] text-slate-400 dark:text-slate-500 uppercase tracking-wider font-medium">
            Model
          </div>
          <div className="flex items-center gap-3">
            <div className="text-[11px] text-slate-600 dark:text-slate-300">
              <span className="font-bold text-blue-600 dark:text-blue-400 tabular-nums">{nodeCount}</span>
              <span className="text-slate-400 dark:text-slate-500"> layers</span>
            </div>
            <div className="text-[11px] text-slate-600 dark:text-slate-300">
              <span className="font-bold text-indigo-600 dark:text-indigo-400 tabular-nums">{connectionCount}</span>
              <span className="text-slate-400 dark:text-slate-500"> conns</span>
            </div>
          </div>
          {isComplete && (
            <div className="flex items-center gap-1 text-[9.5px] text-teal-600 dark:text-teal-400">
              <Zap size={9} aria-hidden="true" />
              <span>Training complete!</span>
            </div>
          )}
        </div>

        {/* Spacer */}
        <div className="flex-1" aria-hidden="true" />

        {/* ── Status pill ── */}
        <div
          role="status"
          aria-live="polite"
          aria-label={`Status: ${statusText}`}
          className="flex items-center gap-2 flex-shrink-0 px-3 py-1.5 rounded-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700/60"
        >
          <div className={`w-1.5 h-1.5 rounded-full ${statusColor}`} aria-hidden="true" />
          <span className="text-[10.5px] text-slate-500 dark:text-slate-400 font-medium">{statusText}</span>
        </div>
      </div>
    </div>
  );
});
