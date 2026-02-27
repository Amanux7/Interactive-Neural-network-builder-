import React, { useEffect, useState, useCallback } from 'react';
import { X, ChevronRight, ChevronLeft, Lightbulb } from 'lucide-react';

// ── Step definitions ──────────────────────────────────────────────────────────
export interface TutorialStep {
    id: string;
    title: string;
    description: string;
    /** data-tutorial-id of the element to spotlight; null = center overlay */
    targetId: string | null;
    /** Where to place the tooltip relative to the target */
    placement: 'top' | 'bottom' | 'left' | 'right' | 'center';
    emoji: string;
}

export const TUTORIAL_STEPS: TutorialStep[] = [
    {
        id: 'welcome',
        title: 'Welcome to NeuralForge 🧠',
        description:
            'This quick tour shows you how to build, train, and export a neural network in minutes. Use the arrows or press Esc to skip.',
        targetId: null,
        placement: 'center',
        emoji: '👋',
    },
    {
        id: 'sidebar',
        title: 'Component Library',
        description:
            'The sidebar lists all available layer types — Dense, Conv2D, LSTM, Dropout, and more. Drag any layer onto the canvas to add it to your network.',
        targetId: 'tutorial-sidebar',
        placement: 'right',
        emoji: '📦',
    },
    {
        id: 'canvas',
        title: 'Canvas & Wiring',
        description:
            'Nodes live on the canvas. Connect them by clicking the right port of one node and then the left port of another. Right-click a node for advanced options.',
        targetId: 'tutorial-canvas',
        placement: 'top',
        emoji: '🔗',
    },
    {
        id: 'templates',
        title: 'Pre-built Templates',
        description:
            'Load an example network instantly — MNIST (7-layer MLP), Binary Classifier, or a ConvNet. The current canvas shows the MNIST demo network.',
        targetId: 'tutorial-templates',
        placement: 'bottom',
        emoji: '⚡',
    },
    {
        id: 'training',
        title: 'Train & Simulate',
        description:
            'Use the Simulation Bar at the bottom to run training. Hit Space to start, or open the Training Panel for full control over datasets and hyperparameters.',
        targetId: 'tutorial-simbar',
        placement: 'top',
        emoji: '🎓',
    },
    {
        id: 'export',
        title: 'Export Your Model',
        description:
            'Click the Export button (↓ icon) in the header to download a JSON with your architecture, training history, and a TensorFlow.js-compatible layer descriptor.',
        targetId: 'tutorial-export',
        placement: 'bottom',
        emoji: '💾',
    },
];

// ── Spotlight box ──────────────────────────────────────────────────────────────
function useTargetRect(targetId: string | null, step: number) {
    const [rect, setRect] = useState<DOMRect | null>(null);
    useEffect(() => {
        if (!targetId) { setRect(null); return; }
        const el = document.querySelector(`[data-tutorial-id="${targetId}"]`);
        if (el) setRect(el.getBoundingClientRect());
        else setRect(null);
    }, [targetId, step]);
    return rect;
}

// ── Tooltip placement ──────────────────────────────────────────────────────────
function tooltipStyle(
    rect: DOMRect | null,
    placement: TutorialStep['placement'],
): React.CSSProperties {
    if (!rect || placement === 'center') {
        return {
            position: 'fixed',
            top: '50%', left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 1002,
            width: '320px',
        };
    }
    const pad = 14;
    const w = 300;
    switch (placement) {
        case 'right':
            return { position: 'fixed', top: rect.top + rect.height / 2 - 80, left: rect.right + pad, zIndex: 1002, width: `${w}px` };
        case 'left':
            return { position: 'fixed', top: rect.top + rect.height / 2 - 80, left: rect.left - w - pad, zIndex: 1002, width: `${w}px` };
        case 'bottom':
            return { position: 'fixed', top: rect.bottom + pad, left: Math.min(Math.max(rect.left + rect.width / 2 - w / 2, 12), window.innerWidth - w - 12), zIndex: 1002, width: `${w}px` };
        case 'top':
        default:
            return { position: 'fixed', top: rect.top - pad - 160, left: Math.min(Math.max(rect.left + rect.width / 2 - w / 2, 12), window.innerWidth - w - 12), zIndex: 1002, width: `${w}px` };
    }
}

// ── Props ──────────────────────────────────────────────────────────────────────
interface TutorialOverlayProps {
    isOpen: boolean;
    onClose: () => void;
    isDark?: boolean;
}

// ── Component ──────────────────────────────────────────────────────────────────
export function TutorialOverlay({ isOpen, onClose, isDark }: TutorialOverlayProps) {
    const [step, setStep] = useState(0);
    const current = TUTORIAL_STEPS[step];
    const rect = useTargetRect(current.targetId, step);

    const handleNext = useCallback(() => {
        if (step < TUTORIAL_STEPS.length - 1) setStep(s => s + 1);
        else onClose();
    }, [step, onClose]);

    const handlePrev = useCallback(() => setStep(s => Math.max(0, s - 1)), []);

    // Reset step when opened
    useEffect(() => { if (isOpen) setStep(0); }, [isOpen]);

    // Keyboard navigation
    useEffect(() => {
        if (!isOpen) return;
        const handler = (e: KeyboardEvent) => {
            if (e.key === 'Escape') onClose();
            if (e.key === 'ArrowRight' || e.key === 'Enter') handleNext();
            if (e.key === 'ArrowLeft') handlePrev();
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [isOpen, handleNext, handlePrev, onClose]);

    if (!isOpen) return null;

    const bg = isDark ? '#1E293B' : '#FFFFFF';
    const border = isDark ? '#334155' : '#E2E8F0';
    const text = isDark ? '#E2E8F0' : '#1E293B';
    const sub = isDark ? '#94A3B8' : '#64748B';

    // Spotlight padding around element
    const spotPad = 10;
    const spotlight = rect
        ? { x: rect.left - spotPad, y: rect.top - spotPad, w: rect.width + spotPad * 2, h: rect.height + spotPad * 2 }
        : null;

    return (
        <>
            {/* ── Dark backdrop with SVG spotlight cutout ── */}
            <div
                className="fixed inset-0 z-[1000] pointer-events-none"
                aria-hidden="true"
            >
                {spotlight ? (
                    <svg width="100%" height="100%" style={{ position: 'absolute', inset: 0 }}>
                        <defs>
                            <mask id="nf-tutorial-mask">
                                <rect width="100%" height="100%" fill="white" />
                                <rect
                                    x={spotlight.x} y={spotlight.y}
                                    width={spotlight.w} height={spotlight.h}
                                    rx="12" fill="black"
                                />
                            </mask>
                        </defs>
                        <rect
                            width="100%" height="100%"
                            fill="rgba(0,0,0,0.65)"
                            mask="url(#nf-tutorial-mask)"
                        />
                        {/* Pulse ring */}
                        <rect
                            x={spotlight.x - 3} y={spotlight.y - 3}
                            width={spotlight.w + 6} height={spotlight.h + 6}
                            rx="14" fill="none" stroke="#3B82F6" strokeWidth="2"
                            style={{ opacity: 0.7 }}
                        />
                    </svg>
                ) : (
                    <div className="absolute inset-0" style={{ background: 'rgba(0,0,0,0.65)' }} />
                )}
            </div>

            {/* Click-catcher to prevent interaction with backdrop */}
            <div className="fixed inset-0 z-[1001]" onClick={e => { if (e.target === e.currentTarget) onClose(); }} />

            {/* ── Tooltip card ── */}
            <div
                role="dialog" aria-label={`Tutorial step ${step + 1}: ${current.title}`} aria-modal="true"
                className="nf-animate-in rounded-2xl overflow-hidden"
                style={{
                    ...tooltipStyle(rect, current.placement),
                    background: bg, border: `1px solid ${border}`,
                    boxShadow: '0 24px 64px rgba(0,0,0,0.3), 0 4px 16px rgba(0,0,0,0.15)',
                }}
            >
                {/* Color header bar */}
                <div className="h-1.5 w-full" style={{ background: 'linear-gradient(90deg, #3B82F6, #8B5CF6, #06B6D4)' }} />

                <div className="p-4">
                    {/* Step indicator */}
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex gap-1">
                            {TUTORIAL_STEPS.map((_, i) => (
                                <div
                                    key={i}
                                    className="h-1.5 rounded-full transition-all duration-300"
                                    style={{
                                        width: i === step ? '20px' : '6px',
                                        background: i === step ? '#3B82F6' : (isDark ? '#334155' : '#E2E8F0'),
                                    }}
                                />
                            ))}
                        </div>
                        <div className="flex items-center gap-1">
                            <span className="text-[10px]" style={{ color: sub }}>{step + 1} / {TUTORIAL_STEPS.length}</span>
                            <button onClick={onClose} aria-label="Skip tutorial"
                                className="w-6 h-6 rounded-lg flex items-center justify-center transition-colors"
                                style={{ color: sub }}
                                onMouseEnter={e => (e.currentTarget.style.background = isDark ? '#334155' : '#F1F5F9')}
                                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                                <X size={11} />
                            </button>
                        </div>
                    </div>

                    {/* Emoji + title */}
                    <div className="flex items-start gap-3 mb-3">
                        <div className="text-2xl leading-none flex-shrink-0 mt-0.5">{current.emoji}</div>
                        <div>
                            <h3 className="text-[14px] font-bold leading-snug mb-1" style={{ color: text }}>{current.title}</h3>
                            <p className="text-[11.5px] leading-relaxed" style={{ color: sub }}>{current.description}</p>
                        </div>
                    </div>

                    {/* Tip for non-center steps */}
                    {current.targetId && (
                        <div className="flex items-start gap-1.5 mb-3 rounded-lg px-2.5 py-2"
                            style={{ background: isDark ? '#1E3A5F' : '#EFF6FF', border: `1px solid ${isDark ? '#2563EB33' : '#BFDBFE'}` }}>
                            <Lightbulb size={10} className="text-blue-500 flex-shrink-0 mt-0.5" />
                            <span className="text-[10px]" style={{ color: isDark ? '#93C5FD' : '#1D4ED8' }}>
                                The highlighted area is where this action happens.
                            </span>
                        </div>
                    )}

                    {/* Navigation buttons */}
                    <div className="flex items-center gap-2">
                        {step > 0 && (
                            <button onClick={handlePrev}
                                className="flex items-center gap-1 px-3 py-2 rounded-xl text-[11px] font-semibold border transition-all"
                                style={{ border: `1px solid ${border}`, color: sub, background: 'transparent' }}>
                                <ChevronLeft size={12} /> Back
                            </button>
                        )}
                        <button onClick={handleNext}
                            className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl text-[12px] font-bold text-white transition-all active:scale-[0.98]"
                            style={{ background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)', boxShadow: '0 4px 12px rgba(59,130,246,0.35)' }}>
                            {step === TUTORIAL_STEPS.length - 1 ? 'Done! 🎉' : <>Next <ChevronRight size={12} /></>}
                        </button>
                    </div>
                </div>
            </div>
        </>
    );
}
