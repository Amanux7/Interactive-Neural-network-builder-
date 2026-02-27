export type NodeType =
  | 'input'
  | 'dense'
  | 'activation'
  | 'dropout'
  | 'output'
  | 'conv2d'
  | 'flatten'
  | 'lstm'
  | 'batchnorm'
  | 'embedding';

export interface NodeConfig {
  neurons?: number;
  activationFn?: string;
  dropoutRate?: number;
  filters?: number;
  kernelSize?: number;
  inputShape?: string;
  outputShape?: string;
  units?: number;
  vocabSize?: number;
  embedDim?: number;
  poolSize?: number;
}

export const NODE_DIMS: Record<NodeType, { w: number; h: number }> = {
  input: { w: 160, h: 80 },
  dense: { w: 160, h: 80 },
  activation: { w: 140, h: 60 },
  dropout: { w: 140, h: 60 },
  output: { w: 160, h: 80 },
  conv2d: { w: 160, h: 80 },
  flatten: { w: 140, h: 60 },
  lstm: { w: 160, h: 80 },
  batchnorm: { w: 140, h: 60 },
  embedding: { w: 160, h: 80 },
};

export const NODE_STYLE: Record<NodeType, {
  bg: string; border: string; text: string; accent: string; dot: string;
  glowColor: string; borderHex: string;
}> = {
  // Input: Light cognitive blue
  input: { bg: 'bg-[#EFF6FF]', border: 'border-[#93C5FD]', text: 'text-[#1E40AF]', accent: 'bg-[#1E40AF]', dot: '#3B82F6', glowColor: 'rgba(59,130,246,0.35)', borderHex: '#93C5FD' },
  // Dense: Deep primary blue 
  dense: { bg: 'bg-[#EEF2FF]', border: 'border-[#A5B4FC]', text: 'text-[#1E3A8A]', accent: 'bg-[#3730A3]', dot: '#4F46E5', glowColor: 'rgba(79,70,229,0.35)', borderHex: '#A5B4FC' },
  // Activation: Accent amber/orange warning-ish
  activation: { bg: 'bg-[#FFFBEB]', border: 'border-[#FCD34D]', text: 'text-[#92400E]', accent: 'bg-[#F59E0B]', dot: '#F59E0B', glowColor: 'rgba(245,158,11,0.35)', borderHex: '#FDE68A' },
  // Dropout: Neutral slate blue
  dropout: { bg: 'bg-[#F8FAFC]', border: 'border-[#CBD5E1]', text: 'text-[#334155]', accent: 'bg-[#64748B]', dot: '#64748B', glowColor: 'rgba(100,116,139,0.35)', borderHex: '#E2E8F0' },
  // Output: Secondary teal
  output: { bg: 'bg-[#F0FDFA]', border: 'border-[#5EEAD4]', text: 'text-[#0F766E]', accent: 'bg-[#0D9488]', dot: '#14B8A6', glowColor: 'rgba(20,184,166,0.35)', borderHex: '#99F6E4' },
  // Conv2D: Deep cyan
  conv2d: { bg: 'bg-[#ECFEFF]', border: 'border-[#67E8F9]', text: 'text-[#164E63]', accent: 'bg-[#0891B2]', dot: '#06B6D4', glowColor: 'rgba(6,182,212,0.35)', borderHex: '#A5F3FC' },
  // Flatten: Slate neutral
  flatten: { bg: 'bg-[#F1F5F9]', border: 'border-[#94A3B8]', text: 'text-[#0F172A]', accent: 'bg-[#475569]', dot: '#94A3B8', glowColor: 'rgba(148,163,184,0.35)', borderHex: '#CBD5E1' },
  // LSTM: Purple/Indigo edge
  lstm: { bg: 'bg-[#F5F3FF]', border: 'border-[#C4B5FD]', text: 'text-[#4C1D95]', accent: 'bg-[#6D28D9]', dot: '#8B5CF6', glowColor: 'rgba(139,92,246,0.35)', borderHex: '#DDD6FE' },
  // BatchNorm: Teal highlight
  batchnorm: { bg: 'bg-[#F0FDF4]', border: 'border-[#86EFAC]', text: 'text-[#14532D]', accent: 'bg-[#16A34A]', dot: '#22C55E', glowColor: 'rgba(34,197,94,0.35)', borderHex: '#BBF7D0' },
  // Embedding: Amber deep
  embedding: { bg: 'bg-[#FEF3C7]', border: 'border-[#FBBF24]', text: 'text-[#B45309]', accent: 'bg-[#D97706]', dot: '#F59E0B', glowColor: 'rgba(245,158,11,0.35)', borderHex: '#FDE68A' },
};

export interface NetworkNode {
  id: string;
  type: NodeType;
  x: number;
  y: number;
  w?: number; // custom width (overrides NODE_DIMS default)
  h?: number; // custom height (overrides NODE_DIMS default)
  label: string;
  config: NodeConfig;
}

export interface NetworkConnection {
  id: string;
  fromId: string;
  toId: string;
  weight?: number;   // default 0.5 — signed [-2, 2]; visual encodes magnitude+sign
  active?: boolean;  // default true; false = user-disabled, always dashed + dimmed
}

export interface Particle {
  id: string;
  connectionId: string;
  progress: number;
  speed: number;
}

export interface SimulationState {
  isRunning: boolean;
  isPaused: boolean;
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  speed: number;
  history: Array<{ epoch: number; loss: number; accuracy: number }>;
}

export interface ComponentTemplate {
  type: NodeType;
  label: string;
  description: string;
  defaultConfig: NodeConfig;
  category: 'Layers' | 'Activations' | 'Regularization' | 'Sequence';
}

export const COMPONENT_LIBRARY: ComponentTemplate[] = [
  { type: 'input', label: 'Input Layer', description: 'Network entry point', defaultConfig: { inputShape: '784' }, category: 'Layers' },
  { type: 'dense', label: 'Dense Layer', description: 'Fully connected layer', defaultConfig: { neurons: 128 }, category: 'Layers' },
  { type: 'conv2d', label: 'Conv2D', description: '2D convolution layer', defaultConfig: { filters: 32, kernelSize: 3 }, category: 'Layers' },
  { type: 'output', label: 'Output Layer', description: 'Final prediction layer', defaultConfig: { outputShape: '10' }, category: 'Layers' },
  { type: 'flatten', label: 'Flatten', description: 'Flatten to 1D tensor', defaultConfig: {}, category: 'Layers' },
  { type: 'activation', label: 'ReLU', description: 'Rectified linear unit', defaultConfig: { activationFn: 'ReLU' }, category: 'Activations' },
  { type: 'activation', label: 'Sigmoid', description: 'Sigmoid activation fn', defaultConfig: { activationFn: 'Sigmoid' }, category: 'Activations' },
  { type: 'activation', label: 'Tanh', description: 'Hyperbolic tangent fn', defaultConfig: { activationFn: 'Tanh' }, category: 'Activations' },
  { type: 'activation', label: 'Softmax', description: 'Probability distribution', defaultConfig: { activationFn: 'Softmax' }, category: 'Activations' },
  { type: 'dropout', label: 'Dropout', description: 'Regularization via dropout', defaultConfig: { dropoutRate: 0.5 }, category: 'Regularization' },
  { type: 'batchnorm', label: 'Batch Norm', description: 'Normalize activations', defaultConfig: {}, category: 'Regularization' },
  { type: 'lstm', label: 'LSTM', description: 'Long short-term memory', defaultConfig: { units: 64 }, category: 'Sequence' },
  { type: 'embedding', label: 'Embedding', description: 'Learns word representations', defaultConfig: { vocabSize: 10000, embedDim: 64 }, category: 'Sequence' },
];