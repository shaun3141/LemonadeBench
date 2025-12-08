import {
  Target,
  Brain,
  Zap,
  Shield,
  Swords,
  Heart,
  TrendingUp as Growth,
  Calculator,
  Code,
  MessageSquare,
  Network,
  FlaskConical,
} from 'lucide-react';

// Model tiers from the paper
export const MODEL_TIERS = {
  premium: {
    name: 'Premium',
    color: 'retro-gold' as const,
    bgColor: 'bg-gradient-to-b from-[#FFF9C4] to-[#FFECB3]',
    borderColor: 'border-[#FFA000]',
    textColor: 'text-[#5D4037]',
    description: '$1-15/M tokens',
    models: [
      { id: 'anthropic/claude-sonnet-4', name: 'Claude Sonnet 4', provider: 'Anthropic' },
      { id: 'anthropic/claude-opus-4.5', name: 'Claude Opus 4.5', provider: 'Anthropic' },
      { id: 'openai/o1', name: 'o1', provider: 'OpenAI' },
      { id: 'openai/gpt-5.1', name: 'GPT-5.1', provider: 'OpenAI' },
      { id: 'google/gemini-3-pro', name: 'Gemini 3 Pro', provider: 'Google' },
    ],
  },
  balanced: {
    name: 'Balanced',
    color: 'retro-blue' as const,
    bgColor: 'bg-gradient-to-b from-[#E3F2FD] to-[#BBDEFB]',
    borderColor: 'border-[#1976D2]',
    textColor: 'text-[#0D47A1]',
    description: '$1-6/M tokens',
    models: [
      { id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet', provider: 'Anthropic' },
      { id: 'openai/o3-mini', name: 'o3-mini', provider: 'OpenAI' },
      { id: 'google/gemini-2.5-flash', name: 'Gemini 2.5 Flash', provider: 'Google' },
      { id: 'x-ai/grok-2', name: 'Grok-2', provider: 'xAI' },
    ],
  },
  value: {
    name: 'Value',
    color: 'retro-green' as const,
    bgColor: 'bg-gradient-to-b from-[#E8F5E9] to-[#C8E6C9]',
    borderColor: 'border-[#388E3C]',
    textColor: 'text-[#1B5E20]',
    description: '$0.14-2.19/M tokens',
    models: [
      { id: 'deepseek/deepseek-r1', name: 'DeepSeek R1', provider: 'DeepSeek' },
      { id: 'deepseek/deepseek-v3', name: 'DeepSeek V3', provider: 'DeepSeek' },
      { id: 'qwen/qwq-32b', name: 'QwQ-32B', provider: 'Qwen' },
      { id: 'mistralai/mistral-small-3', name: 'Mistral Small 3', provider: 'Mistral' },
    ],
  },
  opensource: {
    name: 'Open Source',
    color: 'retro' as const,
    bgColor: 'bg-gradient-to-b from-[#FFFDE7] to-[#FFF9C4]',
    borderColor: 'border-[#FFA000]',
    textColor: 'text-[#5D4037]',
    description: '$0.20-2/M tokens',
    models: [
      { id: 'meta-llama/llama-3.3-70b-instruct', name: 'Llama 3.3 70B', provider: 'Meta' },
      { id: 'meta-llama/llama-3.1-405b-instruct', name: 'Llama 3.1 405B', provider: 'Meta' },
      { id: 'qwen/qwen-2.5-72b-instruct', name: 'Qwen 2.5 72B', provider: 'Qwen' },
      { id: 'mistralai/mistral-large', name: 'Mistral Large', provider: 'Mistral' },
    ],
  },
  fast: {
    name: 'Fast',
    color: 'retro-pink' as const,
    bgColor: 'bg-gradient-to-b from-[#FCE4EC] to-[#F8BBD9]',
    borderColor: 'border-[#C2185B]',
    textColor: 'text-[#880E4F]',
    description: '$0.075-4/M tokens',
    models: [
      { id: 'anthropic/claude-3.5-haiku', name: 'Claude 3.5 Haiku', provider: 'Anthropic' },
      { id: 'openai/gpt-4o-mini', name: 'GPT-4o-mini', provider: 'OpenAI' },
      { id: 'google/gemini-flash-1.5', name: 'Gemini Flash 1.5', provider: 'Google' },
    ],
  },
};

// Goal framing conditions
export const GOAL_FRAMINGS = [
  {
    id: 'baseline',
    name: 'Baseline',
    icon: Target,
    bgColor: 'bg-gradient-to-b from-[#FFFDE7] to-[#FFF9C4]',
    borderColor: 'border-[#5D4037]',
    iconBg: 'bg-[#8B4513]/20',
    iconColor: 'text-[#5D4037]',
    description: 'No additional framing — base prompt only',
  },
  {
    id: 'aggressive',
    name: 'Aggressive',
    icon: Zap,
    bgColor: 'bg-gradient-to-b from-[#FFEBEE] to-[#FFCDD2]',
    borderColor: 'border-[#C62828]',
    iconBg: 'bg-[#C62828]/20',
    iconColor: 'text-[#C62828]',
    description: 'Risk-taking entrepreneur who maximizes returns',
  },
  {
    id: 'conservative',
    name: 'Conservative',
    icon: Shield,
    bgColor: 'bg-gradient-to-b from-[#E3F2FD] to-[#BBDEFB]',
    borderColor: 'border-[#1976D2]',
    iconBg: 'bg-[#1976D2]/20',
    iconColor: 'text-[#1976D2]',
    description: 'Cautious owner who prioritizes avoiding losses',
  },
  {
    id: 'competitive',
    name: 'Competitive',
    icon: Swords,
    bgColor: 'bg-gradient-to-b from-[#F3E5F5] to-[#E1BEE7]',
    borderColor: 'border-[#7B1FA2]',
    iconBg: 'bg-[#7B1FA2]/20',
    iconColor: 'text-[#7B1FA2]',
    description: 'Tournament framing against 10 other stands',
  },
  {
    id: 'survival',
    name: 'Survival',
    icon: Heart,
    bgColor: 'bg-gradient-to-b from-[#FFF3E0] to-[#FFE0B2]',
    borderColor: 'border-[#E65100]',
    iconBg: 'bg-[#E65100]/20',
    iconColor: 'text-[#E65100]',
    description: 'Family depends on success — survival is priority',
  },
  {
    id: 'growth',
    name: 'Growth',
    icon: Growth,
    bgColor: 'bg-gradient-to-b from-[#E8F5E9] to-[#C8E6C9]',
    borderColor: 'border-[#388E3C]',
    iconBg: 'bg-[#388E3C]/20',
    iconColor: 'text-[#388E3C]',
    description: 'Building an empire — focus on learning and reputation',
  },
];

// Architecture types
export const ARCHITECTURES = [
  {
    id: 'react',
    name: 'ReAct',
    icon: Zap,
    flow: 'Observe → Decide → Act',
    description: 'Baseline architecture based on ReAct framework',
  },
  {
    id: 'plan_act',
    name: 'Plan-Act',
    icon: Brain,
    flow: 'Observe → Plan → Decide → Act',
    description: 'Adds explicit multi-day planning before decisions',
  },
  {
    id: 'act_reflect',
    name: 'Act-Reflect',
    icon: MessageSquare,
    flow: 'Observe → Decide → Act → Reflect',
    description: 'Adds retrospective analysis after each action',
  },
  {
    id: 'full',
    name: 'Full',
    icon: Network,
    flow: 'Plan → Decide → Act → Reflect',
    description: 'Combines planning and reflection (3x API cost)',
  },
];

// Scaffolding types
export const SCAFFOLDINGS = [
  {
    id: 'none',
    name: 'None',
    icon: Target,
    description: 'No additional cognitive scaffolding',
  },
  {
    id: 'calculator',
    name: 'Calculator',
    icon: Calculator,
    description: 'Access to calculator tool for arithmetic',
  },
  {
    id: 'math_prompt',
    name: 'Math Prompt',
    icon: FlaskConical,
    description: 'Encouragement to show calculations step-by-step',
  },
  {
    id: 'code_interpreter',
    name: 'Code Interpreter',
    icon: Code,
    description: 'Python code execution for complex analysis',
  },
];

