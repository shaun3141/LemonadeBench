// Formatting utilities

/**
 * Format cents as dollar string (e.g., 150 -> "$1.50")
 */
export function formatCents(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}

/**
 * Format cents as profit string with +/- prefix (e.g., 150 -> "+$1.50")
 */
export function formatProfit(cents: number): string {
  const prefix = cents >= 0 ? '+' : '';
  return `${prefix}$${(cents / 100).toFixed(2)}`;
}

/**
 * Format a date string for display
 */
export function formatDate(dateStr: string | null): string {
  if (!dateStr) return 'In progress';
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Get CSS class for profit color
 */
export function getProfitColor(profit: number): string {
  if (profit > 0) return 'text-green-600';
  if (profit < 0) return 'text-red-500';
  return 'text-gray-500';
}

/**
 * Format reputation as star string (0.0-1.0 -> "⭐⭐⭐☆☆")
 */
export function getReputationStars(reputation: number): string {
  const stars = Math.round(reputation * 5);
  return '⭐'.repeat(stars) + '☆'.repeat(5 - stars);
}

