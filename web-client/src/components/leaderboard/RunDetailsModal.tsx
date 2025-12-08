import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Citrus,
  TrendingUp,
  CupSoda,
  Star,
  Calendar,
  Hash,
  Loader2,
  AlertCircle,
} from 'lucide-react';
import type { LeaderboardRun, RunTurn } from '@/types';
import { getRunTurns } from '@/api';
import { formatCents } from '@/lib/format';
import { TurnCard } from './TurnCard';

interface RunDetailsModalProps {
  run: LeaderboardRun;
  onClose: () => void;
}

export function RunDetailsModal({ run, onClose }: RunDetailsModalProps) {
  const [turns, setTurns] = useState<RunTurn[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedTurn, setExpandedTurn] = useState<string | null>(null);

  useEffect(() => {
    async function loadTurns() {
      setLoading(true);
      setError(null);
      try {
        const data = await getRunTurns(run.run_id);
        setTurns(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load turns');
      } finally {
        setLoading(false);
      }
    }
    loadTurns();
  }, [run.run_id]);

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Citrus className="h-5 w-5 text-yellow-500" />
            Run Details: {run.model_name}
          </DialogTitle>
        </DialogHeader>

        {/* Summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 py-4">
          <div className="text-center p-3 bg-green-50 dark:bg-green-950/30 rounded-lg">
            <TrendingUp className="h-5 w-5 mx-auto text-green-600 mb-1" />
            <div className="text-xl font-bold text-green-600">
              {formatCents(run.total_profit)}
            </div>
            <div className="text-xs text-muted-foreground">Total Profit</div>
          </div>
          <div className="text-center p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg">
            <CupSoda className="h-5 w-5 mx-auto text-blue-600 mb-1" />
            <div className="text-xl font-bold text-blue-600">{run.total_cups_sold}</div>
            <div className="text-xs text-muted-foreground">Cups Sold</div>
          </div>
          <div className="text-center p-3 bg-yellow-50 dark:bg-yellow-950/30 rounded-lg">
            <Star className="h-5 w-5 mx-auto text-yellow-600 mb-1" />
            <div className="text-xl font-bold text-yellow-600">
              {(run.final_reputation * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-muted-foreground">Reputation</div>
          </div>
          <div className="text-center p-3 bg-purple-50 dark:bg-purple-950/30 rounded-lg">
            <Hash className="h-5 w-5 mx-auto text-purple-600 mb-1" />
            <div className="text-xl font-bold text-purple-600">{run.seed ?? 'Random'}</div>
            <div className="text-xs text-muted-foreground">Seed</div>
          </div>
        </div>

        {/* Turn-by-turn data */}
        <div className="border-t pt-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold flex items-center gap-2">
              <Calendar className="h-4 w-4" />
              Turn-by-Turn Results
            </h3>
            <span className="text-xs text-muted-foreground">Click a day to expand details</span>
          </div>

          {loading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          )}

          {error && (
            <div className="text-center py-8 text-red-500">
              <AlertCircle className="h-6 w-6 mx-auto mb-2" />
              {error}
            </div>
          )}

          {!loading && !error && turns.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">No turn data available</div>
          )}

          {!loading && !error && turns.length > 0 && (
            <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
              {turns.map((turn) => (
                <TurnCard
                  key={turn.id}
                  turn={turn}
                  isExpanded={expandedTurn === turn.id}
                  onToggle={() => setExpandedTurn(expandedTurn === turn.id ? null : turn.id)}
                />
              ))}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

