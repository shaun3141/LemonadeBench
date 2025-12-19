import { useState, useEffect, useMemo } from 'react';
import { PageLayout } from '@/components/layout';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Database,
  Loader2,
  AlertCircle,
  Search,
  ChevronUp,
  ChevronDown,
  CheckCircle,
  XCircle,
  Filter,
} from 'lucide-react';
import type { LeaderboardRun } from '@/types';
import { getAllRunsIncludingFailed } from '@/api';

type SortField = 'started_at' | 'total_profit' | 'model_name' | 'turn_count' | 'taxonomy_version';
type SortDirection = 'asc' | 'desc';

export function DataPage() {
  const [runs, setRuns] = useState<LeaderboardRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Filter state
  const [searchQuery, setSearchQuery] = useState('');
  const [showOnlyFailed, setShowOnlyFailed] = useState(false);
  const [showOnlyCompleted, setShowOnlyCompleted] = useState(false);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const [selectedScaffolding, setSelectedScaffolding] = useState<string | null>(null);
  
  // Sort state
  const [sortField, setSortField] = useState<SortField>('started_at');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);
      try {
        const data = await getAllRunsIncludingFailed();
        setRuns(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // Get unique values for filters
  const versions = useMemo(() => {
    const v = new Set(runs.map(r => r.taxonomy_version || 1));
    return Array.from(v).sort((a, b) => b - a);
  }, [runs]);

  const scaffoldings = useMemo(() => {
    const s = new Set(runs.map(r => r.scaffolding));
    return Array.from(s).sort();
  }, [runs]);

  // Filter and sort runs
  const filteredRuns = useMemo(() => {
    let result = runs;
    
    // Search filter
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      result = result.filter(r => 
        r.model_name.toLowerCase().includes(q) ||
        r.provider.toLowerCase().includes(q) ||
        r.error_message?.toLowerCase().includes(q) ||
        r.goal_framing.toLowerCase().includes(q) ||
        r.architecture.toLowerCase().includes(q) ||
        r.scaffolding.toLowerCase().includes(q)
      );
    }
    
    // Status filters
    if (showOnlyFailed) {
      result = result.filter(r => r.error_message);
    }
    if (showOnlyCompleted) {
      result = result.filter(r => r.completed_at && !r.error_message);
    }
    
    // Version filter
    if (selectedVersion !== null) {
      result = result.filter(r => (r.taxonomy_version || 1) === selectedVersion);
    }
    
    // Scaffolding filter
    if (selectedScaffolding) {
      result = result.filter(r => r.scaffolding === selectedScaffolding);
    }
    
    // Sort
    result = [...result].sort((a, b) => {
      let aVal: string | number = '';
      let bVal: string | number = '';
      
      switch (sortField) {
        case 'started_at':
          aVal = a.started_at;
          bVal = b.started_at;
          break;
        case 'total_profit':
          aVal = a.total_profit;
          bVal = b.total_profit;
          break;
        case 'model_name':
          aVal = a.model_name.toLowerCase();
          bVal = b.model_name.toLowerCase();
          break;
        case 'turn_count':
          aVal = a.turn_count;
          bVal = b.turn_count;
          break;
        case 'taxonomy_version':
          aVal = a.taxonomy_version || 1;
          bVal = b.taxonomy_version || 1;
          break;
      }
      
      if (sortDirection === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      }
    });
    
    return result;
  }, [runs, searchQuery, showOnlyFailed, showOnlyCompleted, selectedVersion, selectedScaffolding, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDirection === 'asc' ? 
      <ChevronUp className="h-4 w-4" /> : 
      <ChevronDown className="h-4 w-4" />;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatProfit = (cents: number) => {
    return `$${(cents / 100).toFixed(2)}`;
  };

  // Stats
  const stats = useMemo(() => {
    const total = runs.length;
    const completed = runs.filter(r => r.completed_at && !r.error_message).length;
    const failed = runs.filter(r => r.error_message).length;
    const inProgress = runs.filter(r => !r.completed_at && !r.error_message).length;
    return { total, completed, failed, inProgress };
  }, [runs]);

  return (
    <PageLayout headerSubtitle="Data Explorer" footerTagline="Run Analysis">
      <section className="container mx-auto px-4 py-6">
        <div className="max-w-7xl mx-auto space-y-6">
          {/* Header */}
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-[#8B4513]/20 border-2 border-[#8B4513]">
              <Database className="h-6 w-6 text-[#8B4513]" />
            </div>
            <div>
              <h1 className="font-display text-2xl text-[#5D4037]">Runs Data Explorer</h1>
              <p className="text-sm text-[#8B4513]/70">View, filter, and analyze all benchmark runs</p>
            </div>
          </div>

          {/* Stats Cards */}
          {!loading && !error && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card variant="retro" className="p-4">
                <div className="text-2xl font-bold text-[#5D4037]">{stats.total}</div>
                <div className="text-sm text-[#8B4513]/70">Total Runs</div>
              </Card>
              <Card variant="retro" className="p-4 bg-green-50">
                <div className="text-2xl font-bold text-green-700">{stats.completed}</div>
                <div className="text-sm text-green-600/70">Completed</div>
              </Card>
              <Card variant="retro" className="p-4 bg-red-50">
                <div className="text-2xl font-bold text-red-700">{stats.failed}</div>
                <div className="text-sm text-red-600/70">Failed</div>
              </Card>
              <Card variant="retro" className="p-4 bg-yellow-50">
                <div className="text-2xl font-bold text-yellow-700">{stats.inProgress}</div>
                <div className="text-sm text-yellow-600/70">In Progress</div>
              </Card>
            </div>
          )}

          {/* Filters */}
          <Card variant="retro" className="p-4">
            <div className="flex flex-wrap gap-4 items-center">
              {/* Search */}
              <div className="relative flex-1 min-w-[200px]">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[#8B4513]/50" />
                <Input
                  placeholder="Search models, errors, conditions..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
              
              {/* Status Filters */}
              <div className="flex gap-2">
                <Button
                  variant={showOnlyFailed ? "default" : "outline"}
                  size="sm"
                  onClick={() => {
                    setShowOnlyFailed(!showOnlyFailed);
                    if (!showOnlyFailed) setShowOnlyCompleted(false);
                  }}
                  className="gap-1"
                >
                  <XCircle className="h-4 w-4" />
                  Failed
                </Button>
                <Button
                  variant={showOnlyCompleted ? "default" : "outline"}
                  size="sm"
                  onClick={() => {
                    setShowOnlyCompleted(!showOnlyCompleted);
                    if (!showOnlyCompleted) setShowOnlyFailed(false);
                  }}
                  className="gap-1"
                >
                  <CheckCircle className="h-4 w-4" />
                  Completed
                </Button>
              </div>
              
              {/* Version Filter */}
              <div className="flex gap-2 items-center">
                <Filter className="h-4 w-4 text-[#8B4513]/50" />
                <select
                  value={selectedVersion ?? ''}
                  onChange={(e) => setSelectedVersion(e.target.value ? Number(e.target.value) : null)}
                  className="border rounded px-2 py-1 text-sm"
                >
                  <option value="">All Versions</option>
                  {versions.map(v => (
                    <option key={v} value={v}>v{v}</option>
                  ))}
                </select>
              </div>
              
              {/* Scaffolding Filter */}
              <select
                value={selectedScaffolding ?? ''}
                onChange={(e) => setSelectedScaffolding(e.target.value || null)}
                className="border rounded px-2 py-1 text-sm"
              >
                <option value="">All Scaffolding</option>
                {scaffoldings.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
            
            {/* Active Filter Count */}
            <div className="mt-2 text-sm text-[#8B4513]/70">
              Showing {filteredRuns.length} of {runs.length} runs
            </div>
          </Card>

          {/* Loading State */}
          {loading && (
            <div className="flex items-center justify-center py-12">
              <div className="bg-white p-4 rounded-xl border-4 border-[#8B4513] shadow-[4px_4px_0_#5D4037]">
                <Loader2 className="h-8 w-8 animate-spin text-[#FF6B35]" />
              </div>
            </div>
          )}

          {/* Error State */}
          {error && (
            <Card variant="retro" className="max-w-md mx-auto">
              <CardContent className="p-8 text-center">
                <AlertCircle className="h-12 w-12 mx-auto text-[#C62828] mb-4" />
                <h3 className="font-display text-lg text-[#5D4037] mb-2">Failed to Load Data</h3>
                <p className="text-[#5D4037]/70 mb-4">{error}</p>
                <Button variant="retro" onClick={() => window.location.reload()}>
                  Retry
                </Button>
              </CardContent>
            </Card>
          )}

          {/* Data Table */}
          {!loading && !error && (
            <div className="overflow-x-auto rounded-xl border-4 border-[#8B4513] shadow-[4px_4px_0_#5D4037]">
              <table className="w-full text-sm">
                <thead className="bg-gradient-to-b from-[#FFFDE7] to-[#FFF9C4]">
                  <tr>
                    <th className="px-4 py-3 text-left font-display text-[#5D4037]">Status</th>
                    <th 
                      className="px-4 py-3 text-left font-display text-[#5D4037] cursor-pointer hover:bg-[#FFE135]/30"
                      onClick={() => handleSort('model_name')}
                    >
                      <div className="flex items-center gap-1">
                        Model
                        <SortIcon field="model_name" />
                      </div>
                    </th>
                    <th className="px-4 py-3 text-left font-display text-[#5D4037]">Conditions</th>
                    <th 
                      className="px-4 py-3 text-right font-display text-[#5D4037] cursor-pointer hover:bg-[#FFE135]/30"
                      onClick={() => handleSort('total_profit')}
                    >
                      <div className="flex items-center justify-end gap-1">
                        Profit
                        <SortIcon field="total_profit" />
                      </div>
                    </th>
                    <th 
                      className="px-4 py-3 text-right font-display text-[#5D4037] cursor-pointer hover:bg-[#FFE135]/30"
                      onClick={() => handleSort('turn_count')}
                    >
                      <div className="flex items-center justify-end gap-1">
                        Turns
                        <SortIcon field="turn_count" />
                      </div>
                    </th>
                    <th 
                      className="px-4 py-3 text-center font-display text-[#5D4037] cursor-pointer hover:bg-[#FFE135]/30"
                      onClick={() => handleSort('taxonomy_version')}
                    >
                      <div className="flex items-center justify-center gap-1">
                        Ver
                        <SortIcon field="taxonomy_version" />
                      </div>
                    </th>
                    <th 
                      className="px-4 py-3 text-left font-display text-[#5D4037] cursor-pointer hover:bg-[#FFE135]/30"
                      onClick={() => handleSort('started_at')}
                    >
                      <div className="flex items-center gap-1">
                        Started
                        <SortIcon field="started_at" />
                      </div>
                    </th>
                    <th className="px-4 py-3 text-left font-display text-[#5D4037]">Error</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-[#8B4513]/20">
                  {filteredRuns.map((run) => (
                    <tr key={run.run_id} className="hover:bg-[#FFFDE7]/50">
                      <td className="px-4 py-3">
                        {run.error_message ? (
                          <Badge variant="destructive" className="gap-1">
                            <XCircle className="h-3 w-3" />
                            Failed
                          </Badge>
                        ) : run.completed_at ? (
                          <Badge variant="default" className="gap-1 bg-green-600">
                            <CheckCircle className="h-3 w-3" />
                            Done
                          </Badge>
                        ) : (
                          <Badge variant="secondary" className="gap-1">
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Running
                          </Badge>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <div className="font-medium text-[#5D4037]">{run.model_name}</div>
                        <div className="text-xs text-[#8B4513]/60">{run.provider} • seed={run.seed}</div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex flex-wrap gap-1">
                          <Badge variant="outline" className="text-xs">{run.goal_framing}</Badge>
                          <Badge variant="outline" className="text-xs">{run.architecture}</Badge>
                          <Badge variant="outline" className="text-xs">{run.scaffolding}</Badge>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        <span className={run.total_profit >= 0 ? 'text-green-700' : 'text-red-700'}>
                          {formatProfit(run.total_profit)}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-[#5D4037]">
                        {run.turn_count}/14
                      </td>
                      <td className="px-4 py-3 text-center">
                        <Badge variant="secondary" className="text-xs">v{run.taxonomy_version || 1}</Badge>
                      </td>
                      <td className="px-4 py-3 text-[#8B4513]/70 text-xs">
                        {formatDate(run.started_at)}
                      </td>
                      <td className="px-4 py-3 max-w-xs">
                        {run.error_message && (
                          <div className="text-xs text-red-600 truncate" title={run.error_message}>
                            {run.error_message.slice(0, 60)}...
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              
              {filteredRuns.length === 0 && (
                <div className="p-8 text-center text-[#8B4513]/70">
                  No runs match your filters
                </div>
              )}
            </div>
          )}
        </div>
      </section>
    </PageLayout>
  );
}

export default DataPage;
