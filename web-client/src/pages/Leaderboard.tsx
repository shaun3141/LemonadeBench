import { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { PageLayout } from '@/components/layout';
import {
  Trophy,
  Loader2,
  AlertCircle,
  TrendingUp,
  RefreshCw,
  Target,
  Brain,
  Wrench,
  Sparkles,
  Cpu,
  Bot,
} from 'lucide-react';
import type { LeaderboardRun } from '@/types';
import { 
  getBestRunsPerModel, 
  getAllRuns,
  getResultsByGoalFraming,
  getResultsByArchitecture,
  getResultsByScaffolding,
  getResultsByModel,
  aggregateRunsByField,
  type AggregatedResult,
} from '@/api';
import { 
  RunDetailsModal,
  FilterPill,
  ModelTiersDisplay,
  GoalFramingsDisplay,
  ArchitecturesDisplay,
  ScaffoldingsDisplay,
  LeaderboardTable,
  runMatchesModel,
  runMatchesTier,
  getModelDisplayName,
  getSelectedTierName,
  getSelectedGoalName,
} from '@/components/leaderboard';
import { ProfitByConditionChart, DistributionBoxPlot, ProfitabilityRateChart } from '@/components/charts';

export function Leaderboard() {
  const [bestRuns, setBestRuns] = useState<LeaderboardRun[]>([]);
  const [allRuns, setAllRuns] = useState<LeaderboardRun[]>([]);
  const [goalFramingResults, setGoalFramingResults] = useState<AggregatedResult[]>([]);
  const [architectureResults, setArchitectureResults] = useState<AggregatedResult[]>([]);
  const [scaffoldingResults, setScaffoldingResults] = useState<AggregatedResult[]>([]);
  const [modelResults, setModelResults] = useState<AggregatedResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [chartsLoading, setChartsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<LeaderboardRun | null>(null);
  
  // Filter state
  const [selectedGoal, setSelectedGoal] = useState<string | null>(null);
  const [selectedTier, setSelectedTier] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null); // Stores model ID (e.g., "openai/gpt-4o-mini")

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setChartsLoading(true);
      setError(null);
      try {
        const [best, all] = await Promise.all([getBestRunsPerModel(), getAllRuns()]);
        setBestRuns(best);
        setAllRuns(all);
        setLoading(false);
        
        // Load aggregated chart data (may take longer)
        const [goalFraming, architecture, scaffolding, byModel] = await Promise.all([
          getResultsByGoalFraming(),
          getResultsByArchitecture(),
          getResultsByScaffolding(),
          getResultsByModel(),
        ]);
        setGoalFramingResults(goalFraming);
        setArchitectureResults(architecture);
        setScaffoldingResults(scaffolding);
        setModelResults(byModel);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load leaderboard');
      } finally {
        setLoading(false);
        setChartsLoading(false);
      }
    }
    loadData();
  }, []);

  // Filter and sort runs for each tab (sorted by profit, best first)
  // Goal Framing: Show all runs (the main study varies goal framing)
  const goalFramingRuns = useMemo(() => {
    return allRuns
      .filter(run => {
        // Base filter for goal framing study
        if (run.architecture !== 'react' || run.scaffolding !== 'none') return false;
        
        // Apply user filters
        if (selectedGoal && run.goal_framing !== selectedGoal) return false;
        if (selectedModel && !runMatchesModel(run.model_name, selectedModel)) return false;
        if (selectedTier && !runMatchesTier(run.model_name, selectedTier)) return false;
        
        return true;
      })
      .sort((a, b) => b.total_profit - a.total_profit);
  }, [allRuns, selectedGoal, selectedModel, selectedTier]);

  // Runs for charts - filtered by model/tier but NOT by selectedGoal (so charts always compare all goals)
  const goalFramingChartRuns = useMemo(() => {
    return allRuns.filter(run => {
      if (run.architecture !== 'react' || run.scaffolding !== 'none') return false;
      if (selectedModel && !runMatchesModel(run.model_name, selectedModel)) return false;
      if (selectedTier && !runMatchesTier(run.model_name, selectedTier)) return false;
      return true;
    });
  }, [allRuns, selectedModel, selectedTier]);

  // Filtered chart data based on model/tier selection
  const filteredGoalFramingResults = useMemo(() => {
    if (!selectedModel && !selectedTier) return goalFramingResults;
    return aggregateRunsByField(goalFramingChartRuns, 'goal_framing', 'baseline');
  }, [goalFramingChartRuns, goalFramingResults, selectedModel, selectedTier]);

  // Architecture: Show runs from the architecture ablation study (varied architectures)
  const architectureRuns = useMemo(() => {
    return allRuns
      .filter(run => {
        // Base filter for architecture study
        if (run.goal_framing !== 'baseline' || run.scaffolding !== 'none') return false;
        
        // Apply user filters
        if (selectedModel && !runMatchesModel(run.model_name, selectedModel)) return false;
        if (selectedTier && !runMatchesTier(run.model_name, selectedTier)) return false;
        
        return true;
      })
      .sort((a, b) => b.total_profit - a.total_profit);
  }, [allRuns, selectedModel, selectedTier]);

  // Filtered chart data for architecture based on model/tier selection
  const filteredArchitectureResults = useMemo(() => {
    if (!selectedModel && !selectedTier) return architectureResults;
    
    const filteredRuns = allRuns.filter(run => {
      if (run.goal_framing !== 'baseline' || run.scaffolding !== 'none') return false;
      if (selectedModel && !runMatchesModel(run.model_name, selectedModel)) return false;
      if (selectedTier && !runMatchesTier(run.model_name, selectedTier)) return false;
      return true;
    });
    
    return aggregateRunsByField(filteredRuns, 'architecture', 'react');
  }, [allRuns, architectureResults, selectedModel, selectedTier]);

  // Scaffolding: Show runs from the scaffolding ablation study (varied scaffolding)
  const scaffoldingRuns = useMemo(() => {
    return allRuns
      .filter(run => {
        // Base filter for scaffolding study
        if (run.goal_framing !== 'baseline' || run.architecture !== 'react') return false;
        
        // Apply user filters
        if (selectedModel && !runMatchesModel(run.model_name, selectedModel)) return false;
        if (selectedTier && !runMatchesTier(run.model_name, selectedTier)) return false;
        
        return true;
      })
      .sort((a, b) => b.total_profit - a.total_profit);
  }, [allRuns, selectedModel, selectedTier]);

  // Filtered chart data for scaffolding based on model/tier selection
  const filteredScaffoldingResults = useMemo(() => {
    if (!selectedModel && !selectedTier) return scaffoldingResults;
    
    const filteredRuns = allRuns.filter(run => {
      if (run.goal_framing !== 'baseline' || run.architecture !== 'react') return false;
      if (selectedModel && !runMatchesModel(run.model_name, selectedModel)) return false;
      if (selectedTier && !runMatchesTier(run.model_name, selectedTier)) return false;
      return true;
    });
    
    return aggregateRunsByField(filteredRuns, 'scaffolding', 'none');
  }, [allRuns, scaffoldingResults, selectedModel, selectedTier]);

  // Best runs filtered
  const filteredBestRuns = useMemo(() => {
    if (!selectedModel && !selectedTier) return bestRuns;
    
    return bestRuns.filter(run => {
      if (selectedModel && !runMatchesModel(run.model_name, selectedModel)) return false;
      if (selectedTier && !runMatchesTier(run.model_name, selectedTier)) return false;
      return true;
    });
  }, [bestRuns, selectedModel, selectedTier]);

  return (
    <PageLayout headerSubtitle="Model Leaderboard" footerTagline="AI Agent Evaluation Research">
      {/* Hero Section */}
      <section className="container mx-auto px-4 py-6 sm:py-8 text-center">
        <div className="max-w-3xl mx-auto">
          <h2 className="font-display text-2xl sm:text-4xl mb-2 text-[#5D4037] drop-shadow-[3px_3px_0_#FFD700]">
            Benchmark Results
          </h2>
        </div>
      </section>

      {/* Main Content */}
      <section className="container mx-auto px-4 pb-8">
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="bg-white p-4 rounded-xl border-4 border-[#8B4513] shadow-[4px_4px_0_#5D4037]">
              <Loader2 className="h-8 w-8 animate-spin text-[#FF6B35]" />
            </div>
          </div>
        )}

        {error && (
          <Card variant="retro" className="max-w-md mx-auto">
            <CardContent className="p-8 text-center">
              <AlertCircle className="h-12 w-12 mx-auto text-[#C62828] mb-4" />
              <h3 className="font-display text-lg text-[#5D4037] mb-2">Failed to Load Leaderboard</h3>
              <p className="text-[#5D4037]/70 mb-4">{error}</p>
              <Button variant="retro" onClick={() => window.location.reload()}>
                Retry
              </Button>
            </CardContent>
          </Card>
        )}

        {!loading && !error && (
          <div className="max-w-6xl mx-auto space-y-6">
            {/* Global Model Filter - Compact */}
            <div className="p-4 rounded-2xl bg-gradient-to-b from-[#FFFDE7] to-[#FFF9C4] border-4 border-[#8B4513] shadow-[4px_4px_0_#5D4037] space-y-3">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm font-semibold text-[#5D4037] flex items-center gap-1.5">
                  <Cpu className="h-4 w-4 text-[#FF6B35]" />
                  Filter by Model:
                </span>
                {selectedModel ? (
                  <FilterPill label={getModelDisplayName(selectedModel)} onClear={() => setSelectedModel(null)} variant="model" />
                ) : selectedTier ? (
                  <FilterPill label={getSelectedTierName(selectedTier)!} onClear={() => setSelectedTier(null)} variant="model" />
                ) : (
                  <span className="text-sm text-[#8B4513]/70">Click a tier to filter</span>
                )}
              </div>
              <ModelTiersDisplay
                selectedTier={selectedTier}
                selectedModel={selectedModel}
                onSelectTier={setSelectedTier}
                onSelectModel={setSelectedModel}
              />
            </div>

            <Tabs defaultValue="goal-framing">
              <TabsList className="mb-6 grid w-full grid-cols-4 bg-gradient-to-b from-[#FFFDE7] to-[#FFF9C4] border-4 border-[#8B4513] rounded-xl p-1 shadow-[4px_4px_0_#5D4037]">
              <TabsTrigger
                value="goal-framing"
                className="gap-1.5 font-display text-[#5D4037] data-[state=active]:bg-gradient-to-b data-[state=active]:from-[#FFE135] data-[state=active]:to-[#FFB300] data-[state=active]:text-[#5D4037] data-[state=active]:shadow-[2px_2px_0_#8B4513] rounded-lg"
              >
                <Target className="h-4 w-4" />
                <span className="hidden sm:inline">Goal Framing</span>
                <span className="sm:hidden">Goals</span>
              </TabsTrigger>
              <TabsTrigger
                value="architecture"
                className="gap-1.5 font-display text-[#5D4037] data-[state=active]:bg-gradient-to-b data-[state=active]:from-[#FFE135] data-[state=active]:to-[#FFB300] data-[state=active]:text-[#5D4037] data-[state=active]:shadow-[2px_2px_0_#8B4513] rounded-lg"
              >
                <Brain className="h-4 w-4" />
                <span className="hidden sm:inline">Architecture</span>
                <span className="sm:hidden">Arch</span>
              </TabsTrigger>
              <TabsTrigger
                value="scaffolding"
                className="gap-1.5 font-display text-[#5D4037] data-[state=active]:bg-gradient-to-b data-[state=active]:from-[#FFE135] data-[state=active]:to-[#FFB300] data-[state=active]:text-[#5D4037] data-[state=active]:shadow-[2px_2px_0_#8B4513] rounded-lg"
              >
                <Wrench className="h-4 w-4" />
                <span className="hidden sm:inline">Scaffolding</span>
                <span className="sm:hidden">Tools</span>
              </TabsTrigger>
              <TabsTrigger
                value="by-model"
                className="gap-1.5 font-display text-[#5D4037] data-[state=active]:bg-gradient-to-b data-[state=active]:from-[#FFE135] data-[state=active]:to-[#FFB300] data-[state=active]:text-[#5D4037] data-[state=active]:shadow-[2px_2px_0_#8B4513] rounded-lg"
              >
                <Bot className="h-4 w-4" />
                <span className="hidden sm:inline">By Model</span>
                <span className="sm:hidden">Models</span>
              </TabsTrigger>
            </TabsList>

            {/* Goal Framing Study Tab */}
            <TabsContent value="goal-framing" className="space-y-6">
              {/* Results Charts with Tab Switcher */}
              <div className="space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                  <h3 className="font-display text-lg text-[#5D4037] flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-[#FF6B35]" />
                    Results by Goal Framing
                  </h3>
                  {selectedModel ? (
                    <FilterPill label={getModelDisplayName(selectedModel)} onClear={() => setSelectedModel(null)} variant="model" />
                  ) : selectedTier ? (
                    <FilterPill label={getSelectedTierName(selectedTier)!} onClear={() => setSelectedTier(null)} variant="model" />
                  ) : (
                    <Badge variant="retro" className="text-xs">All Models</Badge>
                  )}
                </div>
                
                {/* Chart Type Tabs */}
                <Tabs defaultValue="mean-profit" className="w-full">
                  <TabsList className="mb-4 inline-flex bg-gradient-to-b from-[#FFF9C4] to-[#FFECB3] border-2 border-[#8B4513] rounded-lg p-1 shadow-[2px_2px_0_#5D4037]">
                    <TabsTrigger
                      value="mean-profit"
                      className="px-3 py-1.5 text-sm font-medium text-[#5D4037] data-[state=active]:bg-[#FF6B35] data-[state=active]:text-white data-[state=active]:shadow-[1px_1px_0_#8B4513] rounded-md transition-all"
                    >
                      Mean Profit
                    </TabsTrigger>
                    <TabsTrigger
                      value="distribution"
                      className="px-3 py-1.5 text-sm font-medium text-[#5D4037] data-[state=active]:bg-[#FF6B35] data-[state=active]:text-white data-[state=active]:shadow-[1px_1px_0_#8B4513] rounded-md transition-all"
                    >
                      Distribution
                    </TabsTrigger>
                    <TabsTrigger
                      value="success-rate"
                      className="px-3 py-1.5 text-sm font-medium text-[#5D4037] data-[state=active]:bg-[#FF6B35] data-[state=active]:text-white data-[state=active]:shadow-[1px_1px_0_#8B4513] rounded-md transition-all"
                    >
                      Success Rate
                    </TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="mean-profit">
                    <ProfitByConditionChart
                      data={filteredGoalFramingResults}
                      title="Mean Profit by Goal Framing Condition"
                      loading={chartsLoading}
                    />
                  </TabsContent>
                  
                  <TabsContent value="distribution">
                    <DistributionBoxPlot
                      runs={goalFramingChartRuns}
                      title="Profit Distribution by Goal Framing (Box Plot)"
                      loading={chartsLoading}
                    />
                  </TabsContent>
                  
                  <TabsContent value="success-rate">
                    <ProfitabilityRateChart
                      runs={goalFramingChartRuns}
                      title="Profitability Rate by Goal Framing"
                      loading={chartsLoading}
                    />
                  </TabsContent>
                </Tabs>
              </div>

              <Card variant="retro-yellow">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-xl bg-[#FFA000]/20 border-2 border-[#FFA000]">
                      <Target className="h-5 w-5 text-[#FF6B35]" />
                    </div>
                    <CardTitle variant="retro" className="text-[#5D4037]">
                      Goal Framing Study
                    </CardTitle>
                  </div>
                  <CardDescription className="text-base text-[#8B4513]/80">
                    How do motivational prompts affect AI economic decision-making?
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-sm text-[#5D4037]/80">
                    We systematically study how <strong>goal-framing prompts</strong> shape agent behavior over extended
                    decision horizons. While prior work has explored prompt engineering for task completion, we
                    investigate how motivational framing influences risk calibration, loss aversion, and strategic
                    adaptation.
                  </p>
                  <div className="flex flex-wrap gap-2 text-sm">
                    <Badge variant="retro" className="gap-1">
                      <Cpu className="h-3 w-3" />
                      20 models
                    </Badge>
                    <Badge variant="retro" className="gap-1">
                      <Target className="h-3 w-3" />
                      6 goal framings
                    </Badge>
                    <Badge variant="retro" className="gap-1">
                      <RefreshCw className="h-3 w-3" />
                      5 seeds
                    </Badge>
                    <Badge variant="retro" className="gap-1">
                      <Sparkles className="h-3 w-3" />
                      600 episodes
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              <div className="space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                  <h3 className="font-display text-lg text-[#5D4037] flex items-center gap-2">
                    <Target className="h-5 w-5 text-[#FF6B35]" />
                    Goal Framing Conditions
                  </h3>
                  {selectedGoal ? (
                    <FilterPill label={getSelectedGoalName(selectedGoal)!} onClear={() => setSelectedGoal(null)} variant="goal" />
                  ) : (
                    <Badge variant="retro" className="text-xs">All Goals</Badge>
                  )}
                </div>
                <GoalFramingsDisplay
                  selectedGoal={selectedGoal}
                  onSelectGoal={setSelectedGoal}
                />
              </div>

              <div className="space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                  <h3 className="font-display text-lg text-[#5D4037] flex items-center gap-2">
                    <Trophy className="h-5 w-5 text-[#FFD700]" />
                    Best Runs
                  </h3>
                  {/* Show active filters */}
                  {selectedGoal && (
                    <FilterPill label={getSelectedGoalName(selectedGoal)!} onClear={() => setSelectedGoal(null)} variant="goal" />
                  )}
                  {goalFramingRuns.length > 0 && (
                    <Badge variant="retro" className="text-xs">{goalFramingRuns.length} runs</Badge>
                  )}
                </div>
                <LeaderboardTable runs={goalFramingRuns.slice(0, 20)} onSelectRun={setSelectedRun} showExperimentFactors />
              </div>
            </TabsContent>

            {/* Architecture Ablation Tab */}
            <TabsContent value="architecture" className="space-y-6">
              {/* Results Chart */}
              <div className="space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                  <h3 className="font-display text-lg text-[#5D4037] flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-[#1976D2]" />
                    Results by Architecture
                  </h3>
                  {selectedModel ? (
                    <FilterPill label={getModelDisplayName(selectedModel)} onClear={() => setSelectedModel(null)} variant="model" />
                  ) : selectedTier ? (
                    <FilterPill label={getSelectedTierName(selectedTier)!} onClear={() => setSelectedTier(null)} variant="model" />
                  ) : (
                    <Badge variant="retro" className="text-xs">All Models</Badge>
                  )}
                </div>
                <ProfitByConditionChart
                  data={filteredArchitectureResults}
                  title="Mean Profit by Agent Architecture"
                  loading={chartsLoading}
                />
              </div>

              <Card variant="retro-blue">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-xl bg-[#1976D2]/20 border-2 border-[#1976D2]">
                      <Brain className="h-5 w-5 text-[#1976D2]" />
                    </div>
                    <CardTitle variant="retro" className="text-[#0D47A1]">
                      Architecture Ablation Study
                    </CardTitle>
                  </div>
                  <CardDescription className="text-base text-[#0D47A1]/70">
                    Does explicit planning or reflection improve agent performance?
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-sm text-[#0D47A1]/80">
                    Beyond goal framing, we investigate how the <strong>structure of the agent loop</strong> affects
                    performance. We implement four architectures of increasing cognitive complexity, from simple ReAct
                    to full planning + reflection cycles. This tests whether additional compute for reasoning yields
                    proportional improvements.
                  </p>
                  <div className="flex flex-wrap gap-2 text-sm">
                    <Badge variant="retro-blue" className="gap-1">
                      <Cpu className="h-3 w-3" />
                      4 representative models
                    </Badge>
                    <Badge variant="retro-blue" className="gap-1">
                      <Brain className="h-3 w-3" />
                      4 architectures
                    </Badge>
                    <Badge variant="retro-blue" className="gap-1">
                      <RefreshCw className="h-3 w-3" />
                      10 seeds
                    </Badge>
                    <Badge variant="retro-blue" className="gap-1">
                      <Sparkles className="h-3 w-3" />
                      160 episodes
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              <div className="space-y-4">
                <h3 className="font-display text-lg text-[#5D4037] flex items-center gap-2">
                  <Brain className="h-5 w-5 text-[#1976D2]" />
                  Agent Architectures
                </h3>
                <ArchitecturesDisplay />
              </div>

              <Card variant="retro">
                <CardHeader>
                  <CardTitle variant="retro" className="text-base text-[#5D4037]">
                    Key Research Questions
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2 text-sm text-[#5D4037]/80">
                    <li className="flex items-start gap-2">
                      <span className="font-pixel text-[#FF6B35]">H6:</span>
                      Does explicit planning (Plan-Act) improve inventory management and reduce spoilage?
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-pixel text-[#FF6B35]">H7:</span>
                      Does reflection (Act-Reflect) show stronger improvement in the second half of the season?
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-pixel text-[#FF6B35]">H8:</span>
                      Does the Full architecture justify its 3× API cost with proportional benefits?
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-pixel text-[#FF6B35]">H9:</span>
                      Are explicit plans followed, or abandoned when conditions change unexpectedly?
                    </li>
                  </ul>
                </CardContent>
              </Card>

              <div className="space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                  <h3 className="font-display text-lg text-[#5D4037] flex items-center gap-2">
                    <Trophy className="h-5 w-5 text-[#FFD700]" />
                    Best Runs
                  </h3>
                  {architectureRuns.length > 0 && (
                    <Badge variant="retro-blue" className="text-xs">{architectureRuns.length} runs</Badge>
                  )}
                </div>
                <LeaderboardTable runs={architectureRuns.slice(0, 20)} onSelectRun={setSelectedRun} showExperimentFactors />
              </div>
            </TabsContent>

            {/* Scaffolding Ablation Tab */}
            <TabsContent value="scaffolding" className="space-y-6">
              {/* Results Chart */}
              <div className="space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                  <h3 className="font-display text-lg text-[#5D4037] flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-[#388E3C]" />
                    Results by Scaffolding
                  </h3>
                  {selectedModel ? (
                    <FilterPill label={getModelDisplayName(selectedModel)} onClear={() => setSelectedModel(null)} variant="model" />
                  ) : selectedTier ? (
                    <FilterPill label={getSelectedTierName(selectedTier)!} onClear={() => setSelectedTier(null)} variant="model" />
                  ) : (
                    <Badge variant="retro" className="text-xs">All Models</Badge>
                  )}
                </div>
                <ProfitByConditionChart
                  data={filteredScaffoldingResults}
                  title="Mean Profit by Cognitive Scaffolding"
                  loading={chartsLoading}
                />
              </div>

              <Card variant="retro-green">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-xl bg-[#388E3C]/20 border-2 border-[#388E3C]">
                      <Wrench className="h-5 w-5 text-[#388E3C]" />
                    </div>
                    <CardTitle variant="retro" className="text-[#1B5E20]">
                      Cognitive Scaffolding Study
                    </CardTitle>
                  </div>
                  <CardDescription className="text-base text-[#1B5E20]/70">
                    Do external tools help LLMs make better business decisions?
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-sm text-[#1B5E20]/80">
                    We test whether providing explicit <strong>cognitive tools</strong> improves agent reasoning. This
                    includes calculator access for arithmetic, prompts encouraging step-by-step math, and full Python
                    code execution for complex analysis like Monte Carlo simulations.
                  </p>
                  <div className="flex flex-wrap gap-2 text-sm">
                    <Badge variant="retro-green" className="gap-1">
                      <Cpu className="h-3 w-3" />
                      4 representative models
                    </Badge>
                    <Badge variant="retro-green" className="gap-1">
                      <Wrench className="h-3 w-3" />
                      4 scaffolding types
                    </Badge>
                    <Badge variant="retro-green" className="gap-1">
                      <RefreshCw className="h-3 w-3" />
                      10 seeds
                    </Badge>
                    <Badge variant="retro-green" className="gap-1">
                      <Sparkles className="h-3 w-3" />
                      160 episodes
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              <div className="space-y-4">
                <h3 className="font-display text-lg text-[#5D4037] flex items-center gap-2">
                  <Wrench className="h-5 w-5 text-[#388E3C]" />
                  Scaffolding Conditions
                </h3>
                <ScaffoldingsDisplay />
              </div>

              <Card variant="retro">
                <CardHeader>
                  <CardTitle variant="retro" className="text-base text-[#5D4037]">
                    Key Research Questions
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2 text-sm text-[#5D4037]/80">
                    <li className="flex items-start gap-2">
                      <span className="font-pixel text-[#FF6B35]">H10:</span>
                      Does calculator access reduce pricing errors (prices too high or too low for conditions)?
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-pixel text-[#FF6B35]">H11:</span>
                      Do math encouragement prompts improve performance even without external tools?
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-pixel text-[#FF6B35]">H12:</span>
                      Is code interpreter access utilized, or do agents rarely write optimization code?
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="font-pixel text-[#FF6B35]">H13:</span>
                      Do all LLM agents underperform the Reactive baseline on inventory management?
                    </li>
                  </ul>
                </CardContent>
              </Card>

              <div className="space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                  <h3 className="font-display text-lg text-[#5D4037] flex items-center gap-2">
                    <Trophy className="h-5 w-5 text-[#FFD700]" />
                    Best Runs
                  </h3>
                  {scaffoldingRuns.length > 0 && (
                    <Badge variant="retro-green" className="text-xs">{scaffoldingRuns.length} runs</Badge>
                  )}
                </div>
                <LeaderboardTable runs={scaffoldingRuns.slice(0, 20)} onSelectRun={setSelectedRun} showExperimentFactors />
              </div>
            </TabsContent>

            {/* By Model Tab */}
            <TabsContent value="by-model" className="space-y-6">
              {/* Results Chart */}
              <div className="space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                  <h3 className="font-display text-lg text-[#5D4037] flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-[#8B4513]" />
                    Mean Profit by Model
                  </h3>
                  <Badge variant="retro" className="text-xs">Across all conditions</Badge>
                </div>
                <ProfitByConditionChart
                  data={modelResults}
                  title="Mean Profit by Model (All Runs)"
                  loading={chartsLoading}
                />
              </div>

              <Card variant="retro">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-xl bg-[#8B4513]/20 border-2 border-[#8B4513]">
                      <Bot className="h-5 w-5 text-[#8B4513]" />
                    </div>
                    <CardTitle variant="retro" className="text-[#5D4037]">
                      Model Performance Comparison
                    </CardTitle>
                  </div>
                  <CardDescription className="text-base text-[#5D4037]/70">
                    Aggregated results across all experimental conditions
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-[#5D4037]/80">
                    This view shows mean profit for each model across all goal framings, architectures, and scaffolding 
                    conditions. Use this to compare overall model capability regardless of experimental setup.
                  </p>
                </CardContent>
              </Card>

              {/* Best Runs Per Model */}
              <div className="space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                  <h3 className="font-display text-lg text-[#5D4037] flex items-center gap-2">
                    <Trophy className="h-5 w-5 text-[#FFD700]" />
                    Best Run Per Model
                  </h3>
                  {filteredBestRuns.length > 0 && (
                    <Badge variant="retro" className="text-xs">{filteredBestRuns.length} models</Badge>
                  )}
                </div>
                <LeaderboardTable runs={filteredBestRuns} onSelectRun={setSelectedRun} showExperimentFactors />
              </div>
            </TabsContent>
            </Tabs>
          </div>
        )}
      </section>

      {/* Run Details Modal */}
      {selectedRun && <RunDetailsModal run={selectedRun} onClose={() => setSelectedRun(null)} />}
    </PageLayout>
  );
}

export default Leaderboard;
