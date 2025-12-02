/**
 * API Client - All endpoints wired to Python backend
 * NO FALLBACKS - Everything fetches from real backend
 */

const API_BASE = '/api';

export class APIError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'APIError';
  }
}

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });
  
  if (!res.ok) {
    throw new APIError(res.status, `API error: ${res.status} ${res.statusText}`);
  }
  
  return res.json();
}

// ============================================================================
// CORE DATA ENDPOINTS
// ============================================================================

export async function getBenchmarkData() {
  return fetchAPI('/data');
}

export async function getGpuInfo() {
  return fetchAPI('/gpu');
}

export async function getGpuHistory() {
  return fetchAPI('/gpu/history');
}

// GPU metrics SSE stream URL
export function getGpuStreamUrl() {
  return `${API_BASE}/gpu/stream`;
}

export async function getSoftwareInfo() {
  return fetchAPI('/software');
}

export async function getDependencies() {
  return fetchAPI('/deps');
}

export async function checkDependencyUpdates() {
  return fetchAPI('/deps/check-updates');
}

export async function getSystemContext() {
  return fetchAPI('/system-context');
}

export async function getTargets(): Promise<string[]> {
  return fetchAPI('/targets');
}

export async function getAvailableBenchmarks() {
  return fetchAPI('/available');
}

export async function scanAllBenchmarks() {
  return fetchAPI('/scan-all');
}

// Bench root configuration
export async function getBenchRootConfig() {
  return fetchAPI('/config/bench-root');
}

export async function setBenchRootConfig(payload: { bench_root?: string; data_file?: string | null }) {
  return fetchAPI('/config/bench-root', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

// Quick benchmark runner
export async function runBenchmark(
  chapter: string,
  name: string,
  options?: { run_baseline?: boolean; run_optimized?: boolean; precheck_only?: boolean; dry_run?: boolean; timeout_seconds?: number }
) {
  return fetchAPI('/benchmark/run', {
    method: 'POST',
    body: JSON.stringify({
      chapter,
      name,
      run_baseline: options?.run_baseline ?? true,
      run_optimized: options?.run_optimized ?? true,
      precheck_only: options?.precheck_only ?? false,
      dry_run: options?.dry_run ?? false,
      timeout_seconds: options?.timeout_seconds,
    }),
  });
}

export async function verifyBenchmark(
  chapter: string,
  name: string,
  options?: { precheck_only?: boolean; dry_run?: boolean; timeout_seconds?: number }
) {
  return fetchAPI('/benchmark/verify', {
    method: 'POST',
    body: JSON.stringify({
      chapter,
      name,
      precheck_only: options?.precheck_only ?? false,
      dry_run: options?.dry_run ?? false,
      timeout_seconds: options?.timeout_seconds,
    }),
  });
}

// Baseline vs optimized code diff
export async function getCodeDiff(chapter: string, name: string) {
  return fetchAPI(`/code-diff/${encodeURIComponent(chapter)}/${encodeURIComponent(name)}`);
}

// ============================================================================
// EXPLANATION ENDPOINTS (Book + LLM)
// ============================================================================

/**
 * Get explanation from book content (book/ch*.md files)
 * @param technique - The technique name to explain
 * @param chapter - Optional chapter reference (e.g., "ch08")
 */
export async function getBookExplanation(technique: string, chapter?: string) {
  const path = chapter 
    ? `/explain/${encodeURIComponent(technique)}/${chapter}`
    : `/explain/${encodeURIComponent(technique)}`;
  return fetchAPI(path);
}

/**
 * Get LLM-enhanced explanation with book content + hardware context
 * @param technique - The technique name
 * @param chapter - Chapter reference
 * @param benchmark - Benchmark name for context
 */
export async function getLLMExplanation(technique: string, chapter: string, benchmark: string) {
  return fetchAPI(`/explain-llm/${encodeURIComponent(technique)}/${chapter}/${encodeURIComponent(benchmark)}`);
}

// ============================================================================
// LLM ANALYSIS ENDPOINTS
// ============================================================================

export async function getLLMAnalysis() {
  return fetchAPI('/llm-analysis');
}

export async function getLLMStatus() {
  return fetchAPI('/llm/status');
}

export async function analyzeLLMBottlenecks(data: unknown) {
  return fetchAPI('/llm/analyze-bottlenecks', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getLLMDistributed(data: unknown) {
  return fetchAPI('/llm/distributed', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getLLMInference(data: unknown) {
  return fetchAPI('/llm/inference', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getLLMRLHF(data: unknown) {
  return fetchAPI('/llm/rlhf', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getLLMCustomQuery(query: string) {
  return fetchAPI('/llm/custom-query', {
    method: 'POST',
    body: JSON.stringify({ query }),
  });
}

export async function getLLMAdvisor(data: unknown) {
  return fetchAPI('/llm/advisor', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getAISuggestions() {
  return fetchAPI('/ai/suggest');
}

export async function getAIContext() {
  return fetchAPI('/ai/context');
}

export async function runAIAnalysis(type?: string) {
  const search = new URLSearchParams();
  if (type) search.set('type', type);
  const qs = search.toString();
  return fetchAPI(`/ai/analyze${qs ? `?${qs}` : ''}`);
}

// ============================================================================
// Nsight / Profiling Helpers
// ============================================================================

export async function startNsightSystemsCapture(payload: {
  command: string;
  preset?: 'light' | 'full';
  full_timeline?: boolean;
  async?: boolean;
  timeout_seconds?: number;
}) {
  return fetchAPI('/nsight/profile/nsys', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function startNsightComputeCapture(payload: {
  command: string;
  workload_type?: string;
  async?: boolean;
  timeout_seconds?: number;
}) {
  return fetchAPI('/nsight/profile/ncu', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function startTorchProfilerCapture(payload: {
  script: string;
  mode?: string;
  output_name?: string;
  output_dir?: string;
  nvtx_label?: string;
  force_lineinfo?: boolean;
  use_nvtx?: boolean;
  async?: boolean;
  timeout_seconds?: number;
  script_args?: string[];
}) {
  return fetchAPI('/profiler/torch', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function startHTACapture(payload: {
  command: string;
  preset?: string;
  output_name?: string;
  output_dir?: string;
  force_lineinfo?: boolean;
  async?: boolean;
  timeout_seconds?: number;
}) {
  return fetchAPI('/profiler/hta-capture', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

// UNIFIED job status - checks both Dashboard AND MCP job stores
export async function fetchJobStatus(job_id: string) {
  const search = new URLSearchParams({ job_id });
  return fetchAPI(`/job-status?${search.toString()}`);
}

// Legacy: Dashboard profile jobs only (nsys, ncu, torch, hta started via Dashboard UI)
export async function fetchNsightJobStatus(job_id: string) {
  const search = new URLSearchParams({ job_id });
  return fetchAPI(`/nsight/job-status?${search.toString()}`);
}

// MCP job status only (jobs started via MCP tools)
export async function fetchMcpJobStatus(job_id: string) {
  const search = new URLSearchParams({ job_id });
  return fetchAPI(`/mcp/job-status?${search.toString()}`);
}

// Benchmark triage - post-benchmark analysis and recommendations
export async function getBenchmarkTriage() {
  return fetchAPI('/benchmark/triage');
}

export async function runAIQuery(question: string, context?: string) {
  return fetchAPI('/ai/query', {
    method: 'POST',
    body: JSON.stringify({ question, context }),
  });
}

// ============================================================================
// PROFILER ENDPOINTS
// ============================================================================

export async function getProfilerFlame() {
  return fetchAPI('/profiler/flame');
}

export async function getProfilerMemory() {
  return fetchAPI('/profiler/memory');
}

export async function getProfilerTimeline() {
  return fetchAPI('/profiler/timeline');
}

export async function getProfilerKernels() {
  return fetchAPI('/profiler/kernels');
}

export async function getProfilerHTA() {
  return fetchAPI('/profiler/hta');
}

export async function getProfilerTorch() {
  return fetchAPI('/profiler/torch');
}

export async function getProfilerCompile() {
  return fetchAPI('/profiler/compile');
}

export async function getProfilerRoofline() {
  return fetchAPI('/profiler/roofline');
}

export async function getProfilerBottlenecks() {
  return fetchAPI('/analysis/bottlenecks');
}

export async function getOptimizationScore() {
  return fetchAPI('/profiler/optimization-score');
}

export async function analyzeKernel(data: unknown) {
  return fetchAPI('/profiler/analyze-kernel', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function generatePatch(data: unknown) {
  return fetchAPI('/profiler/generate-patch', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function askProfiler(data: unknown) {
  return fetchAPI('/profiler/ask', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// ============================================================================
// DEEP PROFILE ENDPOINTS
// ============================================================================

export async function getDeepProfileList() {
  return fetchAPI('/deep-profile/list');
}

export async function getDeepProfileRecommendations() {
  return fetchAPI('/deep-profile/recommendations');
}

export async function getDeepProfileCompare(chapter: string) {
  return fetchAPI(`/deep-profile/compare/${encodeURIComponent(chapter)}`);
}

export async function getFlameGraphComparison(chapter: string) {
  return fetchAPI(`/deep-profile/flamegraph/${encodeURIComponent(chapter)}`);
}

// ============================================================================
// ROOFLINE ENDPOINTS
// ============================================================================

export async function getRooflineInteractive() {
  return fetchAPI('/roofline/interactive');
}

export async function getHardwareCapabilities() {
  return fetchAPI('/hardware-capabilities');
}

// ============================================================================
// ANALYSIS ENDPOINTS
// ============================================================================

export async function getAnalysisPareto() {
  return fetchAPI('/analysis/pareto');
}

export async function getAnalysisTradeoffs() {
  return fetchAPI('/analysis/tradeoffs');
}

export async function getAnalysisRecommendations() {
  return fetchAPI('/analysis/recommendations');
}

export async function getAnalysisBottlenecks() {
  return fetchAPI('/analysis/bottlenecks');
}

export async function getAnalysisLeaderboards() {
  return fetchAPI('/analysis/leaderboards');
}

export async function getAnalysisStacking() {
  return fetchAPI('/analysis/stacking');
}

export async function getAnalysisPower() {
  return fetchAPI('/analysis/power');
}

export async function getAnalysisScaling() {
  return fetchAPI('/analysis/scaling');
}

export async function getAnalysisCost(params?: { gpu?: string; rate?: number }) {
  const search = new URLSearchParams();
  if (params?.gpu) search.set('gpu', params.gpu);
  if (params?.rate !== undefined) search.set('rate', params.rate.toString());
  const qs = search.toString();
  return fetchAPI(`/analysis/cost${qs ? `?${qs}` : ''}`);
}

export async function getAnalysisCpuMemory() {
  return fetchAPI('/analysis/cpu-memory');
}

export async function getAnalysisSystemParams() {
  return fetchAPI('/analysis/system-params');
}

export async function getAnalysisContainerLimits() {
  return fetchAPI('/analysis/container-limits');
}

export async function getAnalysisFullSystem() {
  return fetchAPI('/analysis/full-system');
}

export async function getAnalysisOptimizations() {
  return fetchAPI('/analysis/optimizations');
}

export async function getAnalysisPlaybooks() {
  return fetchAPI('/analysis/playbooks');
}

export async function getAnalysisWhatIf(params?: { vram?: number; latency?: number; throughput?: number }) {
  const search = new URLSearchParams();
  if (params?.vram !== undefined) search.set('vram', String(params.vram));
  if (params?.latency !== undefined) search.set('latency', String(params.latency));
  if (params?.throughput !== undefined) search.set('throughput', String(params.throughput));
  const qs = search.toString();
  return fetchAPI(`/analysis/whatif${qs ? `?${qs}` : ''}`);
}

export async function getAnalysisWarpDivergence(code: string) {
  const search = new URLSearchParams();
  if (code) search.set('code', code);
  const qs = search.toString();
  return fetchAPI(`/analysis/warp-divergence${qs ? `?${qs}` : ''}`);
}

export async function getAnalysisBankConflicts(params?: { stride?: number; element_size?: number }) {
  const search = new URLSearchParams();
  if (params?.stride !== undefined) search.set('stride', String(params.stride));
  if (params?.element_size !== undefined) search.set('element_size', String(params.element_size));
  const qs = search.toString();
  return fetchAPI(`/analysis/bank-conflicts${qs ? `?${qs}` : ''}`);
}

export async function getAnalysisMemoryAccess(params?: { stride?: number; element_size?: number }) {
  const search = new URLSearchParams();
  if (params?.stride !== undefined) search.set('stride', String(params.stride));
  if (params?.element_size !== undefined) search.set('element_size', String(params.element_size));
  const qs = search.toString();
  return fetchAPI(`/analysis/memory-access${qs ? `?${qs}` : ''}`);
}

export async function getAnalysisAutoTune(params?: { kernel?: string; max_configs?: number }) {
  const search = new URLSearchParams();
  if (params?.kernel) search.set('kernel', params.kernel);
  if (params?.max_configs !== undefined) search.set('max_configs', String(params.max_configs));
  const qs = search.toString();
  return fetchAPI(`/analysis/auto-tune${qs ? `?${qs}` : ''}`);
}

export async function getAnalysisPredictScaling(params?: { from_gpu?: string; to_gpu?: string; workload?: string }) {
  const search = new URLSearchParams();
  if (params?.from_gpu) search.set('from', params.from_gpu);
  if (params?.to_gpu) search.set('to', params.to_gpu);
  if (params?.workload) search.set('workload', params.workload);
  const qs = search.toString();
  return fetchAPI(`/analysis/predict-scaling${qs ? `?${qs}` : ''}`);
}

export async function getAnalysisEnergy(params?: { gpu?: string; power_limit?: number }) {
  const search = new URLSearchParams();
  if (params?.gpu) search.set('gpu', params.gpu);
  if (params?.power_limit !== undefined) search.set('power_limit', String(params.power_limit));
  const qs = search.toString();
  return fetchAPI(`/analysis/energy${qs ? `?${qs}` : ''}`);
}

export async function getAnalysisMultiGpuScaling(params?: { gpus?: number; nvlink?: boolean; workload?: string }) {
  const search = new URLSearchParams();
  if (params?.gpus !== undefined) search.set('gpus', String(params.gpus));
  if (params?.nvlink !== undefined) search.set('nvlink', String(params.nvlink));
  if (params?.workload) search.set('workload', params.workload);
  const qs = search.toString();
  return fetchAPI(`/analysis/multi-gpu-scaling${qs ? `?${qs}` : ''}`);
}

export async function getAnalysisCompound(opts: string[]) {
  const search = new URLSearchParams();
  if (opts?.length) search.set('opts', opts.join(','));
  const qs = search.toString();
  return fetchAPI(`/analysis/compound${qs ? `?${qs}` : ''}`);
}

export async function getAnalysisOptimalStack(params?: { target?: number; difficulty?: string }) {
  const search = new URLSearchParams();
  if (params?.target !== undefined) search.set('target', String(params.target));
  if (params?.difficulty) search.set('difficulty', params.difficulty);
  const qs = search.toString();
  return fetchAPI(`/analysis/optimal-stack${qs ? `?${qs}` : ''}`);
}

export async function getAnalysisOccupancy(params?: { threads?: number; shared?: number; registers?: number }) {
  const search = new URLSearchParams();
  if (params?.threads !== undefined) search.set('threads', String(params.threads));
  if (params?.shared !== undefined) search.set('shared', String(params.shared));
  if (params?.registers !== undefined) search.set('registers', String(params.registers));
  const qs = search.toString();
  return fetchAPI(`/analysis/occupancy${qs ? `?${qs}` : ''}`);
}

// ============================================================================
// COST & EFFICIENCY ENDPOINTS
// ============================================================================

export async function getCostCalculator() {
  return fetchAPI('/cost/calculator');
}

export async function getCostROI() {
  return fetchAPI('/cost/roi');
}

export async function getCostSavingsHeader(opsPerDay?: number) {
  const search = new URLSearchParams();
  if (opsPerDay !== undefined) search.set('ops_per_day', opsPerDay.toString());
  const qs = search.toString();
  return fetchAPI(`/cost/savings-header${qs ? `?${qs}` : ''}`);
}

export async function getEfficiencyKernels() {
  return fetchAPI('/efficiency/kernels');
}

export async function simulateWhatIf(data: unknown) {
  return fetchAPI('/whatif/simulate', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// ============================================================================
// GPU ENDPOINTS
// ============================================================================

export async function getGpuTopology() {
  return fetchAPI('/gpu/topology');
}

export async function getGpuNvlink() {
  return fetchAPI('/gpu/nvlink');
}

export async function getGpuControl() {
  return fetchAPI('/gpu/control');
}

export async function getCudaEnvironment() {
  return fetchAPI('/cuda/environment');
}

export async function setGpuPowerLimit(power_limit: number) {
  return fetchAPI('/gpu/power-limit', {
    method: 'POST',
    body: JSON.stringify({ power_limit }),
  });
}

export async function setGpuClockPin(pin: boolean) {
  return fetchAPI('/gpu/clock-pin', {
    method: 'POST',
    body: JSON.stringify({ pin }),
  });
}

export async function setGpuPersistence(enabled: boolean) {
  return fetchAPI('/gpu/persistence', {
    method: 'POST',
    body: JSON.stringify({ enabled }),
  });
}

export async function applyGpuPreset(preset: 'max' | 'balanced' | 'quiet') {
  return fetchAPI('/gpu/preset', {
    method: 'POST',
    body: JSON.stringify({ preset }),
  });
}

// ============================================================================
// HISTORY ENDPOINTS
// ============================================================================

export async function getHistoryRuns() {
  return fetchAPI('/history/runs');
}

export async function getHistoryTrends() {
  return fetchAPI('/history/trends');
}

// ============================================================================
// OPTIMIZATION ENDPOINTS
// ============================================================================

export async function getOptimizeJobs() {
  return fetchAPI('/optimize/jobs');
}

export async function startOptimization(target: string) {
  return fetchAPI('/optimize/start', {
    method: 'POST',
    body: JSON.stringify({ target }),
  });
}

export async function stopOptimization(jobId: string) {
  return fetchAPI('/optimize/stop', {
    method: 'POST',
    body: JSON.stringify({ job_id: jobId }),
  });
}

// SSE stream for live optimization
export function subscribeToOptimization(jobId: string, onMessage: (data: unknown) => void) {
  const eventSource = new EventSource(`${API_BASE}/optimize/stream/${jobId}`);
  
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error('Failed to parse SSE message:', e);
    }
  };
  
  eventSource.onerror = () => {
    eventSource.close();
  };
  
  return () => eventSource.close();
}

// ============================================================================
// BATCH OPTIMIZATION ENDPOINTS
// ============================================================================

export async function batchOptimize(data: unknown) {
  return fetchAPI('/batch/optimize', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getModelsThatFit() {
  return fetchAPI('/batch/models-that-fit');
}

export async function calculateBatch(params: unknown) {
  return fetchAPI('/batch/calculate', {
    method: 'POST',
    body: JSON.stringify(params || {}),
  });
}

export async function getQuantizationComparison(params?: unknown) {
  return fetchAPI('/batch/quantization', {
    method: 'POST',
    body: JSON.stringify(params || {}),
  });
}

export async function getBatchThroughput(params?: { params?: number; precision?: string }) {
  const search = new URLSearchParams();
  if (params?.params !== undefined) search.set('params', String(params.params));
  if (params?.precision) search.set('precision', params.precision);
  const qs = search.toString();
  return fetchAPI(`/batch/throughput${qs ? `?${qs}` : ''}`);
}

export async function getBatchCloudCost(payload?: unknown) {
  return fetchAPI('/batch/cloud-cost', {
    method: 'POST',
    body: JSON.stringify(payload || {}),
  });
}

export async function getBatchDeployConfig(payload?: unknown) {
  return fetchAPI('/batch/deploy-config', {
    method: 'POST',
    body: JSON.stringify(payload || {}),
  });
}

export async function getBatchFinetuneEstimate(payload?: unknown) {
  return fetchAPI('/batch/finetune', {
    method: 'POST',
    body: JSON.stringify(payload || {}),
  });
}

export async function getBatchMultiGpuScaling(payload?: unknown) {
  return fetchAPI('/batch/multi-gpu', {
    method: 'POST',
    body: JSON.stringify(payload || {}),
  });
}

export async function getBatchLLMAdvisor(payload?: unknown) {
  return fetchAPI('/batch/llm-advisor', {
    method: 'POST',
    body: JSON.stringify(payload || {}),
  });
}

export async function getBatchCompound(payload?: unknown) {
  return fetchAPI('/batch/compound', {
    method: 'POST',
    body: JSON.stringify(payload || {}),
  });
}

// ============================================================================
// PARALLELISM/DISTRIBUTED ENDPOINTS
// ============================================================================

export async function getParallelismTopology() {
  return fetchAPI('/parallelism/topology');
}

export async function getParallelismPresets() {
  return fetchAPI('/parallelism/presets');
}

export async function getParallelismClusters() {
  return fetchAPI('/parallelism/clusters');
}

export async function getParallelismCalibration() {
  return fetchAPI('/parallelism/calibration');
}

export async function getParallelismPareto() {
  return fetchAPI('/parallelism/pareto');
}

export async function getParallelismProfiles() {
  return fetchAPI('/parallelism/profiles');
}

export async function getParallelismTroubleshootTopics() {
  return fetchAPI('/parallelism/troubleshoot/topics');
}

export async function recommendParallelism(data: unknown) {
  return fetchAPI('/parallelism/recommend', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getShardingPlan(data: unknown) {
  return fetchAPI('/parallelism/sharding', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getParallelismEstimate(params: {
  model?: string;
  tokens?: number;
  throughput?: number;
  gpus?: number;
  gpu_cost?: number;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.tokens !== undefined) search.set('tokens', String(params.tokens));
  if (params.throughput !== undefined) search.set('throughput', String(params.throughput));
  if (params.gpus !== undefined) search.set('gpus', String(params.gpus));
  if (params.gpu_cost !== undefined) search.set('cost', String(params.gpu_cost));
  const qs = search.toString();
  return fetchAPI(`/parallelism/estimate${qs ? `?${qs}` : ''}`);
}

export async function getParallelismCompare(models: string[]) {
  const search = new URLSearchParams();
  if (models.length > 0) {
    search.set('models', models.join(','));
  }
  const qs = search.toString();
  return fetchAPI(`/parallelism/compare${qs ? `?${qs}` : ''}`);
}

export async function getParallelismSlurm(params: { model?: string; nodes?: number; gpus?: number; framework?: string }) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.nodes !== undefined) search.set('nodes', String(params.nodes));
  if (params.gpus !== undefined) search.set('gpus', String(params.gpus));
  if (params.framework) search.set('framework', params.framework);
  const qs = search.toString();
  return fetchAPI(`/parallelism/slurm${qs ? `?${qs}` : ''}`);
}

export async function getParallelismWhatif(params: { model?: string; tp?: number; pp?: number; dp?: number; batch_size?: number }) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.tp !== undefined) search.set('tp', String(params.tp));
  if (params.pp !== undefined) search.set('pp', String(params.pp));
  if (params.dp !== undefined) search.set('dp', String(params.dp));
  if (params.batch_size !== undefined) search.set('batch', String(params.batch_size));
  const qs = search.toString();
  return fetchAPI(`/parallelism/whatif${qs ? `?${qs}` : ''}`);
}

export async function getParallelismMemory(params: {
  model?: string;
  batch_size?: number;
  seq_length?: number;
  tp?: number;
  pp?: number;
  dp?: number;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.batch_size !== undefined) search.set('batch', String(params.batch_size));
  if (params.seq_length !== undefined) search.set('seq', String(params.seq_length));
  if (params.tp !== undefined) search.set('tp', String(params.tp));
  if (params.pp !== undefined) search.set('pp', String(params.pp));
  if (params.dp !== undefined) search.set('dp', String(params.dp));
  const qs = search.toString();
  return fetchAPI(`/parallelism/memory${qs ? `?${qs}` : ''}`);
}

export async function getParallelismVLLM(params: {
  model?: string;
  goal?: string;
  target?: string;
  gpus?: number;
  max_seq_len?: number;
  max_seq_length?: number;
  compare?: boolean;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.goal) search.set('goal', params.goal);
  if (params.target) search.set('target', params.target);
  if (params.gpus !== undefined) search.set('gpus', String(params.gpus));
  const seq = params.max_seq_len ?? params.max_seq_length;
  if (seq !== undefined) search.set('seq', String(seq));
  if (params.compare !== undefined) search.set('compare', String(params.compare));
  const qs = search.toString();
  return fetchAPI(`/parallelism/vllm${qs ? `?${qs}` : ''}`);
}

export async function getParallelismBottleneck(params: {
  model?: string;
  batch_size?: number;
  seq_length?: number;
  tp?: number;
  pp?: number;
  dp?: number;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.batch_size !== undefined) search.set('batch', String(params.batch_size));
  if (params.seq_length !== undefined) search.set('seq', String(params.seq_length));
  if (params.tp !== undefined) search.set('tp', String(params.tp));
  if (params.pp !== undefined) search.set('pp', String(params.pp));
  if (params.dp !== undefined) search.set('dp', String(params.dp));
  const qs = search.toString();
  return fetchAPI(`/parallelism/bottleneck${qs ? `?${qs}` : ''}`);
}

export async function getParallelismScaling(params: {
  model?: string;
  throughput?: number;
  gpus?: number;
  max_gpus?: number;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.throughput !== undefined) search.set('throughput', String(params.throughput));
  if (params.gpus !== undefined) search.set('gpus', String(params.gpus));
  if (params.max_gpus !== undefined) search.set('max_gpus', String(params.max_gpus));
  const qs = search.toString();
  return fetchAPI(`/parallelism/scaling${qs ? `?${qs}` : ''}`);
}

export async function getParallelismBatchSize(params: {
  model?: string;
  seq_length?: number;
  tp?: number;
  pp?: number;
  dp?: number;
  target_batch?: number;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.seq_length !== undefined) search.set('seq', String(params.seq_length));
  if (params.tp !== undefined) search.set('tp', String(params.tp));
  if (params.pp !== undefined) search.set('pp', String(params.pp));
  if (params.dp !== undefined) search.set('dp', String(params.dp));
  if (params.target_batch !== undefined) search.set('target', String(params.target_batch));
  const qs = search.toString();
  return fetchAPI(`/parallelism/batch-size${qs ? `?${qs}` : ''}`);
}

export async function getParallelismAutoTune(params: { model?: string; goal?: string; target_batch?: number }) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.goal) search.set('goal', params.goal);
  if (params.target_batch !== undefined) search.set('target', String(params.target_batch));
  const qs = search.toString();
  return fetchAPI(`/parallelism/auto-tune${qs ? `?${qs}` : ''}`);
}

export async function getParallelismExport(params: {
  model?: string;
  nodes?: number;
  gpus?: number;
  tp?: number;
  pp?: number;
  dp?: number;
  batch_size?: number;
  zero_stage?: number;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.nodes !== undefined) search.set('nodes', String(params.nodes));
  if (params.gpus !== undefined) search.set('gpus', String(params.gpus));
  if (params.tp !== undefined) search.set('tp', String(params.tp));
  if (params.pp !== undefined) search.set('pp', String(params.pp));
  if (params.dp !== undefined) search.set('dp', String(params.dp));
  if (params.batch_size !== undefined) search.set('batch', String(params.batch_size));
  if (params.zero_stage !== undefined) search.set('zero', String(params.zero_stage));
  const qs = search.toString();
  return fetchAPI(`/parallelism/export${qs ? `?${qs}` : ''}`);
}

export async function getParallelismRLHF(params: { model?: string; algorithm?: string; compare?: boolean }) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.algorithm) search.set('algorithm', params.algorithm);
  if (params.compare !== undefined) search.set('compare', String(params.compare));
  const qs = search.toString();
  return fetchAPI(`/parallelism/rlhf${qs ? `?${qs}` : ''}`);
}

export async function getParallelismMoe(params: { model?: string }) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  const qs = search.toString();
  return fetchAPI(`/parallelism/moe${qs ? `?${qs}` : ''}`);
}

export async function getParallelismLongContext(params: { model?: string; seq_length?: number }) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.seq_length !== undefined) search.set('seq_length', String(params.seq_length));
  const qs = search.toString();
  return fetchAPI(`/parallelism/long-context${qs ? `?${qs}` : ''}`);
}

export async function getParallelismCommOverlap(params: { model?: string }) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  const qs = search.toString();
  return fetchAPI(`/parallelism/comm-overlap${qs ? `?${qs}` : ''}`);
}

export async function getParallelismInferenceOpt(params: { model?: string; goal?: string }) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.goal) search.set('goal', params.goal);
  const qs = search.toString();
  return fetchAPI(`/parallelism/inference-opt${qs ? `?${qs}` : ''}`);
}

export async function getParallelismNccl(params: { nodes?: number; gpus?: number; diagnose?: boolean }) {
  const search = new URLSearchParams();
  if (params.nodes !== undefined) search.set('nodes', String(params.nodes));
  if (params.gpus !== undefined) search.set('gpus', String(params.gpus));
  if (params.diagnose !== undefined) search.set('diagnose', String(params.diagnose));
  const qs = search.toString();
  return fetchAPI(`/parallelism/nccl${qs ? `?${qs}` : ''}`);
}

export async function getParallelismLargeScale(params: {
  model?: string;
  nodes?: number;
  gpus_per_node?: number;
  network?: string;
  batch_size?: number;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.nodes !== undefined) search.set('nodes', String(params.nodes));
  if (params.gpus_per_node !== undefined) search.set('gpus', String(params.gpus_per_node));
  if (params.network) search.set('network', params.network);
  if (params.batch_size !== undefined) search.set('batch', String(params.batch_size));
  const qs = search.toString();
  return fetchAPI(`/parallelism/large-scale${qs ? `?${qs}` : ''}`);
}

// ============================================================================
// NCU DEEP DIVE ENDPOINTS
// ============================================================================

export async function getNcuDeepDive() {
  return fetchAPI('/ncu/deepdive');
}

// ============================================================================
// DISTRIBUTED TRAINING ENDPOINTS
// ============================================================================

export async function getDistributedNccl(params: {
  nodes?: number;
  gpus?: number;
  model_size?: number;
  tp?: number;
  pp?: number;
  diagnose?: boolean;
}) {
  const search = new URLSearchParams();
  if (params.nodes !== undefined) search.set('nodes', String(params.nodes));
  if (params.gpus !== undefined) search.set('gpus', String(params.gpus));
  if (params.model_size !== undefined) search.set('model_size', String(params.model_size));
  if (params.tp !== undefined) search.set('tp', String(params.tp));
  if (params.pp !== undefined) search.set('pp', String(params.pp));
  if (params.diagnose !== undefined) search.set('diagnose', String(params.diagnose));
  const qs = search.toString();
  return fetchAPI(`/distributed/nccl${qs ? `?${qs}` : ''}`);
}

export async function getDistributedCommOverlap(params: {
  model?: string;
  tp?: number;
  pp?: number;
  dp?: number;
  batch_size?: number;
  seq_length?: number;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.tp !== undefined) search.set('tp', String(params.tp));
  if (params.pp !== undefined) search.set('pp', String(params.pp));
  if (params.dp !== undefined) search.set('dp', String(params.dp));
  if (params.batch_size !== undefined) search.set('batch', String(params.batch_size));
  if (params.seq_length !== undefined) search.set('seq', String(params.seq_length));
  const qs = search.toString();
  return fetchAPI(`/distributed/comm-overlap${qs ? `?${qs}` : ''}`);
}

export async function getDistributedMoe(params: {
  model?: string;
  num_experts?: number;
  gpus?: number;
  memory?: number;
  batch_size?: number;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.num_experts !== undefined) search.set('experts', String(params.num_experts));
  if (params.gpus !== undefined) search.set('gpus', String(params.gpus));
  if (params.memory !== undefined) search.set('memory', String(params.memory));
  if (params.batch_size !== undefined) search.set('batch', String(params.batch_size));
  const qs = search.toString();
  return fetchAPI(`/distributed/moe${qs ? `?${qs}` : ''}`);
}

export async function getDistributedLongContext(params: {
  model?: string;
  seq_length?: number;
  gpus?: number;
  memory?: number;
  method?: string;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.seq_length !== undefined) search.set('seq', String(params.seq_length));
  if (params.gpus !== undefined) search.set('gpus', String(params.gpus));
  if (params.memory !== undefined) search.set('memory', String(params.memory));
  if (params.method) search.set('method', params.method);
  const qs = search.toString();
  return fetchAPI(`/distributed/long-context${qs ? `?${qs}` : ''}`);
}

export async function getDistributedRLHF(params: {
  model?: string;
  algorithm?: string;
  batch_size?: number;
  seq_length?: number;
  memory?: number;
  compare?: boolean;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.algorithm) search.set('algorithm', params.algorithm);
  if (params.batch_size !== undefined) search.set('batch', String(params.batch_size));
  if (params.seq_length !== undefined) search.set('seq', String(params.seq_length));
  if (params.memory !== undefined) search.set('memory', String(params.memory));
  if (params.compare !== undefined) search.set('compare', String(params.compare));
  const qs = search.toString();
  return fetchAPI(`/distributed/rlhf${qs ? `?${qs}` : ''}`);
}

export async function getDistributedVllm(params: {
  model?: string;
  gpus?: number;
  memory?: number;
  target?: string;
  max_seq_length?: number;
  quantization?: string;
  compare_engines?: boolean;
}) {
  const search = new URLSearchParams();
  if (params.model) search.set('model', params.model);
  if (params.gpus !== undefined) search.set('gpus', String(params.gpus));
  if (params.memory !== undefined) search.set('memory', String(params.memory));
  if (params.target) search.set('target', params.target);
  if (params.max_seq_length !== undefined) search.set('seq', String(params.max_seq_length));
  if (params.quantization) search.set('quant', params.quantization);
  if (params.compare_engines !== undefined) search.set('compare', String(params.compare_engines));
  const qs = search.toString();
  return fetchAPI(`/distributed/vllm${qs ? `?${qs}` : ''}`);
}

// ============================================================================
// INTELLIGENCE ENDPOINTS
// ============================================================================

export async function getIntelligenceTechniques() {
  return fetchAPI('/intelligence/techniques');
}

export async function getIntelligenceRecommendation(params?: Record<string, string | number | boolean>) {
  const search = new URLSearchParams();
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null) search.set(k, String(v));
  });
  const qs = search.toString();
  return fetchAPI(`/intelligence/recommend${qs ? `?${qs}` : ''}`);
}

export async function getIntelligenceDistributed(params?: Record<string, string | number | boolean>) {
  const search = new URLSearchParams();
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null) search.set(k, String(v));
  });
  const qs = search.toString();
  return fetchAPI(`/intelligence/distributed${qs ? `?${qs}` : ''}`);
}

export async function getIntelligenceVllm(params?: Record<string, string | number | boolean>) {
  const search = new URLSearchParams();
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null) search.set(k, String(v));
  });
  const qs = search.toString();
  return fetchAPI(`/intelligence/vllm${qs ? `?${qs}` : ''}`);
}

export async function getIntelligenceRL(params?: Record<string, string | number | boolean>) {
  const search = new URLSearchParams();
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null) search.set(k, String(v));
  });
  const qs = search.toString();
  return fetchAPI(`/intelligence/rl${qs ? `?${qs}` : ''}`);
}

// ============================================================================
// CLUSTER MANAGEMENT ENDPOINTS
// ============================================================================

export async function getClusterFaultTolerance(params: {
  params?: number;
  nodes?: number;
  gpus?: number;
  hours?: number;
  spot?: boolean;
  cloud?: string;
}) {
  const search = new URLSearchParams();
  if (params.params !== undefined) search.set('params', String(params.params));
  if (params.nodes !== undefined) search.set('nodes', String(params.nodes));
  if (params.gpus !== undefined) search.set('gpus', String(params.gpus));
  if (params.hours !== undefined) search.set('hours', String(params.hours));
  if (params.spot !== undefined) search.set('spot', String(params.spot));
  if (params.cloud) search.set('cloud', params.cloud);
  const qs = search.toString();
  return fetchAPI(`/cluster/fault-tolerance${qs ? `?${qs}` : ''}`);
}

export async function getClusterSpotConfig(params: { params?: number; cloud?: string; budget?: boolean }) {
  const search = new URLSearchParams();
  if (params.params !== undefined) search.set('params', String(params.params));
  if (params.cloud) search.set('cloud', params.cloud);
  if (params.budget !== undefined) search.set('budget', String(params.budget));
  const qs = search.toString();
  return fetchAPI(`/cluster/spot-config${qs ? `?${qs}` : ''}`);
}

export async function getClusterElasticScaling(params: { params?: number; nodes?: number; traffic?: string }) {
  const search = new URLSearchParams();
  if (params.params !== undefined) search.set('params', String(params.params));
  if (params.nodes !== undefined) search.set('nodes', String(params.nodes));
  if (params.traffic) search.set('traffic', params.traffic);
  const qs = search.toString();
  return fetchAPI(`/cluster/elastic-scaling${qs ? `?${qs}` : ''}`);
}

export async function diagnoseClusterError(error: string) {
  const search = new URLSearchParams();
  if (error) search.set('error', error);
  const qs = search.toString();
  return fetchAPI(`/cluster/diagnose${qs ? `?${qs}` : ''}`);
}

// ============================================================================
// REPORT ENDPOINTS
// ============================================================================

export async function generateReport(format: 'html' | 'pdf' | 'md' = 'html') {
  return fetchAPI(`/report/generate?format=${format}`);
}

// ============================================================================
// SPEED TEST ENDPOINTS
// ============================================================================

export async function runSpeedTest() {
  return fetchAPI('/speedtest');
}

export async function runGpuBandwidthTest() {
  return fetchAPI('/gpu-bandwidth');
}

export async function runNetworkTest() {
  return fetchAPI('/network-test');
}

// ============================================================================
// PROFILES ENDPOINTS
// ============================================================================

export async function getProfiles() {
  return fetchAPI('/profiles');
}

export async function getHistorySummary() {
  return fetchAPI('/history');
}

// ============================================================================
// THEMES ENDPOINTS
// ============================================================================

export async function getThemes() {
  return fetchAPI('/themes');
}

// ============================================================================
// RLHF ENDPOINTS (NEW!)
// ============================================================================

export async function getRLHFMethods() {
  return fetchAPI('/rlhf/methods');
}

export async function getRLHFConfig(params: {
  method?: string;
  model_size?: number;
  gpus?: number;
  memory_gb?: number;
}) {
  const search = new URLSearchParams();
  if (params.method) search.set('method', params.method);
  if (params.model_size) search.set('model_size', params.model_size.toString());
  if (params.gpus) search.set('gpus', params.gpus.toString());
  if (params.memory_gb) search.set('memory_gb', params.memory_gb.toString());
  return fetchAPI(`/rlhf/config?${search.toString()}`);
}

export async function getRLHFMemoryEstimate(params: {
  model_size?: number;
  method?: string;
  precision?: string;
  use_lora?: boolean;
  batch_size?: number;
  seq_length?: number;
}) {
  const search = new URLSearchParams();
  if (params.model_size) search.set('model_size', params.model_size.toString());
  if (params.method) search.set('method', params.method);
  if (params.precision) search.set('precision', params.precision);
  if (params.use_lora !== undefined) search.set('use_lora', params.use_lora.toString());
  if (params.batch_size) search.set('batch_size', params.batch_size.toString());
  if (params.seq_length) search.set('seq_length', params.seq_length.toString());
  return fetchAPI(`/rlhf/memory?${search.toString()}`);
}

// ============================================================================
// INFERENCE ENGINE ENDPOINTS (NEW!)
// ============================================================================

export async function getInferenceEngines() {
  return fetchAPI('/inference/engines');
}

export async function getInferenceOptimizationTechniques() {
  return fetchAPI('/inference/techniques');
}

export async function getInferenceModelsFit() {
  return fetchAPI('/inference/models-fit');
}

// ============================================================================
// MCP SERVER INTEGRATION
// ============================================================================

export async function getMcpTools() {
  return fetchAPI('/mcp/tools');
}

export async function getMcpStatus() {
  return fetchAPI('/mcp/status');
}

export async function callMcpTool(tool: string, params: Record<string, unknown> = {}) {
  return fetchAPI('/mcp/call', {
    method: 'POST',
    body: JSON.stringify({ tool, params }),
  });
}

// ============================================================================
// PERFORMANCE INSIGHTS
// ============================================================================

export async function getPerformanceInsights() {
  return fetchAPI('/insights');
}

export async function refreshPerformanceInsights() {
  return fetchAPI('/insights/refresh');
}

// ============================================================================
// HUGGINGFACE ENDPOINTS
// ============================================================================

export async function getHfTrending() {
  return fetchAPI('/hf/trending');
}

export async function searchHfModels(query: string) {
  const search = new URLSearchParams();
  if (query) search.set('q', query);
  const qs = search.toString();
  return fetchAPI(`/hf/search${qs ? `?${qs}` : ''}`);
}

export async function getHfModel(modelId: string) {
  return fetchAPI(`/hf/model/${encodeURIComponent(modelId)}`);
}

// ============================================================================
// WEBHOOK ENDPOINTS
// ============================================================================

export async function getWebhooks() {
  return fetchAPI('/webhooks');
}

export async function saveWebhooks(webhooks: unknown[]) {
  return fetchAPI('/webhooks/save', {
    method: 'POST',
    body: JSON.stringify({ webhooks }),
  });
}

export async function testWebhook(config: { name: string; url: string; events: string[]; platform?: string }) {
  return fetchAPI('/webhook/test', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function sendWebhookNotification(payload: unknown) {
  return fetchAPI('/webhook/send', {
    method: 'POST',
    body: JSON.stringify(payload || {}),
  });
}

// ============================================================================
// UNIFIED API
// ============================================================================

export async function callUnifiedApi(path: string, payload?: unknown) {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return fetchAPI(`/unified${normalized}`, {
    method: 'POST',
    body: JSON.stringify(payload || {}),
  });
}

// ============================================================================
// EXPORT ENDPOINTS
// ============================================================================

export async function exportCSV() {
  const res = await fetch(`${API_BASE}/export/csv`);
  return res.blob();
}

export async function exportCSVDetailed() {
  const res = await fetch(`${API_BASE}/export/csv/detailed`);
  return res.blob();
}

export async function exportPDF() {
  const res = await fetch(`${API_BASE}/export/pdf`);
  return res.blob();
}

export async function exportHTML() {
  const res = await fetch(`${API_BASE}/export/html`);
  return res.blob();
}

// Generic export (csv|markdown|json payload in JSON wrapper)
export async function exportGeneric(format: 'csv' | 'markdown' | 'json') {
  return fetchAPI<{ format: string; payload: string | object }>(
    `/export/generic?format=${encodeURIComponent(format)}`
  );
}

// Compare two benchmark runs
export async function compareRuns(params: { baseline: string; candidate: string; top?: number }) {
  const search = new URLSearchParams();
  search.set('baseline', params.baseline);
  search.set('candidate', params.candidate);
  if (params.top !== undefined) search.set('top', params.top.toString());
  return fetchAPI('/compare-runs?' + search.toString());
}

// Generate launch plan
export async function generateLaunchPlan(params: {
  model_params: number;
  nodes: number;
  gpus: number;
  tp: number;
  pp: number;
  dp: number;
  batch_size: number;
  script?: string;
  extra_args?: string;
}) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null) search.set(k, String(v));
  });
  return fetchAPI('/launch-plan?' + search.toString());
}

// Roofline stride sweep
export async function getRooflineSweep(sizeMb: number, strides?: number[]) {
  const search = new URLSearchParams();
  search.set('size_mb', sizeMb.toString());
  (strides || []).forEach((s) => search.append('stride', s.toString()));
  return fetchAPI('/roofline?' + search.toString());
}
