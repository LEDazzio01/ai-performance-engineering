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

// Quick benchmark runner
export async function runBenchmark(chapter: string, name: string, options?: { run_baseline?: boolean; run_optimized?: boolean }) {
  return fetchAPI('/benchmark/run', {
    method: 'POST',
    body: JSON.stringify({
      chapter,
      name,
      run_baseline: options?.run_baseline ?? true,
      run_optimized: options?.run_optimized ?? true,
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

// ============================================================================
// COST & EFFICIENCY ENDPOINTS
// ============================================================================

export async function getCostCalculator() {
  return fetchAPI('/cost/calculator');
}

export async function getCostROI() {
  return fetchAPI('/cost/roi');
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

// ============================================================================
// NCU DEEP DIVE ENDPOINTS
// ============================================================================

export async function getNcuDeepDive() {
  return fetchAPI('/ncu/deepdive');
}

// ============================================================================
// INTELLIGENCE ENDPOINTS
// ============================================================================

export async function getIntelligenceTechniques() {
  return fetchAPI('/intelligence/techniques');
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
// HUGGINGFACE ENDPOINTS
// ============================================================================

export async function getHfTrending() {
  return fetchAPI('/hf/trending');
}

// ============================================================================
// WEBHOOK ENDPOINTS
// ============================================================================

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
