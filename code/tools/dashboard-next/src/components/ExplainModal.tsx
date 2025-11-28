'use client';

import { useState, useEffect } from 'react';
import { X, Book, Sparkles, Cpu, Code, Loader2, AlertTriangle, ChevronDown } from 'lucide-react';
import { getBookExplanation, getLLMExplanation } from '@/lib/api';
import { cn } from '@/lib/utils';

interface ExplainModalProps {
  isOpen: boolean;
  onClose: () => void;
  technique: string;
  speedup: number;
  benchmarkName: string;
  chapter?: string;
}

interface BookData {
  found: boolean;
  title?: string;
  summary?: string;
  key_points?: string[];
  content_sections?: Array<{
    heading: string;
    content: string;
    chapter: string;
  }>;
  source_file?: string;
}

interface LLMResponse {
  llm_response?: {
    summary?: string;
    why_it_works?: string;
    hardware_specific?: string;
    raw_text?: string;
  };
  context_used?: string[];
  error?: string;
}

export function ExplainModal({ isOpen, onClose, technique, speedup, benchmarkName, chapter }: ExplainModalProps) {
  const [bookData, setBookData] = useState<BookData | null>(null);
  const [llmData, setLlmData] = useState<LLMResponse | null>(null);
  const [loadingBook, setLoadingBook] = useState(true);
  const [loadingLLM, setLoadingLLM] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showRelated, setShowRelated] = useState(false);

  // Extract chapter from benchmark name if not provided
  const extractedChapter = chapter || benchmarkName.match(/^(ch\d+)/i)?.[1] || 'unknown';

  // Format chapter reference
  const formatChapter = (sourceFile?: string) => {
    if (!sourceFile) return 'AI Systems Performance Engineering';
    const match = sourceFile.match(/ch(\d+)/i);
    if (match) {
      return `Chapter ${parseInt(match[1], 10)} of AI Systems Performance Engineering`;
    }
    return 'AI Systems Performance Engineering';
  };

  const formatChapterShort = (sourceFile?: string) => {
    if (!sourceFile) return 'the book';
    const match = sourceFile.match(/ch(\d+)/i);
    if (match) {
      return `Chapter ${parseInt(match[1], 10)}`;
    }
    return 'the book';
  };

  useEffect(() => {
    if (!isOpen) return;

    async function loadExplanation() {
      setLoadingBook(true);
      setLoadingLLM(true);
      setError(null);
      setBookData(null);
      setLlmData(null);

      try {
        // Step 1: Load book content first
        const book = await getBookExplanation(technique, extractedChapter);
        setBookData(book as any);
        setLoadingBook(false);

        // Step 2: Load LLM enhancement
        try {
          const llm = await getLLMExplanation(technique, extractedChapter, benchmarkName);
          setLlmData(llm as any);
        } catch (llmError) {
          console.error('LLM enhancement failed:', llmError);
          setLlmData({ error: 'LLM enhancement unavailable' });
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load explanation');
        setLoadingBook(false);
      } finally {
        setLoadingLLM(false);
      }
    }

    loadExplanation();
  }, [isOpen, technique, extractedChapter, benchmarkName]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[9999] bg-black/70 backdrop-blur-sm flex items-start justify-center pt-[10vh] overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="w-[700px] max-w-[95vw] mb-10 bg-brand-card border border-white/10 rounded-2xl shadow-2xl overflow-hidden animate-slide-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between p-6 border-b border-white/5">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-2xl">💡</span>
              <h2 className="text-xl font-bold text-white">
                {loadingBook ? 'Loading...' : bookData?.title || technique}
              </h2>
              {speedup > 0 && (
                <span className="px-3 py-1 bg-accent-success/20 text-accent-success rounded-full text-sm font-bold">
                  ⚡ {speedup.toFixed(2)}x
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              {bookData?.source_file && (
                <span className="px-2 py-1 bg-accent-secondary/20 text-accent-secondary rounded-full text-xs">
                  📖 {formatChapterShort(bookData.source_file)}
                </span>
              )}
              <span className="px-2 py-1 bg-accent-primary/20 text-accent-primary rounded-full text-xs">
                Applied in: {benchmarkName}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/5 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-white/50" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 max-h-[70vh] overflow-y-auto">
          {error ? (
            <div className="text-center py-8">
              <AlertTriangle className="w-12 h-12 text-accent-danger mx-auto mb-4" />
              <p className="text-white/70">{error}</p>
            </div>
          ) : (
            <>
              {/* Book Content Section */}
              {loadingBook ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-8 h-8 animate-spin text-accent-secondary" />
                  <span className="ml-3 text-white/50">Searching book content...</span>
                </div>
              ) : bookData?.found && bookData.content_sections && bookData.content_sections.length > 0 ? (
                <div className="p-5 bg-gradient-to-r from-accent-secondary/10 to-transparent border border-accent-secondary/20 rounded-xl">
                  <div className="flex items-center gap-2 mb-4">
                    <Book className="w-5 h-5 text-accent-secondary" />
                    <h3 className="font-medium text-accent-secondary">
                      From {formatChapter(bookData.source_file)}
                    </h3>
                  </div>
                  
                  {/* Quote from book */}
                  <blockquote className="border-l-3 border-accent-secondary pl-4 py-2 italic text-white/70 leading-relaxed mb-4">
                    &ldquo;{(bookData.summary || bookData.content_sections[0].content).substring(0, 500)}
                    {(bookData.summary || bookData.content_sections[0].content).length > 500 ? '...' : ''}&rdquo;
                  </blockquote>

                  {/* Key points */}
                  {bookData.key_points && bookData.key_points.length > 0 && (
                    <div className="mt-4">
                      <div className="text-sm text-white/50 mb-2">Key points from the book:</div>
                      <ul className="space-y-2">
                        {bookData.key_points.slice(0, 3).map((point, i) => (
                          <li key={i} className="flex items-start gap-2 text-white/80">
                            <span className="text-accent-secondary mt-0.5">→</span>
                            {point}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Related sections (collapsible) */}
                  {bookData.content_sections.length > 1 && (
                    <div className="mt-4">
                      <button
                        onClick={() => setShowRelated(!showRelated)}
                        className="flex items-center gap-2 text-sm text-white/50 hover:text-white/70"
                      >
                        <ChevronDown className={cn('w-4 h-4 transition-transform', showRelated && 'rotate-180')} />
                        📖 {bookData.content_sections.length - 1} more related sections from the book...
                      </button>
                      {showRelated && (
                        <div className="mt-3 space-y-2">
                          {bookData.content_sections.slice(1).map((section, i) => (
                            <div
                              key={i}
                              className="p-3 bg-white/5 rounded-lg border-l-2 border-accent-primary"
                            >
                              <div className="font-medium text-white">{section.heading}</div>
                              <div className="text-sm text-white/50">from {section.chapter}.md</div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div className="p-4 bg-white/5 rounded-lg text-center text-white/50">
                  📚 No specific book content found for &quot;{technique}&quot;
                </div>
              )}

              {/* LLM Enhancement Section */}
              <div className="p-5 bg-gradient-to-r from-accent-tertiary/10 to-accent-info/5 border border-accent-tertiary/20 rounded-xl">
                <div className="flex items-center gap-2 mb-4">
                  <Sparkles className="w-5 h-5 text-accent-tertiary" />
                  <h3 className="font-medium text-accent-tertiary">
                    AI-Enhanced Explanation
                  </h3>
                  <span className="text-xs text-white/40">(citing the book + your hardware)</span>
                </div>

                {loadingLLM ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="w-6 h-6 animate-spin text-accent-tertiary" />
                    <span className="ml-3 text-white/50">Generating contextual explanation...</span>
                  </div>
                ) : llmData?.error ? (
                  <div className="text-center py-4">
                    <p className="text-accent-warning">⚠️ {llmData.error}</p>
                    <p className="text-sm text-white/50 mt-1">The book content above is still available for reference.</p>
                  </div>
                ) : llmData?.llm_response ? (
                  <div>
                    {/* Context badges */}
                    {llmData.context_used && llmData.context_used.length > 0 && (
                      <div className="flex gap-2 mb-4">
                        {llmData.context_used.includes('book') && (
                          <span className="px-2 py-1 bg-accent-secondary/20 text-accent-secondary rounded text-xs flex items-center gap-1">
                            <Book className="w-3 h-3" /> book
                          </span>
                        )}
                        {llmData.context_used.includes('hardware') && (
                          <span className="px-2 py-1 bg-accent-info/20 text-accent-info rounded text-xs flex items-center gap-1">
                            <Cpu className="w-3 h-3" /> hardware
                          </span>
                        )}
                        {llmData.context_used.includes('baseline_code') && (
                          <span className="px-2 py-1 bg-accent-warning/20 text-accent-warning rounded text-xs flex items-center gap-1">
                            <Code className="w-3 h-3" /> code
                          </span>
                        )}
                      </div>
                    )}

                    {/* LLM Response content */}
                    {llmData.llm_response.raw_text ? (
                      <div className="text-white/80 leading-relaxed whitespace-pre-wrap">
                        {llmData.llm_response.raw_text}
                      </div>
                    ) : (
                      <div className="space-y-4">
                        {llmData.llm_response.summary && (
                          <p className="text-white/90 leading-relaxed">
                            {llmData.llm_response.summary}
                          </p>
                        )}

                        {llmData.llm_response.why_it_works && (
                          <div>
                            <h4 className="text-sm font-medium text-accent-success mb-2">
                              🔬 Why It Works on Your Hardware
                            </h4>
                            <p className="text-white/70 text-sm">
                              {llmData.llm_response.why_it_works}
                            </p>
                          </div>
                        )}

                        {llmData.llm_response.hardware_specific && (
                          <div>
                            <h4 className="text-sm font-medium text-accent-info mb-2">
                              🖥️ Hardware-Specific Insights
                            </h4>
                            <p className="text-white/70 text-sm">
                              {llmData.llm_response.hardware_specific}
                            </p>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-white/50 text-center py-4">
                    AI enhancement unavailable. Refer to the book content above.
                  </p>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
