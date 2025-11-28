'use client';

import { useState } from 'react';
import { Download, FileText, FileSpreadsheet, FileDown, Loader2 } from 'lucide-react';
import { exportCSV, exportCSVDetailed, exportPDF, exportHTML, generateReport } from '@/lib/api';
import { useToast } from './Toast';
import { cn } from '@/lib/utils';

export function ExportMenu() {
  const [isOpen, setIsOpen] = useState(false);
  const [exporting, setExporting] = useState<string | null>(null);
  const { showToast } = useToast();

  const handleExport = async (type: string) => {
    setExporting(type);
    try {
      let blob: Blob;
      let filename: string;

      switch (type) {
        case 'csv':
          blob = await exportCSV();
          filename = `benchmarks_${new Date().toISOString().split('T')[0]}.csv`;
          break;
        case 'csv-detailed':
          blob = await exportCSVDetailed();
          filename = `benchmarks_detailed_${new Date().toISOString().split('T')[0]}.csv`;
          break;
        case 'pdf':
          blob = await exportPDF();
          filename = `performance_report_${new Date().toISOString().split('T')[0]}.pdf`;
          break;
        case 'html':
          blob = await exportHTML();
          filename = `performance_report_${new Date().toISOString().split('T')[0]}.html`;
          break;
        case 'md':
          const report = await generateReport('md');
          blob = new Blob([JSON.stringify(report, null, 2)], { type: 'text/markdown' });
          filename = `performance_report_${new Date().toISOString().split('T')[0]}.md`;
          break;
        default:
          throw new Error('Unknown export type');
      }

      // Download the file
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      showToast(`✅ Exported ${filename}`, 'success');
      setIsOpen(false);
    } catch (e) {
      showToast(`❌ Export failed: ${e instanceof Error ? e.message : 'Unknown error'}`, 'error');
    } finally {
      setExporting(null);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors"
      >
        <Download className="w-4 h-4" />
        <span>Export</span>
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />

          {/* Dropdown */}
          <div className="absolute right-0 mt-2 w-56 bg-brand-card border border-white/10 rounded-lg shadow-xl z-50 py-2">
            <div className="px-3 py-2 text-xs text-white/40 uppercase">Export Options</div>
            
            <button
              onClick={() => handleExport('csv')}
              disabled={!!exporting}
              className="w-full flex items-center gap-3 px-4 py-2 hover:bg-white/5 text-left"
            >
              {exporting === 'csv' ? (
                <Loader2 className="w-4 h-4 animate-spin text-accent-primary" />
              ) : (
                <FileSpreadsheet className="w-4 h-4 text-accent-success" />
              )}
              <span className="text-white">CSV (Basic)</span>
            </button>

            <button
              onClick={() => handleExport('csv-detailed')}
              disabled={!!exporting}
              className="w-full flex items-center gap-3 px-4 py-2 hover:bg-white/5 text-left"
            >
              {exporting === 'csv-detailed' ? (
                <Loader2 className="w-4 h-4 animate-spin text-accent-primary" />
              ) : (
                <FileSpreadsheet className="w-4 h-4 text-accent-info" />
              )}
              <span className="text-white">CSV (Detailed)</span>
            </button>

            <div className="my-2 border-t border-white/5" />

            <button
              onClick={() => handleExport('pdf')}
              disabled={!!exporting}
              className="w-full flex items-center gap-3 px-4 py-2 hover:bg-white/5 text-left"
            >
              {exporting === 'pdf' ? (
                <Loader2 className="w-4 h-4 animate-spin text-accent-primary" />
              ) : (
                <FileDown className="w-4 h-4 text-accent-danger" />
              )}
              <span className="text-white">PDF Report</span>
            </button>

            <button
              onClick={() => handleExport('html')}
              disabled={!!exporting}
              className="w-full flex items-center gap-3 px-4 py-2 hover:bg-white/5 text-left"
            >
              {exporting === 'html' ? (
                <Loader2 className="w-4 h-4 animate-spin text-accent-primary" />
              ) : (
                <FileText className="w-4 h-4 text-accent-warning" />
              )}
              <span className="text-white">HTML Report</span>
            </button>

            <button
              onClick={() => handleExport('md')}
              disabled={!!exporting}
              className="w-full flex items-center gap-3 px-4 py-2 hover:bg-white/5 text-left"
            >
              {exporting === 'md' ? (
                <Loader2 className="w-4 h-4 animate-spin text-accent-primary" />
              ) : (
                <FileText className="w-4 h-4 text-accent-secondary" />
              )}
              <span className="text-white">Markdown Report</span>
            </button>
          </div>
        </>
      )}
    </div>
  );
}


