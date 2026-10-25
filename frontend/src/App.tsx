import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { ImageUploader } from './components/ImageUploader';
import { PredictionResults } from './components/PredictionResults';
import { BreedInfoRAG } from './components/BreedInfoRAG';
import { ModelModal } from './components/ModelModal';
import type { ClassificationResult, ModelInfo } from './types';
import { AlertCircle, Cpu } from 'lucide-react';


export function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [threshold, setThreshold] = useState<number>(0.50);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Model & Samples Info
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [sampleImages, setSampleImages] = useState<{ breed: string; filename: string; sample_url: string }[]>([]);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);

  // Fetch model info and sample images on mount
  useEffect(() => {
    const fetchMeta = async () => {
      try {
        const [infoRes, samplesRes] = await Promise.all([
          fetch('/api/model/info'),
          fetch('/api/samples')
        ]);
        if (infoRes.ok) {
          const info = await infoRes.json();
          setModelInfo(info);
        }
        if (samplesRes.ok) {
          const s = await samplesRes.json();
          setSampleImages(s.samples || []);
        }
      } catch (err) {
        console.error('Failed to load initial metadata:', err);
      }
    };
    fetchMeta();
  }, []);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const resp = await fetch(`/api/classify?threshold=${threshold}`, {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({}));
        throw new Error(errJson.detail || `Classification request failed (${resp.status})`);
      }

      const data: ClassificationResult = await resp.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred during model inference.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Navbar */}
      <Navbar
        onOpenModelInfo={() => setIsModalOpen(true)}
        device={modelInfo?.device || 'CUDA'}
      />

      {/* Hero Subtitle Banner */}
      <div className="bg-gradient-to-b from-slate-900 via-slate-950 to-slate-950 border-b border-slate-900 py-6 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight flex items-center gap-2">
              BovineVision AI
              <span className="text-amber-400 font-normal text-lg sm:text-xl">
                &mdash; Cattle Breed Classification &amp; RAG Intelligence
              </span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 max-w-2xl leading-relaxed">
              Custom PyTorch Deep CNN trained completely from scratch without pretrained weights on Mendeley Dataset xc3j9v382w.
              Returns top 3 predictions with confidence scores, minimum threshold rejection, and decoupled ICAR-NBAGR RAG knowledge.
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs bg-slate-900/90 border border-slate-800 rounded-xl px-4 py-2 self-start md:self-auto shadow-sm">
            <Cpu className="w-4 h-4 text-amber-400" />
            <div>
              <span className="text-slate-400 block text-[10px]">Active Hardware</span>
              <span className="font-semibold text-slate-200">
                {modelInfo?.metrics?.device_name || 'NVIDIA GeForce GTX 1650 Ti (CUDA)'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Workspace */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full space-y-8">
        {error && (
          <div className="bg-rose-950/60 border border-rose-800 rounded-xl p-4 text-rose-200 flex items-center gap-3 text-sm shadow-lg">
            <AlertCircle className="w-5 h-5 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        {/* Two-Column Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Upload, Preview, Threshold & CTA */}
          <div className="lg:col-span-5 space-y-6">
            <ImageUploader
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              onFileSelect={handleFileSelect}
              onClear={handleClear}
              threshold={threshold}
              onThresholdChange={setThreshold}
              onAnalyze={handleAnalyze}
              isLoading={isLoading}
              sampleImages={sampleImages}
            />
          </div>

          {/* Right Column: Predictions & Decoupled RAG */}
          <div className="lg:col-span-7 space-y-6">
            {/* Prediction Results & Rejection Notice */}
            <PredictionResults
              result={result}
              isLoading={isLoading}
            />

            {/* RAG Breed Intelligence Dossier (Strictly separate from classification) */}
            <BreedInfoRAG
              topBreed={result?.top_prediction?.breed || null}
              isAccepted={result ? result.is_accepted : false}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 px-4 sm:px-6 lg:px-8 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            <strong>BovineVision AI</strong> &bull; Custom PyTorch CNN (from scratch) &bull; Mendeley Data DOI: 10.17632/xc3j9v382w.1
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>FastAPI Backend</span>
            <span>&bull;</span>
            <span>React + TypeScript</span>
            <span>&bull;</span>
            <span>PyTorch CUDA</span>
          </div>
        </div>
      </footer>

      {/* Architecture & Metrics Modal */}
      <ModelModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        modelInfo={modelInfo}
      />
    </div>
  );
}

export default App;
