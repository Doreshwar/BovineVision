import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, Sliders, RefreshCw, AlertCircle, Sparkles, CheckCircle2 } from 'lucide-react';

interface ImageUploaderProps {
  selectedFile: File | null;
  previewUrl: string | null;
  onFileSelect: (file: File) => void;
  onClear: () => void;
  threshold: number;
  onThresholdChange: (val: number) => void;
  onAnalyze: () => void;
  isLoading: boolean;
  sampleImages: { breed: string; filename: string; sample_url: string }[];
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  selectedFile,
  previewUrl,
  onFileSelect,
  onClear,
  threshold,
  onThresholdChange,
  onAnalyze,
  isLoading,
  sampleImages,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onFileSelect(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const handleSampleClick = async (url: string, name: string) => {
    try {
      const resp = await fetch(url);
      const blob = await resp.blob();
      const file = new File([blob], `${name}.jpg`, { type: 'image/jpeg' });
      onFileSelect(file);
    } catch (err) {
      console.error('Failed to load sample image:', err);
    }
  };

  // Helper to load an intentional rejection test sample
  const handleLoadRejectionSample = () => {
    // Generate a simple colored canvas image that has no cattle features
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 300;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, 300, 300);
      ctx.fillStyle = '#64748b';
      ctx.font = '20px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Non-Cattle / Random Image', 150, 150);
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], 'random-non-cattle.jpg', { type: 'image/jpeg' });
          onFileSelect(file);
        }
      }, 'image/jpeg');
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <UploadCloud className="w-5 h-5 text-amber-400" />
            Image Upload & Processing
          </h2>
          <p className="text-xs text-slate-400">
            Upload an authentic cattle photo (JPEG / PNG) for custom CNN evaluation
          </p>
        </div>

        {selectedFile && (
          <button
            onClick={onClear}
            className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200 bg-slate-800 hover:bg-slate-700 px-2.5 py-1 rounded-md transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Clear
          </button>
        )}
      </div>

      {/* Drag & Drop Area or Preview */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={handleFileChange}
      />

      {!previewUrl ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-3 ${
            isDragOver
              ? 'border-amber-400 bg-amber-500/5'
              : 'border-slate-700 hover:border-slate-500 bg-slate-950/40 hover:bg-slate-950/70'
          }`}
        >
          <div className="w-14 h-14 rounded-2xl bg-slate-800/80 flex items-center justify-center text-slate-400 group-hover:text-amber-400">
            <ImageIcon className="w-7 h-7" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-200">
              Drag and drop cattle image here, or{' '}
              <span className="text-amber-400 underline underline-offset-2">browse</span>
            </p>
            <p className="text-xs text-slate-500 mt-1">Supports JPG, PNG, WEBP (Max 15MB)</p>
          </div>
        </div>
      ) : (
        <div className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-800 group">
          <img
            src={previewUrl}
            alt="Preview"
            className="w-full h-72 object-contain bg-slate-950/80"
          />
          <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-slate-950 via-slate-950/80 to-transparent p-4 flex items-center justify-between">
            <div className="text-xs text-slate-300 truncate max-w-[260px]">
              <span className="font-semibold text-white">{selectedFile?.name}</span>
              <span className="text-slate-500 ml-2">
                {selectedFile ? (selectedFile.size / 1024).toFixed(1) + ' KB' : ''}
              </span>
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="text-xs text-amber-400 hover:text-amber-300 font-semibold bg-slate-900/90 px-3 py-1 rounded-md border border-slate-700"
            >
              Replace
            </button>
          </div>
        </div>
      )}

      {/* Quick Test Samples */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          Quick Test Authentic Samples (from Test Split)
        </label>
        <div className="flex flex-wrap gap-2">
          {sampleImages.map((sample) => (
            <button
              key={sample.breed}
              onClick={() => handleSampleClick(sample.sample_url, sample.breed)}
              className="text-xs font-medium px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition-all"
            >
              🐂 {sample.breed}
            </button>
          ))}
          <button
            onClick={handleLoadRejectionSample}
            className="text-xs font-medium px-3 py-1.5 rounded-lg bg-rose-950/40 hover:bg-rose-900/60 text-rose-300 border border-rose-800/50 transition-all flex items-center gap-1"
            title="Load random non-cattle image to test threshold rejection"
          >
            <AlertCircle className="w-3 h-3 text-rose-400" />
            Test Rejection (&lt;50%)
          </button>
        </div>
      </div>

      {/* Threshold Slider */}
      <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-amber-400" />
            Confidence Rejection Threshold
          </label>
          <span className="text-xs font-bold px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30">
            {Math.round(threshold * 100)}%
          </span>
        </div>
        <input
          type="range"
          min="0.10"
          max="0.90"
          step="0.05"
          value={threshold}
          onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
          className="w-full accent-amber-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
        />
        <p className="text-[11px] text-slate-400 leading-relaxed">
          If none of the top 3 predictions cross this threshold (e.g. 50%), detection will be rejected
          with message <span className="text-amber-300 font-semibold">&ldquo;Could not detect, try another picture&rdquo;</span>,
          and RAG responses will be halted.
        </p>
      </div>

      {/* CTA Button */}
      <button
        onClick={onAnalyze}
        disabled={!selectedFile || isLoading}
        className={`w-full py-3.5 px-6 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all shadow-lg ${
          !selectedFile || isLoading
            ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50'
            : 'bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 text-slate-950 shadow-amber-500/20 active:scale-[0.99]'
        }`}
      >
        {isLoading ? (
          <>
            <RefreshCw className="w-4 h-4 animate-spin" />
            <span>Analyzing Features with PyTorch CNN...</span>
          </>
        ) : (
          <>
            <CheckCircle2 className="w-4 h-4" />
            <span>Run Custom CNN Prediction</span>
          </>
        )}
      </button>
    </div>
  );
};
