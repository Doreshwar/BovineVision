import React, { useState, useEffect } from 'react';
import {
  BookOpen, Compass, Milk, Shield, Award,
  Send, Sparkles, AlertCircle, CheckCircle2, Lock
} from 'lucide-react';
import type { BreedDossier, RAGAnswer } from '../types';


interface BreedInfoRAGProps {
  topBreed: string | null;
  isAccepted: boolean;
}

export const BreedInfoRAG: React.FC<BreedInfoRAGProps> = ({ topBreed, isAccepted }) => {
  const [dossier, setDossier] = useState<BreedDossier | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'morphology' | 'production' | 'husbandry' | 'citations' | 'qa'>('overview');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // QA state
  const [question, setQuestion] = useState('');
  const [qaLoading, setQaLoading] = useState(false);
  const [qaAnswer, setQaAnswer] = useState<RAGAnswer | null>(null);

  useEffect(() => {
    // STRICT RULE: If not accepted, do not fetch RAG
    if (!isAccepted || !topBreed) {
      setDossier(null);
      setQaAnswer(null);
      setError(null);
      return;
    }

    const fetchDossier = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const resp = await fetch(`/api/rag/breed/${encodeURIComponent(topBreed)}?is_rejected=false`);
        if (!resp.ok) {
          const errData = await resp.json().catch(() => ({}));
          throw new Error(errData.detail || `Failed to fetch breed info (${resp.status})`);
        }
        const data = await resp.json();
        setDossier(data);
      } catch (err: any) {
        setError(err.message || 'Error loading RAG breed dossier');
      } finally {
        setIsLoading(false);
      }
    };

    fetchDossier();
  }, [topBreed, isAccepted]);

  const handleAskQuestion = async (qText?: string) => {
    const query = qText || question;
    if (!query.trim() || !topBreed || !isAccepted) return;

    setQaLoading(true);
    try {
      const resp = await fetch('/api/rag/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ breed_name: topBreed, question: query })
      });
      const data = await resp.json();
      setQaAnswer(data);
      if (qText) setQuestion(qText);
    } catch (err: any) {
      console.error('RAG QA error:', err);
    } finally {
      setQaLoading(false);
    }
  };

  // If rejected or no detection
  if (!isAccepted || !topBreed) {
    return (
      <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-8 text-center flex flex-col items-center justify-center gap-3 min-h-[350px]">
        <div className="w-12 h-12 rounded-xl bg-slate-800/50 flex items-center justify-center text-slate-500">
          <Lock className="w-6 h-6" />
        </div>
        <div>
          <h3 className="text-base font-bold text-slate-300">RAG Intelligence Pipeline Dormant</h3>
          <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto leading-relaxed">
            Retrieval-Augmented Generation (RAG) is strictly decoupled from classification.
            When a prediction is rejected or unverified, RAG requests are locked to prevent synthesizing unverified livestock information.
          </p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4 animate-pulse">
        <div className="h-6 bg-slate-800 rounded w-1/4"></div>
        <div className="h-10 bg-slate-800/60 rounded-lg"></div>
        <div className="h-48 bg-slate-800/40 rounded-xl"></div>
      </div>
    );
  }

  if (error || !dossier) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="bg-rose-950/40 border border-rose-800 rounded-xl p-4 text-rose-300 flex items-center gap-3 text-xs">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error || 'Unable to retrieve knowledge profile for this breed.'}</span>
        </div>
      </div>
    );
  }

  const { scientific_classification, origin_and_geography, morphological_traits, production_and_utility, management_and_resilience, citations } = dossier;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30">
              RAG Knowledge Dossier
            </span>
            <span className="text-xs text-slate-400">{scientific_classification.nbagr_accession}</span>
          </div>
          <h2 className="text-2xl font-black text-white mt-1 flex items-center gap-2">
            <span>{dossier.breed_name} Cattle</span>
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
              {scientific_classification.category}
            </span>
          </h2>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-400 bg-slate-950/60 px-3 py-1.5 rounded-lg border border-slate-800 self-start sm:self-auto">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          <span>ICAR-NBAGR Verified Evidence</span>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex flex-wrap gap-1.5 border-b border-slate-800 pb-3">
        {[
          { id: 'overview', label: 'Overview & Origin', icon: Compass },
          { id: 'morphology', label: 'Morphology & Horns', icon: Award },
          { id: 'production', label: 'Yield & Draught', icon: Milk },
          { id: 'husbandry', label: 'Resilience & Care', icon: Shield },
          { id: 'citations', label: 'Official Citations', icon: BookOpen },
          { id: 'qa', label: 'Ask AI (RAG QA)', icon: Sparkles },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                isActive
                  ? 'bg-amber-500 text-slate-950 shadow-md font-bold'
                  : 'bg-slate-800/60 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-700/60'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Tab Contents */}
      <div className="text-sm">
        {/* OVERVIEW & ORIGIN */}
        {activeTab === 'overview' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-1.5">
                <span className="text-[11px] font-bold text-amber-400 uppercase tracking-wider">
                  Native Breeding Tract
                </span>
                <p className="text-slate-200 font-medium leading-relaxed">
                  {origin_and_geography.native_tract}
                </p>
              </div>

              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-1.5">
                <span className="text-[11px] font-bold text-amber-400 uppercase tracking-wider">
                  Agro-Climatic Zone
                </span>
                <p className="text-slate-200 font-medium leading-relaxed">
                  {origin_and_geography.agro_climatic_zone}
                </p>
              </div>
            </div>

            <div className="bg-slate-950/40 border border-slate-800 rounded-xl p-4 space-y-2">
              <span className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                <BookOpen className="w-3.5 h-3.5 text-amber-400" />
                Historical Heritage & Traditional Husbandry
              </span>
              <p className="text-xs text-slate-300 leading-relaxed">
                {origin_and_geography.historical_background}
              </p>
            </div>

            <div className="p-3 rounded-lg bg-amber-500/5 border border-amber-500/20 text-xs text-amber-300/90 flex items-center justify-between">
              <span>Conservation Status: <strong>{dossier.conservation_status}</strong></span>
              <span className="text-[11px] text-slate-400">Species: {scientific_classification.species}</span>
            </div>
          </div>
        )}

        {/* MORPHOLOGY */}
        {activeTab === 'morphology' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            {[
              { title: 'Horn Structure & Shape', text: morphological_traits.horns, highlight: true },
              { title: 'Forehead & Facial Profile', text: morphological_traits.forehead },
              { title: 'Coat Coloration & Pattern', text: morphological_traits.coat_color },
              { title: 'Ear Shape & Orientation', text: morphological_traits.ears },
              { title: 'Hump & Dewlap Development', text: morphological_traits.hump_and_dewlap },
              { title: 'Skeletal & Body Conformation', text: morphological_traits.body_conformation },
            ].map((item, idx) => (
              <div
                key={idx}
                className={`p-4 rounded-xl border space-y-1.5 ${
                  item.highlight
                    ? 'bg-amber-500/5 border-amber-500/30'
                    : 'bg-slate-950/60 border-slate-800'
                }`}
              >
                <span className="text-[11px] font-bold text-amber-400 uppercase tracking-wider">
                  {item.title}
                </span>
                <p className="text-xs text-slate-200 leading-relaxed">{item.text}</p>
              </div>
            ))}
          </div>
        )}

        {/* PRODUCTION & UTILITY */}
        {activeTab === 'production' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 text-center">
                <span className="text-[11px] text-slate-400 block mb-1">Average Lactation Yield</span>
                <span className="text-xl font-black text-amber-400">
                  {production_and_utility.average_lactation_yield_kg}
                </span>
                <span className="text-[10px] text-slate-500 block mt-1">per lactation cycle</span>
              </div>

              <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 text-center">
                <span className="text-[11px] text-slate-400 block mb-1">Milk Fat Content</span>
                <span className="text-xl font-black text-emerald-400">
                  {production_and_utility.fat_percentage}
                </span>
                <span className="text-[10px] text-slate-500 block mt-1">rich in A2 beta-casein</span>
              </div>

              <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 text-center">
                <span className="text-[11px] text-slate-400 block mb-1">Lactation Length</span>
                <span className="text-xl font-black text-sky-400">
                  {production_and_utility.lactation_length_days}
                </span>
                <span className="text-[10px] text-slate-500 block mt-1">typical milking days</span>
              </div>
            </div>

            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-1.5">
              <span className="text-xs font-bold text-amber-400">Draught & Agricultural Capacity</span>
              <p className="text-xs text-slate-300 leading-relaxed">
                {production_and_utility.draught_capacity}
              </p>
            </div>
          </div>
        )}

        {/* RESILIENCE & CARE */}
        {activeTab === 'husbandry' && (
          <div className="space-y-3.5">
            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-1.5">
              <span className="text-xs font-bold text-amber-400">Dietary & Forage Adaptation</span>
              <p className="text-xs text-slate-300 leading-relaxed">
                {management_and_resilience.diet_and_forage}
              </p>
            </div>

            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-1.5">
              <span className="text-xs font-bold text-emerald-400">Thermotolerance & Disease Immunity</span>
              <p className="text-xs text-slate-300 leading-relaxed">
                {management_and_resilience.disease_resistance}
              </p>
            </div>

            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-1.5">
              <span className="text-xs font-bold text-slate-300">Temperament & Behavior</span>
              <p className="text-xs text-slate-400 leading-relaxed">
                {management_and_resilience.temperament}
              </p>
            </div>
          </div>
        )}

        {/* CITATIONS */}
        {activeTab === 'citations' && (
          <div className="space-y-3">
            <p className="text-xs text-slate-400 mb-2">
              All information in this RAG dossier is retrieved strictly from authoritative government and academic repositories:
            </p>
            {citations.map((c, idx) => (
              <div key={idx} className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-amber-400">{c.source}</span>
                  <span className="text-[10px] text-slate-500 uppercase">{c.authority}</span>
                </div>
                <p className="text-xs text-slate-300 font-medium">{c.document}</p>
              </div>
            ))}
          </div>
        )}

        {/* INTERACTIVE RAG QA */}
        {activeTab === 'qa' && (
          <div className="space-y-4">
            <div className="space-y-2">
              <span className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                Ask Any Specific Question About {dossier.breed_name}
              </span>
              <p className="text-xs text-slate-400">
                Queries are answered strictly using grounded vector search over ICAR-NBAGR verified records.
              </p>

              {/* Suggested Questions */}
              <div className="flex flex-wrap gap-1.5 pt-1">
                {[
                  `What is the average milk yield of ${dossier.breed_name}?`,
                  `Describe the horn morphology of ${dossier.breed_name}.`,
                  `Where is the native breeding tract located?`,
                  `How heat and disease resistant is this breed?`
                ].map((sug, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleAskQuestion(sug)}
                    className="text-[11px] px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-all text-left"
                  >
                    💬 {sug}
                  </button>
                ))}
              </div>

              {/* Input Box */}
              <div className="flex gap-2 pt-2">
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAskQuestion()}
                  placeholder={`e.g. What is the milk fat percentage and draught stamina?`}
                  className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-amber-500"
                />
                <button
                  onClick={() => handleAskQuestion()}
                  disabled={qaLoading || !question.trim()}
                  className="px-4 py-2.5 bg-amber-500 hover:bg-amber-400 disabled:bg-slate-800 text-slate-950 font-bold rounded-xl text-xs flex items-center gap-1.5 transition-all shadow-md"
                >
                  {qaLoading ? <Sparkles className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                  <span>Ask RAG</span>
                </button>
              </div>
            </div>

            {/* Answer Display */}
            {qaAnswer && (
              <div className="bg-slate-950 border border-amber-500/30 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                  <span className="text-[11px] font-bold text-amber-400 uppercase tracking-wider flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    Grounded RAG Response
                  </span>
                  <span className="text-[10px] text-slate-500">Query: &ldquo;{qaAnswer.question}&rdquo;</span>
                </div>

                <div className="text-xs text-slate-200 leading-relaxed whitespace-pre-line">
                  {qaAnswer.answer}
                </div>

                {qaAnswer.citations && qaAnswer.citations.length > 0 && (
                  <div className="text-[11px] text-slate-400 pt-2 border-t border-slate-900">
                    <strong className="text-slate-300">Cited Sources:</strong>{' '}
                    {qaAnswer.citations.map((c: any) => c.source).join(', ')}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
