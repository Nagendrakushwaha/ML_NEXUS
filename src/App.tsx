import React, { useState, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Settings, Play, Database, Info, Layers, BarChart3, Binary, RefreshCw, Plus, Trash2 } from 'lucide-react';
import { datasetRegistry } from './ml/datasets';
import { KNN, LogisticRegression, DecisionTree, NaiveBayes, SVM, RandomForest, MLPClassifier, AdaBoost, BaseModel } from './ml/models';
import { VotingClassifier } from './ml/ensemble';
import { Point, calculateMetrics, scaleData, parseCSV } from './utils/ml-utils';
import { Visualization } from './components/Visualization';
import { MathView } from './components/MathView';
import { PieChart, Pie, Cell, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

export default function App() {
  // --- State ---
  const [datasetType, setDatasetType] = useState<string>('moons');
  const [samples, setSamples] = useState(200);
  const [noise, setNoise] = useState(0.2);
  const [imbalance, setImbalance] = useState(0.5);
  const [points, setPoints] = useState<Point[]>([]);

  const [activeTab, setActiveTab] = useState<'viz' | 'metrics' | 'math'>('viz');
  
  // Model management (supports multi-model)
  const [selectedModelKeys, setSelectedModelKeys] = useState<string[]>(['logistic']);
  const [modelStates, setModelStates] = useState<Record<string, BaseModel>>({
    knn: new KNN(),
    logistic: new LogisticRegression(),
    tree: new DecisionTree(),
    nb: new NaiveBayes(),
    svm: new SVM(),
    rf: new RandomForest(),
    mlp: new MLPClassifier(),
    ada: new AdaBoost()
  });

  const availableModels = {
    knn: "K-Nearest Neighbors",
    logistic: "Logistic Regression",
    tree: "Decision Tree",
    nb: "Naive Bayes",
    svm: "SVM (Linear)",
    rf: "Random Forest",
    mlp: "Neural Network (MLP)",
    ada: "AdaBoost"
  };

  const [ensembleEnabled, setEnsembleEnabled] = useState(false);
  const [ensembleModel, setEnsembleModel] = useState<VotingClassifier | null>(null);

  // --- Effects ---
  useEffect(() => {
    generateData();
  }, [datasetType, samples, noise, imbalance]);

  const generateData = () => {
    const raw = datasetRegistry[datasetType].generate({ samples, noise, imbalance });
    setPoints(scaleData(raw));
  };

  // Train and compute results for selected models
  const results = useMemo(() => {
    const trainedModels = selectedModelKeys.map(k => {
      const model = modelStates[k];
      model.train(points);
      return { key: k, model };
    });

    let ensemble = null;
    if (ensembleEnabled && trainedModels.length >= 2) {
      const ensMode = new VotingClassifier(trainedModels.map(m => m.model));
      ensMode.train(points);
      ensemble = ensMode;
    }

    const allModels = [...trainedModels];
    if (ensemble) allModels.push({ key: 'ensemble', model: ensemble });

    return allModels.map(({ key, model }) => {
      const predicted = points.map(p => model.predict(p.x, p.y));
      const actual = points.map(p => p.label);
      const metrics = calculateMetrics(actual, predicted);
      return { key, model, metrics };
    });
  }, [points, selectedModelKeys, modelStates, ensembleEnabled]);

  const handleHyperparamChange = (modelKey: string, paramAlias: string, val: any) => {
    setModelStates(prev => {
      const newModel = Object.assign(Object.create(Object.getPrototypeOf(prev[modelKey])), prev[modelKey]);
      newModel.hyperparams[paramAlias].value = val;
      return { ...prev, [modelKey]: newModel };
    });
  };

  const handleCSVUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const parsed = scaleData(parseCSV(text));
      if (parsed.length > 0) setPoints(parsed);
    };
    reader.readAsText(file);
  };

  // --- UI Parts ---
  const Sidebar = () => (
    <div className="w-[260px] h-full bg-white flex flex-col shrink-0 border-r border-slate-200">
      <div className="p-6 bg-slate-900 text-white">
        <h1 className="text-lg font-bold tracking-tight">ML NEXUS <span className="text-indigo-400 font-light text-xs ml-1">v2.4</span></h1>
        <p className="text-[10px] text-slate-400 mt-1 uppercase tracking-widest font-semibold">Advanced Playground</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        <section>
          <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Dataset Registry</label>
          <select 
            className="w-full mt-1 border border-slate-200 rounded p-2 text-sm bg-slate-50 focus:ring-2 focus:ring-indigo-500 outline-none"
            value={datasetType}
            onChange={(e) => setDatasetType(e.target.value)}
          >
            {Object.entries(datasetRegistry).map(([k, v]) => (
              <option key={k} value={k}>{v.label}</option>
            ))}
          </select>
          
          <div className="grid grid-cols-2 gap-2 mt-4">
             <div className="text-center bg-slate-50 border border-slate-200 p-2 rounded">
               <p className="text-[10px] text-slate-400 uppercase font-bold">Samples</p>
               <input 
                 type="range" min="50" max="1000" step="50" 
                 value={samples} 
                 onChange={(e) => setSamples(parseInt(e.target.value))} 
                 className="w-full accent-indigo-600 mt-1" 
               />
               <p className="text-xs font-bold text-slate-700 mt-1">{samples}</p>
             </div>
             <div className="text-center bg-slate-50 border border-slate-200 p-2 rounded">
               <p className="text-[10px] text-slate-400 uppercase font-bold">Noise</p>
               <input 
                 type="range" min="0" max="1" step="0.05" 
                 value={noise} 
                 onChange={(e) => setNoise(parseFloat(e.target.value))} 
                 className="w-full accent-indigo-600 mt-1" 
               />
               <p className="text-xs font-bold text-slate-700 mt-1">{noise.toFixed(2)}</p>
             </div>
          </div>
          
          <div className="mt-3">
             <label className="w-full flex items-center justify-center gap-2 bg-white hover:bg-slate-50 text-[10px] font-bold uppercase tracking-wider py-2 rounded border border-slate-200 border-dashed transition-colors cursor-pointer">
                <Plus size={14} /> Upload CSV
                <input type="file" accept=".csv" className="hidden" onChange={handleCSVUpload} />
             </label>
          </div>
        </section>

        <section>
          <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Active Models</label>
          <div className="mt-2 space-y-1">
            {Object.entries(availableModels).map(([k, label]) => (
              <label key={k} className={`flex items-center text-sm p-2 rounded border transition-all cursor-pointer ${selectedModelKeys.includes(k) ? 'bg-indigo-50 border-indigo-200 text-indigo-700 font-medium' : 'bg-transparent border-transparent text-slate-500 hover:bg-slate-50'}`}>
                <input 
                  type="checkbox" 
                  checked={selectedModelKeys.includes(k)}
                  onChange={(e) => {
                    if(e.target.checked) setSelectedModelKeys([...selectedModelKeys, k]);
                    else setSelectedModelKeys(selectedModelKeys.filter(key => key !== k));
                  }}
                  className="mr-3 w-4 h-4 rounded border-slate-300 text-indigo-600" 
                />
                <span className="flex-1">{label}</span>
              </label>
            ))}
          </div>
          
          <div className="mt-4 pt-4 border-t border-slate-100">
             <label className="flex items-center justify-between cursor-pointer group">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Ensemble Logic</span>
                <div 
                  className={`w-9 h-5 rounded-full p-1 transition-colors ${ensembleEnabled ? 'bg-indigo-600' : 'bg-slate-200'}`}
                  onClick={() => setEnsembleEnabled(!ensembleEnabled)}
                >
                   <div className={`w-3 h-3 rounded-full bg-white transition-transform ${ensembleEnabled ? 'translate-x-4' : 'translate-x-0'}`} />
                </div>
             </label>
             {ensembleEnabled && selectedModelKeys.length < 2 && (
               <p className="text-[9px] text-rose-500 mt-2 font-medium">Select 2+ models to activate.</p>
             )}
          </div>
        </section>

        <section>
          <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Hyperparameters</label>
          <div className="mt-3 space-y-5">
            {selectedModelKeys.map(k => {
              const m = modelStates[k];
              if (Object.keys(m.hyperparams).length === 0) return null;
              return (
                <div key={k} className="space-y-3">
                  <div className="text-[9px] font-black text-indigo-500 uppercase tracking-widest flex items-center gap-1">
                    <div className="w-1 h-3 bg-indigo-500 rounded-full" /> {m.name}
                  </div>
                  <div className="space-y-3 pl-2">
                    {Object.entries(m.hyperparams).map(([pk, opt]) => (
                      <div key={pk} className="space-y-1">
                        <div className="flex justify-between text-[11px]">
                          <span className="text-slate-600 capitalize">{pk.replace(/([A-Z])/g, ' $1')}</span>
                          <span className="font-mono text-slate-400">{opt.value}</span>
                        </div>
                        {opt.type === 'number' ? (
                          <input 
                            type="range" 
                            min={opt.min} max={opt.max} step={opt.step} 
                            value={opt.value} 
                            onChange={(e) => handleHyperparamChange(k, pk, parseFloat(e.target.value))} 
                            className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" 
                          />
                        ) : (
                          <select 
                            className="w-full bg-slate-50 border border-slate-200 rounded p-1 text-[11px] outline-none"
                            value={opt.value}
                            onChange={(e) => handleHyperparamChange(k, pk, e.target.value)}
                          >
                            {opt.options?.map(o => <option key={o} value={o}>{o}</option>)}
                          </select>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      </div>

      <div className="p-4 border-t border-slate-200 bg-slate-50">
        <button 
          onClick={generateData}
          className="w-full py-2 bg-indigo-600 text-white rounded font-bold text-sm hover:bg-indigo-700 transition shadow-sm flex items-center justify-center gap-2"
        >
          <Play size={14} fill="currentColor" /> Run Simulation
        </button>
      </div>
    </div>
  );

  return (
    <div className="flex w-full h-screen bg-[#F8FAFC] font-sans overflow-hidden">
      <Sidebar />
      
      <main className="flex-1 flex flex-col h-full overflow-hidden">
        <header className="px-8 pt-8 flex-none">
          <div className="flex gap-8 border-b border-slate-200 w-full mb-8">
            {[
              { id: 'viz', label: 'Visualization' },
              { id: 'metrics', label: 'Metrics Comparison' },
              { id: 'math', label: 'Neural Logic' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`pb-3 text-sm font-bold transition-all relative ${
                  activeTab === tab.id 
                  ? 'text-indigo-600' 
                  : 'text-slate-400 hover:text-slate-600'
                }`}
              >
                {tab.label}
                {activeTab === tab.id && (
                  <motion.div layoutId="tab-underline" className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-600" />
                )}
              </button>
            ))}
          </div>
        </header>

        <div className="flex-1 overflow-y-auto px-8 pb-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="max-w-7xl mx-auto"
            >
              {activeTab === 'viz' && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-2 gap-6">
                  {results.slice(0, 6).map(({ key, model, metrics }) => (
                    <div key={key} className="bg-white rounded-lg shadow-[0_1px_3px_rgba(0,0,0,0.05)] border border-slate-200 flex flex-col h-full group overflow-hidden">
                      <div className="p-3 border-b border-slate-100 flex justify-between items-center bg-white">
                        <h3 className="text-[10px] font-bold text-slate-700 uppercase tracking-widest">{model.name} | Decision Surface</h3>
                        <span className="text-[10px] font-black px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-600 border border-indigo-100">
                          ACC: {metrics.accuracy.toFixed(3)}
                        </span>
                      </div>
                      <div className="flex-1 p-4 flex flex-col justify-between">
                        <Visualization points={points} model={model} />
                        <div className="mt-4 p-3 bg-slate-50/50 rounded border border-slate-100 border-dashed">
                          <p className="text-[10px] text-slate-400 mb-1 uppercase font-bold tracking-tighter">Objective Logic</p>
                          <div className="text-[11px] font-math text-slate-600 italic line-clamp-1 pointer-events-none">
                            {model.mathFormula.replace(/\\\\/g, ' ')}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                  {results.length === 0 && (
                    <div className="col-span-full py-20 flex flex-col items-center justify-center text-slate-300 border-2 border-dashed border-slate-200 rounded-lg bg-white/50">
                      <Layers size={48} className="mb-4 opacity-50" />
                      <p className="text-sm font-bold uppercase tracking-widest text-slate-400">Select Models via Sidebar to Visualize</p>
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'metrics' && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                   {results.map(({ key, model, metrics }) => (
                     <div key={key} className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm">
                        <div className="flex justify-between items-center mb-6">
                           <h3 className="font-bold text-slate-700 text-sm uppercase tracking-widest flex items-center gap-2">
                             <div className="w-1.5 h-4 bg-indigo-500 rounded-full" /> {model.name}
                           </h3>
                           <div className="flex gap-2">
                              {['accuracy', 'f1'].map(m => (
                                <div key={m} className="px-2 py-1 bg-slate-50 rounded text-[10px] border border-slate-100">
                                   <span className="text-slate-400 uppercase mr-1">{m.slice(0,3)}:</span>
                                   <span className="font-bold text-slate-700">{(metrics as any)[m].toFixed(3)}</span>
                                </div>
                              ))}
                           </div>
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="h-48">
                              <ResponsiveContainer width="100%" height="100%">
                                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={[
                                  { subject: 'Accuracy', A: metrics.accuracy * 100 },
                                  { subject: 'Precision', A: metrics.precision * 100 },
                                  { subject: 'Recall', A: metrics.recall * 100 },
                                  { subject: 'F1 Score', A: metrics.f1 * 100 },
                                ]}>
                                  <PolarGrid stroke="#f1f5f9" />
                                  <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 700 }} />
                                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                  <Radar name="Metrics" dataKey="A" stroke="#6366f1" fill="#6366f1" fillOpacity={0.4} />
                                </RadarChart>
                              </ResponsiveContainer>
                          </div>
                          
                          <div className="flex flex-col justify-center">
                             <p className="text-[10px] font-bold text-slate-400 uppercase mb-3">Confusion Distribution</p>
                             <div className="grid grid-cols-2 gap-2 text-sm font-mono">
                                <div className="bg-slate-50 p-4 rounded border border-slate-200 flex flex-col items-center">
                                   <span className="text-[9px] text-slate-400 font-bold mb-1">TN</span>
                                   <span className="text-lg font-bold text-slate-700">{metrics.confusionMatrix[0][0]}</span>
                                </div>
                                <div className="bg-rose-50/30 p-4 rounded border border-rose-100 flex flex-col items-center">
                                   <span className="text-[9px] text-rose-400 font-bold mb-1">FP</span>
                                   <span className="text-lg font-bold text-rose-600">{metrics.confusionMatrix[0][1]}</span>
                                </div>
                                <div className="bg-rose-50/30 p-4 rounded border border-rose-100 flex flex-col items-center">
                                   <span className="text-[9px] text-rose-400 font-bold mb-1">FN</span>
                                   <span className="text-lg font-bold text-rose-600">{metrics.confusionMatrix[1][0]}</span>
                                </div>
                                <div className="bg-slate-50 p-4 rounded border border-slate-200 flex flex-col items-center">
                                   <span className="text-[9px] text-slate-400 font-bold mb-1">TP</span>
                                   <span className="text-lg font-bold text-slate-700">{metrics.confusionMatrix[1][1]}</span>
                                </div>
                             </div>
                          </div>
                        </div>
                     </div>
                   ))}
                </div>
              )}

              {activeTab === 'math' && (
                <div className="grid grid-cols-1 gap-4">
                  {results.map(({ key, model }) => (
                    <div key={key} className="bg-white p-8 rounded-lg border border-slate-200">
                      <div className="flex items-center gap-4 mb-4">
                        <div className="w-10 h-10 bg-slate-900 text-indigo-400 rounded flex items-center justify-center font-bold text-base">
                          {model.name[0]}
                        </div>
                        <div>
                          <h3 className="text-base font-bold text-slate-800 uppercase tracking-widest">{model.name}</h3>
                          <p className="text-xs text-slate-400 font-medium">Algorithmic Formulation & Constraints</p>
                        </div>
                      </div>
                      <div className="p-6 bg-slate-50 rounded border border-slate-100">
                        <MathView formula={model.mathFormula} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>

        <footer className="flex-none h-12 flex items-center justify-between border-t border-slate-200 px-8 bg-white">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"></div>
              <span className="text-[9px] text-slate-400 font-black uppercase tracking-widest">Engine Online</span>
            </div>
            <div className="text-[9px] text-slate-400 font-bold uppercase">LATENCY: <span className="text-slate-600">14ms</span></div>
            <div className="text-[9px] text-slate-400 font-bold uppercase">REGISTRY: <span className="text-slate-600">50+ Models / 100+ Datasets</span></div>
          </div>
          <div className="text-[9px] font-mono text-slate-400 font-bold">BUILD_ID: AI_S_98EF21A</div>
        </footer>
      </main>
    </div>
  );
}
