import React, { useRef, useEffect } from 'react';
import { Point, generateMeshGrid } from '../utils/ml-utils';
import { BaseModel } from '../ml/models';

interface VisualizationProps {
  points: Point[];
  model: BaseModel | null;
  resolution?: number;
}

export const Visualization: React.FC<VisualizationProps> = ({ points, model, resolution = 50 }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.clearRect(0, 0, width, height);

    if (points.length === 0) return;

    // Calculate bounds with padding
    const xMin = Math.min(...points.map(p => p.x)) - 0.5;
    const xMax = Math.max(...points.map(p => p.x)) + 0.5;
    const yMin = Math.min(...points.map(p => p.y)) - 0.5;
    const yMax = Math.max(...points.map(p => p.y)) + 0.5;

    const scaleX = (x: number) => ((x - xMin) / (xMax - xMin)) * width;
    const scaleY = (y: number) => height - ((y - yMin) / (yMax - yMin)) * height;

    // Draw Heatmap (Meshgrid)
    if (model) {
      const { grid, resolution: res } = generateMeshGrid(xMin, xMax, yMin, yMax, resolution);
      const cellW = width / res;
      const cellH = height / res;

      grid.forEach(p => {
        const proba = model.predictProba ? model.predictProba(p.x, p.y) : model.predict(p.x, p.y);
        // Map proba to color: 0 -> Rose/Red (rgba(244, 63, 94, 0.2)), 1 -> Indigo (rgba(99, 102, 241, 0.2))
        const r = 244 + (99 - 244) * proba;
        const g = 63 + (102 - 63) * proba;
        const b = 94 + (241 - 94) * proba;
        
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.2)`;
        ctx.fillRect(scaleX(p.x), scaleY(p.y), cellW, cellH);
      });
    }

    // Draw Points
    points.forEach(p => {
      ctx.beginPath();
      ctx.arc(scaleX(p.x), scaleY(p.y), 3.5, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 1 ? '#6366F1' : '#F43F5E';
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    });

  }, [points, model, resolution]);

  return (
    <div className="relative w-full aspect-square bg-[#F1F5F9] border border-slate-200 rounded-sm overflow-hidden flex items-center justify-center">
      <canvas 
        ref={canvasRef} 
        width={600} 
        height={600} 
        className="w-full h-full"
      />
      <div className="absolute bottom-2 right-2 flex gap-2">
        <div className="flex items-center gap-1.5 text-[9px] font-black tracking-widest uppercase text-slate-500 bg-white shadow-sm px-2 py-0.5 rounded border border-slate-200">
           <div className="w-1.5 h-1.5 rounded-full bg-[#f43f5e]"></div> CLASS B
        </div>
        <div className="flex items-center gap-1.5 text-[9px] font-black tracking-widest uppercase text-slate-500 bg-white shadow-sm px-2 py-0.5 rounded border border-slate-200">
           <div className="w-1.5 h-1.5 rounded-full bg-[#6366f1]"></div> CLASS A
        </div>
      </div>
    </div>
  );
};
