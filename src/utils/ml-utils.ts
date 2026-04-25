/**
 * Utility functions for Machine Learning calculations and visualizations
 */

export interface Point {
  x: number;
  y: number;
  label: number;
}

export interface Dataset {
  points: Point[];
  name: string;
  description: string;
}

export interface MetricResults {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusionMatrix: number[][];
}

/**
 * Standardizes features to zero mean and unit variance
 */
export function scaleData(points: Point[]): Point[] {
  if (points.length === 0) return [];

  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);

  const meanX = xs.reduce((a, b) => a + b, 0) / xs.length;
  const meanY = ys.reduce((a, b) => a + b, 0) / ys.length;

  const stdX = Math.sqrt(xs.reduce((a, b) => a + Math.pow(b - meanX, 2), 0) / xs.length) || 1;
  const stdY = Math.sqrt(ys.reduce((a, b) => a + Math.pow(b - meanY, 2), 0) / ys.length) || 1;

  return points.map(p => ({
    x: (p.x - meanX) / stdX,
    y: (p.y - meanY) / stdY,
    label: p.label
  }));
}

/**
 * Generates a meshgrid for decision boundary visualization
 */
export function generateMeshGrid(
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  resolution: number = 40
) {
  const xStep = (xMax - xMin) / resolution;
  const yStep = (yMax - yMin) / resolution;
  const grid: { x: number, y: number }[] = [];

  for (let i = 0; i <= resolution; i++) {
    for (let j = 0; j <= resolution; j++) {
      grid.push({
        x: xMin + i * xStep,
        y: yMin + j * yStep
      });
    }
  }

  return { grid, resolution };
}

/**
 * Calculates metrics for binary classification
 */
export function calculateMetrics(actual: number[], predicted: number[]): MetricResults {
  let tp = 0, tn = 0, fp = 0, fn = 0;

  for (let i = 0; i < actual.length; i++) {
    if (actual[i] === 1 && predicted[i] === 1) tp++;
    else if (actual[i] === 0 && predicted[i] === 0) tn++;
    else if (actual[i] === 0 && predicted[i] === 1) fp++;
    else if (actual[i] === 1 && predicted[i] === 0) fn++;
  }

  const accuracy = (tp + tn) / actual.length;
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = (2 * precision * recall) / (precision + recall) || 0;

  return {
    accuracy,
    precision,
    recall,
    f1,
    confusionMatrix: [[tn, fp], [fn, tp]]
  };
}

/**
 * Simple CSV parser for 2D points
 */
export function parseCSV(text: string): Point[] {
  const lines = text.split('\n').filter(l => l.trim() !== '');
  const points: Point[] = [];
  
  lines.forEach((line, i) => {
    if (i === 0 && isNaN(parseFloat(line.split(',')[0]))) return; // Skip header
    const parts = line.split(',');
    if (parts.length >= 3) {
      points.push({
        x: parseFloat(parts[0]),
        y: parseFloat(parts[1]),
        label: Math.round(parseFloat(parts[2]))
      });
    }
  });
  
  return points;
}
