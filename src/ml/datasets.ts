import { Point } from '../utils/ml-utils';

export interface DatasetGeneratorOptions {
  samples: number;
  noise: number;
  imbalance: number; // 0.5 is balanced, 0.1 is 10% class 1
}

export type DatasetGenerator = (options: DatasetGeneratorOptions) => Point[];

export const datasetRegistry: Record<string, { label: string, generate: DatasetGenerator }> = {
  moons: {
    label: "Two Moons",
    generate: ({ samples, noise }) => {
      const points: Point[] = [];
      const nSamplesOut = Math.floor(samples / 2);
      const nSamplesIn = samples - nSamplesOut;

      // Outer moon
      for (let i = 0; i < nSamplesOut; i++) {
        const t = (i / nSamplesOut) * Math.PI;
        points.push({
          x: Math.cos(t) + (Math.random() - 0.5) * noise,
          y: Math.sin(t) + (Math.random() - 0.5) * noise,
          label: 0
        });
      }

      // Inner moon
      for (let i = 0; i < nSamplesIn; i++) {
        const t = (i / nSamplesIn) * Math.PI;
        points.push({
          x: 1 - Math.cos(t) + (Math.random() - 0.5) * noise,
          y: 0.5 - Math.sin(t) + (Math.random() - 0.5) * noise,
          label: 1
        });
      }
      return points;
    }
  },
  circles: {
    label: "Circles",
    generate: ({ samples, noise }) => {
      const points: Point[] = [];
      const nSamplesOut = Math.floor(samples / 2);
      const nSamplesIn = samples - nSamplesOut;

      // Outer circle
      for (let i = 0; i < nSamplesOut; i++) {
        const r = 1.0;
        const t = Math.random() * 2 * Math.PI;
        points.push({
          x: r * Math.cos(t) + (Math.random() - 0.5) * noise,
          y: r * Math.sin(t) + (Math.random() - 0.5) * noise,
          label: 0
        });
      }

      // Inner circle
      for (let i = 0; i < nSamplesIn; i++) {
        const r = 0.5;
        const t = Math.random() * 2 * Math.PI;
        points.push({
          x: r * Math.cos(t) + (Math.random() - 0.5) * noise,
          y: r * Math.sin(t) + (Math.random() - 0.5) * noise,
          label: 1
        });
      }
      return points;
    }
  },
  blobs: {
    label: "Gaussian Blobs",
    generate: ({ samples, noise, imbalance }) => {
      const points: Point[] = [];
      const nSamples1 = Math.floor(samples * imbalance);
      const nSamples0 = samples - nSamples1;

      // Cluster 0
      for (let i = 0; i < nSamples0; i++) {
        points.push({
          x: -1 + (Math.random() - 0.5) * noise * 2,
          y: -1 + (Math.random() - 0.5) * noise * 2,
          label: 0
        });
      }

      // Cluster 1
      for (let i = 0; i < nSamples1; i++) {
        points.push({
          x: 1 + (Math.random() - 0.5) * noise * 2,
          y: 1 + (Math.random() - 0.5) * noise * 2,
          label: 1
        });
      }
      return points;
    }
  },
  linear: {
    label: "Linearly Separable",
    generate: ({ samples, noise }) => {
      const points: Point[] = [];
      for (let i = 0; i < samples; i++) {
        const x = (Math.random() - 0.5) * 4;
        const y = (Math.random() - 0.5) * 4;
        const label = y > x ? 1 : 0;
        points.push({
          x: x + (Math.random() - 0.5) * noise,
          y: y + (Math.random() - 0.5) * noise,
          label
        });
      }
      return points;
    }
  }
};
