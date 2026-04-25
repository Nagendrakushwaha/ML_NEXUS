import { Point } from '../utils/ml-utils';

export abstract class BaseModel {
  abstract name: string;
  abstract description: string;
  abstract mathFormula: string;
  abstract hyperparams: Record<string, { type: 'number' | 'select', min?: number, max?: number, step?: number, options?: string[], value: any }>;

  abstract train(points: Point[]): void;
  abstract predict(x: number, y: number): number;
  abstract predictProba?(x: number, y: number): number; // Probability of class 1
}

/**
 * K-Nearest Neighbors
 */
export class KNN extends BaseModel {
  name = "K-Nearest Neighbors";
  description = "A distance-based classifier that assigns a label based on the majority vote of its K closest neighbors.";
  mathFormula = "d(x, x_i) = \\sqrt{\\sum (x_j - x_{i,j})^2} \\\\ \\hat{y} = \\text{mode}(y_{i \\in N_k})";
  hyperparams = {
    k: { type: 'number' as const, min: 1, max: 21, step: 1, value: 5 }
  };

  private data: Point[] = [];

  train(points: Point[]) {
    this.data = points;
  }

  predict(x: number, y: number): number {
    const k = this.hyperparams.k.value;
    const distances = this.data.map(p => ({
      dist: Math.sqrt(Math.pow(p.x - x, 2) + Math.pow(p.y - y, 2)),
      label: p.label
    }));

    distances.sort((a, b) => a.dist - b.dist);
    const neighbors = distances.slice(0, k);

    const counts: Record<number, number> = {};
    neighbors.forEach(n => {
      counts[n.label] = (counts[n.label] || 0) + 1;
    });

    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0] as any * 1;
  }

  predictProba(x: number, y: number): number {
    const k = this.hyperparams.k.value;
    const distances = this.data.map(p => ({
      dist: Math.sqrt(Math.pow(p.x - x, 2) + Math.pow(p.y - y, 2)),
      label: p.label
    }));

    distances.sort((a, b) => a.dist - b.dist);
    const neighbors = distances.slice(0, k);
    const ones = neighbors.filter(n => n.label === 1).length;
    return ones / k;
  }
}

/**
 * Logistic Regression
 */
export class LogisticRegression extends BaseModel {
  name = "Logistic Regression";
  description = "A linear model that estimates the probability of a binary outcome using the logistic sigmoid function.";
  mathFormula = "P(y=1|x) = \\sigma(w^T x + b) = \\frac{1}{1 + e^{-(w^T x + b)}}";
  hyperparams = {
    learningRate: { type: 'number' as const, min: 0.001, max: 1, step: 0.01, value: 0.1 },
    epochs: { type: 'number' as const, min: 10, max: 1000, step: 10, value: 100 }
  };

  private w: number[] = [0, 0];
  private b: number = 0;

  private sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
  }

  train(points: Point[]) {
    const lr = this.hyperparams.learningRate.value;
    const epochs = this.hyperparams.epochs.value;
    this.w = [0, 0];
    this.b = 0;

    for (let e = 0; e < epochs; e++) {
      let dw = [0, 0];
      let db = 0;

      points.forEach(p => {
        const z = this.w[0] * p.x + this.w[1] * p.y + this.b;
        const a = this.sigmoid(z);
        const dz = a - p.label;
        dw[0] += dz * p.x;
        dw[1] += dz * p.y;
        db += dz;
      });

      this.w[0] -= lr * (dw[0] / points.length);
      this.w[1] -= lr * (dw[1] / points.length);
      this.b -= lr * (db / points.length);
    }
  }

  predict(x: number, y: number): number {
    return this.predictProba(x, y) > 0.5 ? 1 : 0;
  }

  predictProba(x: number, y: number): number {
    const z = this.w[0] * x + this.w[1] * y + this.b;
    return this.sigmoid(z);
  }
}

/**
 * Decision Tree (Simplified)
 */
export class DecisionTree extends BaseModel {
  name = "Decision Tree";
  description = "A non-parametric model that splits the feature space into axis-aligned boxes based on information gain or Gini impurity.";
  mathFormula = "Gini = 1 - \\sum p_i^2 \\\\ \\text{Info Gain} = H(S) - \\sum \\frac{|S_v|}{|S|} H(S_v)";
  hyperparams = {
    maxDepth: { type: 'number' as const, min: 1, max: 10, step: 1, value: 3 }
  };

  private tree: any = null;

  train(points: Point[]) {
    this.tree = this.buildTree(points, 0);
  }

  private buildTree(data: Point[], depth: number): any {
    if (depth >= this.hyperparams.maxDepth.value || data.length < 2) {
      const labels = data.map(d => d.label);
      const majority = labels.filter(l => l === 1).length > labels.length / 2 ? 1 : 0;
      return { leaf: true, label: majority, proba: labels.filter(l => l === 1).length / labels.length };
    }

    // Very simple split search (search 10 steps per axis)
    let bestSplit = { feature: 'x', val: 0, gain: -1 };
    ['x', 'y'].forEach(f => {
      const vals = data.map(d => (d as any)[f]);
      const min = Math.min(...vals);
      const max = Math.max(...vals);
      for (let i = 1; i < 10; i++) {
        const threshold = min + (max - min) * (i / 10);
        const left = data.filter(d => (d as any)[f] <= threshold);
        const right = data.filter(d => (d as any)[f] > threshold);
        if (left.length === 0 || right.length === 0) continue;

        const gain = this.calculateGiniGain(data, left, right);
        if (gain > bestSplit.gain) {
          bestSplit = { feature: f, val: threshold, gain };
        }
      }
    });

    if (bestSplit.gain === -1) {
      const labels = data.map(d => d.label);
      return { leaf: true, label: labels.filter(l => l === 1).length > labels.length / 2 ? 1 : 0, proba: labels.filter(l => l === 1).length / labels.length };
    }

    const leftData = data.filter(d => (d as any)[bestSplit.feature] <= bestSplit.val);
    const rightData = data.filter(d => (d as any)[bestSplit.feature] > bestSplit.val);

    return {
      leaf: false,
      feature: bestSplit.feature,
      val: bestSplit.val,
      left: this.buildTree(leftData, depth + 1),
      right: this.buildTree(rightData, depth + 1)
    };
  }

  private calculateGiniGain(parent: Point[], left: Point[], right: Point[]) {
    const gini = (data: Point[]) => {
      const p1 = data.filter(d => d.label === 1).length / data.length;
      return 1 - (p1 * p1 + (1 - p1) * (1 - p1));
    };
    const pGini = gini(parent);
    const lGini = gini(left);
    const rGini = gini(right);
    return pGini - (left.length / parent.length * lGini + right.length / parent.length * rGini);
  }

  predict(x: number, y: number): number {
    let node = this.tree;
    while (node && !node.leaf) {
      const val = node.feature === 'x' ? x : y;
      node = val <= node.val ? node.left : node.right;
    }
    return node ? node.label : 0;
  }

  predictProba(x: number, y: number): number {
    let node = this.tree;
    while (node && !node.leaf) {
      const val = node.feature === 'x' ? x : y;
      node = val <= node.val ? node.left : node.right;
    }
    return node ? node.proba : 0.5;
  }
}

/**
 * Naive Bayes (Gaussian)
 */
export class NaiveBayes extends BaseModel {
  name = "Naive Bayes";
  description = "A classifier based on Bayes' theorem with the 'naive' assumption of feature independence.";
  mathFormula = "P(y|X) \\propto P(y) \\prod P(x_i|y) \\\\ P(x|y) = \\frac{1}{\\sqrt{2\\pi\\sigma_y^2}} e^{-\\frac{(x-\\mu_y)^2}{2\\sigma_y^2}}";
  hyperparams = {};

  private stats: any = {};

  train(points: Point[]) {
    const classes = [0, 1];
    classes.forEach(c => {
      const subset = points.filter(p => p.label === c);
      if (subset.length === 0) {
        this.stats[c] = { prior: 0, x: { mean: 0, var: 1 }, y: { mean: 0, var: 1 } };
        return;
      }
      const x = subset.map(p => p.x);
      const y = subset.map(p => p.y);
      this.stats[c] = {
        prior: subset.length / points.length,
        x: { mean: this.mean(x), var: this.variance(x) + 1e-9 },
        y: { mean: this.mean(y), var: this.variance(y) + 1e-9 }
      };
    });
  }

  private mean(arr: number[]) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
  private variance(arr: number[]) {
    const m = this.mean(arr);
    return arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / arr.length;
  }

  private pdf(x: number, mean: number, variance: number) {
    const exponent = Math.exp(-Math.pow(x - mean, 2) / (2 * variance));
    return (1 / Math.sqrt(2 * Math.PI * variance)) * exponent;
  }

  predict(x: number, y: number): number {
    return this.predictProba(x, y) > 0.5 ? 1 : 0;
  }

  predictProba(x: number, y: number): number {
    const p1 = this.stats[1].prior * this.pdf(x, this.stats[1].x.mean, this.stats[1].x.var) * this.pdf(y, this.stats[1].y.mean, this.stats[1].y.var);
    const p0 = this.stats[0].prior * this.pdf(x, this.stats[0].x.mean, this.stats[0].x.var) * this.pdf(y, this.stats[0].y.mean, this.stats[0].y.var);
    const total = p1 + p0;
    return total === 0 ? 0.5 : p1 / total;
  }
}

/**
 * Support Vector Machine (Linear Kernel Simplified)
 */
export class SVM extends BaseModel {
  name = "SVM";
  description = "A classifier that finds the hyperplane which maximizes the margin between classes.";
  mathFormula = "\\min \\frac{1}{2}||w||^2 + C \\sum \\xi_i \\\\ \\text{s.t. } y_i(w^T x_i + b) \\ge 1 - \\xi_i";
  hyperparams = {
    C: { type: 'number' as const, min: 0.1, max: 10, step: 0.1, value: 1.0 },
    learningRate: { type: 'number' as const, min: 0.001, max: 0.5, step: 0.01, value: 0.01 },
    epochs: { type: 'number' as const, min: 10, max: 1000, step: 10, value: 100 }
  };

  private w: number[] = [0, 0];
  private b: number = 0;

  train(points: Point[]) {
    const lr = this.hyperparams.learningRate.value;
    const epochs = this.hyperparams.epochs.value;
    const C = this.hyperparams.C.value;
    this.w = [0.1, 0.1];
    this.b = 0;

    for (let e = 0; e < epochs; e++) {
      points.forEach(p => {
        const y = p.label === 1 ? 1 : -1;
        const condition = y * (this.w[0] * p.x + this.w[1] * p.y + this.b) >= 1;
        if (condition) {
          this.w[0] -= lr * (2 * 1/epochs * this.w[0]);
          this.w[1] -= lr * (2 * 1/epochs * this.w[1]);
        } else {
          this.w[0] -= lr * (2 * 1/epochs * this.w[0] - C * y * p.x);
          this.w[1] -= lr * (2 * 1/epochs * this.w[1] - C * y * p.y);
          this.b += lr * C * y;
        }
      });
    }
  }

  predict(x: number, y: number): number {
    return (this.w[0] * x + this.w[1] * y + this.b) >= 0 ? 1 : 0;
  }

  predictProba(x: number, y: number): number {
    const z = this.w[0] * x + this.w[1] * y + this.b;
    return 1 / (1 + Math.exp(-z)); // Sigmoid mapping for visualization
  }
}

/**
 * Random Forest
 */
export class RandomForest extends BaseModel {
  name = "Random Forest";
  description = "An ensemble method that trains multiple decision trees on random subsets of the data.";
  mathFormula = "f(x) = \\frac{1}{N} \\sum T_i(x) \\\\ \\text{Bias-Variance tradeoff optimized}";
  hyperparams = {
    nEstimators: { type: 'number' as const, min: 1, max: 20, step: 1, value: 10 },
    maxDepth: { type: 'number' as const, min: 1, max: 10, step: 1, value: 5 }
  };

  private trees: DecisionTree[] = [];

  train(points: Point[]) {
    const n = this.hyperparams.nEstimators.value;
    this.trees = [];
    for (let i = 0; i < n; i++) {
      // Bootstrap sampling
      const sample = Array.from({ length: points.length }, () => points[Math.floor(Math.random() * points.length)]);
      const tree = new DecisionTree();
      tree.hyperparams.maxDepth.value = this.hyperparams.maxDepth.value;
      tree.train(sample);
      this.trees.push(tree);
    }
  }

  predict(x: number, y: number): number {
    const votes = this.trees.map(t => t.predict(x, y));
    const counts: Record<number, number> = {};
    votes.forEach(v => counts[v] = (counts[v] || 0) + 1);
    return (counts[1] || 0) > votes.length / 2 ? 1 : 0;
  }

  predictProba(x: number, y: number): number {
    const probas = this.trees.map(t => t.predictProba(x, y));
    return probas.reduce((a, b) => a + b, 0) / probas.length;
  }
}

/**
 * MLP Classifier (Neural Network)
 */
export class MLPClassifier extends BaseModel {
  name = "Neural Network";
  description = "A multi-layer perceptron with one hidden layer using ReLU and Softmax-like probability.";
  mathFormula = "h = \\text{ReLU}(W^{(1)}x + b^{(1)}) \\\\ \\hat{y} = \\sigma(W^{(2)}h + b^{(2)})";
  hyperparams = {
    hiddenNodes: { type: 'number' as const, min: 2, max: 16, step: 1, value: 8 },
    learningRate: { type: 'number' as const, min: 0.01, max: 0.5, step: 0.01, value: 0.1 },
    epochs: { type: 'number' as const, min: 50, max: 1000, step: 50, value: 200 }
  };

  private w1: number[][] = [];
  private b1: number[] = [];
  private w2: number[] = [];
  private b2: number = 0;

  train(points: Point[]) {
    const nodes = this.hyperparams.hiddenNodes.value;
    const lr = this.hyperparams.learningRate.value;
    const epochs = this.hyperparams.epochs.value;

    // Init weights
    this.w1 = Array.from({ length: nodes }, () => [Math.random() - 0.5, Math.random() - 0.5]);
    this.b1 = Array.from({ length: nodes }, () => 0);
    this.w2 = Array.from({ length: nodes }, () => Math.random() - 0.5);
    this.b2 = 0;

    for (let e = 0; e < epochs; e++) {
      points.forEach(p => {
        // Forward
        const h: number[] = this.w1.map((w, i) => Math.max(0, w[0] * p.x + w[1] * p.y + this.b1[i]));
        const z2 = h.reduce((acc, val, i) => acc + val * this.w2[i], 0) + this.b2;
        const a2 = 1 / (1 + Math.exp(-z2));

        // Backprop
        const dz2 = a2 - p.label;
        const dw2 = h.map(val => dz2 * val);
        const db2 = dz2;

        const dh = this.w2.map(w => dz2 * w);
        const dz1 = h.map((val, i) => val > 0 ? dh[i] : 0);
        const dw1 = dz1.map(val => [val * p.x, val * p.y]);
        const db1 = dz1;

        // Update
        this.w2 = this.w2.map((w, i) => w - lr * dw2[i]);
        this.b2 -= lr * db2;
        this.w1 = this.w1.map((w, i) => [w[0] - lr * dw1[i][0], w[1] - lr * dw1[i][1]]);
        this.b1 = this.b1.map((b, i) => b - lr * db1[i]);
      });
    }
  }

  predict(x: number, y: number): number {
    return this.predictProba(x, y) > 0.5 ? 1 : 0;
  }

  predictProba(x: number, y: number): number {
    const h: number[] = this.w1.map((w, i) => Math.max(0, w[0] * x + w[1] * y + this.b1[i]));
    const z2 = h.reduce((acc, val, i) => acc + val * this.w2[i], 0) + this.b2;
    return 1 / (1 + Math.exp(-z2));
  }
}

/**
 * AdaBoost (Simplified)
 * Using Decision Stumps as weak learners
 */
export class AdaBoost extends BaseModel {
  name = "AdaBoost";
  description = "A boosting ensemble that combines sequential weak learners by focusing on previously misclassified points.";
  mathFormula = "F(x) = \\text{sign}(\\sum_{t=1}^T \\alpha_t h_t(x)) \\\\ \\alpha_t = \\frac{1}{2} \\ln(\\frac{1-\\epsilon_t}{\\epsilon_t})";
  hyperparams = {
    nEstimators: { type: 'number' as const, min: 1, max: 50, step: 1, value: 20 }
  };

  private alphas: number[] = [];
  private stumps: any[] = [];

  train(points: Point[]) {
    const n = points.length;
    const T = this.hyperparams.nEstimators.value;
    let weights = Array(n).fill(1 / n);
    this.alphas = [];
    this.stumps = [];

    for (let t = 0; t < T; t++) {
      let bestStump = { feature: 'x', threshold: 0, polarity: 1, error: Infinity };
      
      // Find best stump
      ['x', 'y'].forEach(f => {
        const vals = points.map(p => (p as any)[f]);
        const uniqueVals = Array.from(new Set(vals)).sort((a,b) => a-b);
        
        uniqueVals.forEach(threshold => {
          [1, -1].forEach(polarity => {
            let error = 0;
            points.forEach((p, i) => {
              const label = p.label === 1 ? 1 : -1;
              const val = (p as any)[f];
              const pred = (polarity * val < polarity * threshold) ? -1 : 1;
              if (pred !== label) error += weights[i];
            });

            if (error < bestStump.error) {
              bestStump = { feature: f, threshold, polarity, error };
            }
          });
        });
      });

      const EPS = 1e-10;
      const alpha = 0.5 * Math.log((1 - bestStump.error + EPS) / (bestStump.error + EPS));
      this.alphas.push(alpha);
      this.stumps.push(bestStump);

      // Update weights
      let totalWeight = 0;
      points.forEach((p, i) => {
        const label = p.label === 1 ? 1 : -1;
        const val = (p as any)[bestStump.feature];
        const pred = (bestStump.polarity * val < bestStump.polarity * bestStump.threshold) ? -1 : 1;
        weights[i] *= Math.exp(-alpha * label * pred);
        totalWeight += weights[i];
      });
      weights = weights.map(w => w / totalWeight);
    }
  }

  predict(x: number, y: number): number {
    let sum = 0;
    for (let i = 0; i < this.stumps.length; i++) {
      const s = this.stumps[i];
      const val = s.feature === 'x' ? x : y;
      const pred = (s.polarity * val < s.polarity * s.threshold) ? -1 : 1;
      sum += this.alphas[i] * pred;
    }
    return sum >= 0 ? 1 : 0;
  }

  predictProba(x: number, y: number): number {
    let sum = 0;
    for (let i = 0; i < this.stumps.length; i++) {
      const s = this.stumps[i];
      const val = s.feature === 'x' ? x : y;
      const pred = (s.polarity * val < s.polarity * s.threshold) ? -1 : 1;
      sum += this.alphas[i] * pred;
    }
    return 1 / (1 + Math.exp(-sum));
  }
}
