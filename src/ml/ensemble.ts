import { BaseModel } from './models';
import { Point } from '../utils/ml-utils';

export class VotingClassifier extends BaseModel {
  name = "Voting Ensemble";
  description = "An ensemble method that combines multiple classifiers. Supports hard (majority vote) and soft (average probabilities) voting.";
  mathFormula = "\\text{Hard: } \\hat{y} = \\text{mode}(C_1(x), ..., C_n(x)) \\\\ \\text{Soft: } \\hat{P} = \\frac{1}{n} \\sum P_i(y=1|x)";
  hyperparams = {
    votingType: { type: 'select' as const, options: ['hard', 'soft'], value: 'soft' }
  };

  private models: BaseModel[] = [];

  constructor(models: BaseModel[]) {
    super();
    this.models = models;
  }

  train(points: Point[]) {
    // Models are assumed to be trained externally or trained here
    this.models.forEach(m => m.train(points));
  }

  predict(x: number, y: number): number {
    if (this.hyperparams.votingType.value === 'hard') {
      const votes = this.models.map(m => m.predict(x, y));
      const counts: Record<number, number> = {};
      votes.forEach(v => counts[v] = (counts[v] || 0) + 1);
      return Object.entries(counts).sort((a,b) => b[1] - a[1])[0][0] as any * 1;
    } else {
      return this.predictProba(x, y) > 0.5 ? 1 : 0;
    }
  }

  predictProba(x: number, y: number): number {
    const probas = this.models.map(m => m.predictProba ? m.predictProba(x, y) : m.predict(x, y));
    return probas.reduce((a, b) => a + b, 0) / probas.length;
  }
}
