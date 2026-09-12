export type ExperimentStatus = "draft" | "running" | "completed" | "failed";
export type RunStatus = "pending" | "running" | "succeeded" | "failed";
export type Stage = "candidate" | "staging" | "production" | "archived";

export interface Dataset {
  id: string;
  slug: string;
  name: string;
  source: string;
  task: string;
  n_rows: number | null;
  n_features: number | null;
  target: string | null;
  feature_names: string[];
  target_names: string[] | null;
  description: string;
  path: string | null;
  created_at: string;
  used_in_runs?: number;
}

export interface Artifact {
  id: string;
  run_id: string;
  name: string;
  kind: string;
  path: string;
  mime: string | null;
  label: string | null;
}

export interface Run {
  id: string;
  experiment_id: string;
  name: string;
  model_name: string;
  status: RunStatus;
  task: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  params: Record<string, unknown>;
  metrics: Record<string, number>;
  artifacts: Artifact[];
  dataset: Dataset | null;
  rank?: number | null;
  primary_value?: number | null;
  delta_vs_best?: number | null;
  is_recommended?: boolean;
  constraint_failed?: boolean;
  missing_metric?: boolean;
  registry?: ModelVersion[];
  experiment?: Experiment;
}

export interface Experiment {
  id: string;
  name: string;
  description: string;
  primary_metric: string | null;
  maximize: boolean;
  status: ExperimentStatus;
  constraints: Array<{ metric: string; min?: number; max?: number }>;
  spec: Record<string, unknown>;
  error: string | null;
  created_at: string;
  updated_at: string;
  run_count?: number;
  candidate_count?: number;
  datasets?: string[];
  best_metric?: string | null;
  best_value?: number | null;
  winner_name?: string | null;
  winner_id?: string | null;
  progress?: { done: number; total: number; current: string | null };
  runs?: Run[];
  compare?: ComparePayload;
}

export interface ComparePayload {
  experiment_id: string;
  primary_metric: string;
  maximize: boolean;
  datasets: string[];
  tasks: string[];
  mixed_task: boolean;
  leaderboard: Run[];
  by_dataset: Record<string, Run[]>;
  recommended: Run | null;
  reason: string | null;
  metric_keys: string[];
  param_keys: string[];
  constraints: Array<{ metric: string; min?: number; max?: number }>;
  experiment?: Experiment;
}

export interface ModelVersion {
  id: string;
  model_id: string;
  version: number;
  run_id: string;
  stage: Stage;
  created_at: string;
  model_name?: string;
  run_name?: string;
  estimator?: string;
  experiment_id?: string;
  promotions?: Promotion[];
}

export interface Promotion {
  id: string;
  version_id: string;
  from_stage: string;
  to_stage: string;
  note: string;
  actor: string;
  created_at: string;
}

export interface RegisteredModel {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  versions: ModelVersion[];
  production: ModelVersion | null;
  alias: string | null;
}

export interface ModelCatalogItem {
  key: string;
  label: string;
  blurb: string;
  tasks: string[];
  params: Array<{
    key: string;
    label: string;
    type: string;
    default: number | null;
    min?: number;
    max?: number;
  }>;
}

export interface Bootstrap {
  version: string;
  home: string;
  stats: {
    experiments: number;
    runs: number;
    production_models: number;
    datasets: number;
  };
  catalog: {
    datasets: Dataset[];
    models: ModelCatalogItem[];
  };
}

export interface ConfusionData {
  labels: string[];
  matrix: number[][];
}

export interface ResidualData {
  y_true: number[];
  y_pred: number[];
  residuals: number[];
}
