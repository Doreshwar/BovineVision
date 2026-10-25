export interface Prediction {
  rank: number;
  breed: string;
  confidence: number;
  confidence_percentage: number;
}

export interface ClassificationResult {
  is_accepted: boolean;
  rejection_message: string | null;
  can_trigger_rag: boolean;
  threshold_applied: number;
  threshold_percentage: number;
  top_prediction: Prediction | null;
  top_3_predictions: Prediction[];
  device: string;
  device_name: string;
}

export interface Citation {
  source: string;
  document: string;
  authority: string;
}

export interface BreedDossier {
  id: string;
  breed_name: string;
  scientific_classification: {
    species: string;
    nbagr_accession: string;
    category: string;
  };
  origin_and_geography: {
    native_tract: string;
    agro_climatic_zone: string;
    historical_background: string;
  };
  morphological_traits: {
    horns: string;
    forehead: string;
    coat_color: string;
    ears: string;
    hump_and_dewlap: string;
    body_conformation: string;
  };
  production_and_utility: {
    primary_utility: string;
    average_lactation_yield_kg: string;
    lactation_length_days: string;
    fat_percentage: string;
    draught_capacity: string;
  };
  management_and_resilience: {
    diet_and_forage: string;
    disease_resistance: string;
    temperament: string;
  };
  conservation_status: string;
  citations: Citation[];
}

export interface RAGAnswer {
  allowed: boolean;
  question: string;
  breed: string;
  answer: string;
  supporting_evidence?: {
    chunk_id: string;
    breed_name: string;
    section: string;
    score: number;
    text: string;
    citations: Citation[];
  }[];
  citations: Citation[];
  error?: string;
}

export interface ModelInfo {
  model_architecture: {
    name: string;
    type: string;
    total_parameters: number;
    trainable_parameters: number;
    input_resolution: string;
    num_classes: number;
    stages: { stage: number; channels: number; blocks: number; attention: string }[];
    stem: string;
    classification_head: string;
    pretrained: boolean;
    weights_origin: string;
  };
  classes: string[];
  device: string;
  metrics: {
    dataset_source: string;
    dataset_title: string;
    dataset_doi: string;
    total_authentic_images: number;
    split_counts: { train: number; val: number; test: number };
    best_epoch: number;
    best_val_accuracy: number;
    final_test_accuracy: number;
    training_time_minutes: number;
    device_name?: string;
    confusion_matrix: number[][];
    classification_report: Record<string, any>;
  };

}
