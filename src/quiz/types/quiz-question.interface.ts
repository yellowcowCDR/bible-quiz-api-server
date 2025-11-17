export type QuizDifficulty = 'easy' | 'medium' | 'hard';

export interface QuizQuestion {
  id: string;
  prompt: string;
  options: string[];
  answer: string;
  difficulty: QuizDifficulty;
  reference: string;
  explanation: string;
}

export interface QuizResponse {
  topic: string;
  difficulty: QuizDifficulty;
  count: number;
  questions: QuizQuestion[];
  generatedAt: string;
}


