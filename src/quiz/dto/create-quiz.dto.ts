import { Transform } from 'class-transformer';
import { IsIn, IsInt, IsNotEmpty, IsString, Max, Min } from 'class-validator';
import type { QuizDifficulty } from '../types/quiz-question.interface';

const DIFFICULTY_OPTIONS: QuizDifficulty[] = ['easy', 'medium', 'hard'];

export class CreateQuizDto {
  @IsString()
  @IsNotEmpty()
  topic!: string;

  @Transform(({ value }) => Number(value))
  @IsInt()
  @Min(1)
  @Max(10)
  count = 5;

  @IsIn(DIFFICULTY_OPTIONS)
  difficulty: QuizDifficulty = 'medium';
}
