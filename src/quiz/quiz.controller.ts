import { Body, Controller, Post } from '@nestjs/common';
import { CreateQuizDto } from './dto/create-quiz.dto';
import { QuizService } from './quiz.service';
import type { QuizResponse } from './types/quiz-question.interface';

@Controller('quiz')
export class QuizController {
  constructor(private readonly quizService: QuizService) {}

  @Post()
  async generateQuiz(
    @Body() createQuizDto: CreateQuizDto,
  ): Promise<QuizResponse> {
    return this.quizService.generateQuiz(createQuizDto);
  }
}
