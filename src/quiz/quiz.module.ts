import { Module } from '@nestjs/common';
import { QuizController } from './quiz.controller';
import { QuizService } from './quiz.service';
import { QdrantService } from './qdrant.service';

@Module({
  controllers: [QuizController],
  providers: [QuizService, QdrantService],
})
export class QuizModule {}


