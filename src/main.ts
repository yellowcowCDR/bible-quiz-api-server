import { Logger, ValidationPipe } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const logger = new Logger('Bootstrap');
  const app = await NestFactory.create(AppModule, {
    bufferLogs: true,
  });
  app.useLogger(logger);

  const allowedOrigins = (process.env.FRONTEND_URLS ?? 'http://localhost:3000')
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);

  app.enableCors({
    origin: allowedOrigins,
    credentials: true,
  });

  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
      transformOptions: { enableImplicitConversion: true },
    }),
  );

  const isProduction = process.env.NODE_ENV === 'production';
  const port = process.env.PORT ? Number(process.env.PORT) : 3001;
  await app.listen(port);

  logger.log(
    `🚀 Quiz API running on ${
      isProduction ? 'production' : 'local'
    } port ${port}`,
  );
}
bootstrap().catch((error: unknown) => {
  const logger = new Logger('Bootstrap');
  const stack =
    error instanceof Error ? (error.stack ?? error.message) : String(error);
  logger.error('Failed to bootstrap application', stack);
  process.exit(1);
});
