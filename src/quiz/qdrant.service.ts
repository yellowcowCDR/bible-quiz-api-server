import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { QdrantClient } from '@qdrant/js-client-rest';

export type BibleSearchResult = {
  id: string;
  text: string;
  bookFile: string;
  chunkIndex: number;
  startLine: number;
  endLine: number;
  score: number;
};

@Injectable()
export class QdrantService {
  private readonly logger = new Logger(QdrantService.name);
  private readonly client?: QdrantClient;
  private readonly collectionName: string;

  constructor(private readonly configService: ConfigService) {
    const url = this.configService.get<string>('QDRANT_URL');
    const apiKey = this.configService.get<string>('QDRANT_API_KEY');
    this.collectionName =
      this.configService.get<string>('QDRANT_COLLECTION') ??
      'bible_embeddings';

    if (!url || !apiKey) {
      this.logger.warn(
        'QDRANT_URL 또는 QDRANT_API_KEY가 설정되어 있지 않아 Qdrant 연동이 비활성화됩니다.',
      );
      return;
    }

    this.client = new QdrantClient({ url, apiKey });
    this.logger.log(
      `Qdrant integration enabled. collection="${this.collectionName}"`,
    );
  }

  get isEnabled(): boolean {
    return !!this.client;
  }

  async searchBibleChunks(
    vector: number[],
    limit = 10,
  ): Promise<BibleSearchResult[]> {
    if (!this.client) {
      this.logger.warn('Qdrant client is not initialized.');
      return [];
    }

    const result = await this.client.search(this.collectionName, {
      vector,
      limit,
      with_payload: true,
      with_vector: false,
    });

    return result
      .map((point) => {
        const payload = (point.payload ?? {}) as Record<string, unknown>;
        const text = typeof payload.text === 'string' ? payload.text : '';
        const bookFile =
          typeof payload.bookFile === 'string' ? payload.bookFile : 'unknown';

        if (!text) {
          return null;
        }

        return {
          id: String(point.id),
          text,
          bookFile,
          chunkIndex:
            typeof payload.chunkIndex === 'number' ? payload.chunkIndex : -1,
          startLine:
            typeof payload.startLine === 'number' ? payload.startLine : -1,
          endLine:
            typeof payload.endLine === 'number' ? payload.endLine : -1,
          score: typeof point.score === 'number' ? point.score : 0,
        } as BibleSearchResult;
      })
      .filter((item): item is BibleSearchResult => item !== null);
  }
}


