import fs from 'fs/promises';
import path from 'path';
import { QdrantClient } from '@qdrant/js-client-rest';
import dotenv from 'dotenv';

type BibleChunk = {
  id: string;
  bookFile: string;
  chunkIndex: number;
  startLine: number;
  endLine: number;
  text: string;
  embedding?: number[];
};

type EmbeddingFile = {
  model: string;
  createdAt: string;
  chunkSizeLines: number;
  entries: BibleChunk[];
};

const COLLECTION_NAME = 'bible_embeddings';
const EMBEDDING_DIM = 1536; // text-embedding-3-small

// 로컬 실행 시 .env.local, .env.production에서 환경변수 로드 시도
dotenv.config({ path: path.resolve(__dirname, '../.env.local') });
dotenv.config({ path: path.resolve(__dirname, '../.env.production') });

async function main() {
  const qdrantUrl = process.env.QDRANT_URL;
  const qdrantApiKey = process.env.QDRANT_API_KEY;

  if (!qdrantUrl || !qdrantApiKey) {
    throw new Error('QDRANT_URL, QDRANT_API_KEY 환경 변수가 필요합니다.');
  }

  const client = new QdrantClient({
    url: qdrantUrl,
    apiKey: qdrantApiKey,
  });

  const filePath = path.resolve(__dirname, '../bible_embeddings.json');
  const raw = await fs.readFile(filePath, 'utf8');
  const parsed = JSON.parse(raw) as EmbeddingFile;

  // eslint-disable-next-line no-console
  console.log(
    `Loaded ${parsed.entries.length} embedding entries from ${filePath}`,
  );

  // 컬렉션이 없으면 생성, 있으면 그대로 사용
  try {
    await client.getCollection(COLLECTION_NAME);
    // eslint-disable-next-line no-console
    console.log(
      `Collection "${COLLECTION_NAME}" already exists. Skipping create.`,
    );
  } catch {
    // eslint-disable-next-line no-console
    console.log(`Creating collection "${COLLECTION_NAME}"...`);
    await client.createCollection(COLLECTION_NAME, {
      vectors: {
        size: EMBEDDING_DIM,
        distance: 'Cosine',
      },
    });
  }

  const batchSize = 256;
  let pointId = 0;

  for (let i = 0; i < parsed.entries.length; i += batchSize) {
    const batch = parsed.entries.slice(i, i + batchSize);

    const points = batch
      .filter((entry) => Array.isArray(entry.embedding))
      .map((entry) => {
        const currentId = pointId;
        pointId += 1;

        return {
          id: currentId, // Qdrant: unsigned integer ID
          vector: entry.embedding as number[],
          payload: {
            originalId: entry.id,
            bookFile: entry.bookFile,
            chunkIndex: entry.chunkIndex,
            startLine: entry.startLine,
            endLine: entry.endLine,
            text: entry.text,
          },
        };
      });

    await client.upsert(COLLECTION_NAME, { points });

    // eslint-disable-next-line no-console
    console.log(`Upserted ${i + batch.length}/${parsed.entries.length}`);
  }

  // eslint-disable-next-line no-console
  console.log('Done seeding Qdrant.');
}

void main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error('Seed script failed:', err);
  process.exit(1);
});


