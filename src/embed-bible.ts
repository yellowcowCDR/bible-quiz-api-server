import fs from 'fs/promises';
import path from 'path';
import OpenAI from 'openai';
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

// __dirname: .../biblequiz/backend/src
// bible_utf8: .../biblequiz/bible_utf8
const BIBLE_DIR = path.resolve(__dirname, '../../../bible_utf8');
const OUTPUT_PATH = path.resolve(__dirname, '../bible_embeddings.json');
const EMBEDDING_MODEL = 'text-embedding-3-small';

const CHUNK_SIZE_LINES = 5;
const EMBEDDING_BATCH_SIZE = 64;

// 우선 .env.production을 시도, 없으면 기본 .env도 같이 시도
dotenv.config({ path: path.resolve(__dirname, '../.env.local') });
dotenv.config({ path: path.resolve(__dirname, '../.env.production') });

async function ensureEnv() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.');
  }
}

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function loadBibleFiles(): Promise<string[]> {
  const entries = await fs.readdir(BIBLE_DIR);
  return entries
    .filter((name) => name.toLowerCase().endsWith('.txt'))
    .sort();
}

function makeChunksFromText(bookFile: string, content: string): BibleChunk[] {
  const lines = content.split(/\r?\n/);
  const chunks: BibleChunk[] = [];

  let currentLines: string[] = [];
  let startLine = 1;
  let chunkIndex = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    if (currentLines.length === 0) {
      startLine = i + 1;
    }

    currentLines.push(line);

    if (currentLines.length >= CHUNK_SIZE_LINES) {
      const text = currentLines.join('\n');
      chunks.push({
        id: `${bookFile}#${chunkIndex}`,
        bookFile,
        chunkIndex,
        startLine,
        endLine: i + 1,
        text,
      });
      chunkIndex++;
      currentLines = [];
    }
  }

  // 남은 라인 처리
  if (currentLines.length > 0) {
    const endLine = lines.length;
    const text = currentLines.join('\n');
    chunks.push({
      id: `${bookFile}#${chunkIndex}`,
      bookFile,
      chunkIndex,
      startLine,
      endLine,
      text,
    });
  }

  return chunks;
}

async function createEmbeddingsForChunks(chunks: BibleChunk[]): Promise<void> {
  for (let i = 0; i < chunks.length; i += EMBEDDING_BATCH_SIZE) {
    const batch = chunks.slice(i, i + EMBEDDING_BATCH_SIZE);
    const input = batch.map((c) => c.text);

    const response = await client.embeddings.create({
      model: EMBEDDING_MODEL,
      input,
    });

    response.data.forEach((item, idx) => {
      batch[idx].embedding = item.embedding;
    });

    // 간단한 진행 로그
    // eslint-disable-next-line no-console
    console.log(
      `Embedded ${i + batch.length}/${chunks.length} chunks (${(
        ((i + batch.length) / chunks.length) *
        100
      ).toFixed(1)}%)`,
    );
  }
}

async function main() {
  try {
    await ensureEnv();

    // eslint-disable-next-line no-console
    console.log('Bible directory:', BIBLE_DIR);

    const files = await loadBibleFiles();
    // eslint-disable-next-line no-console
    console.log(`Found ${files.length} bible files.`);

    const allChunks: BibleChunk[] = [];

    for (const file of files) {
      const filePath = path.join(BIBLE_DIR, file);
      const content = await fs.readFile(filePath, 'utf8');
      const chunks = makeChunksFromText(file, content);
      allChunks.push(...chunks);
      // eslint-disable-next-line no-console
      console.log(`${file}: ${chunks.length} chunks`);
    }

    // eslint-disable-next-line no-console
    console.log(`Total chunks: ${allChunks.length}`);

    await createEmbeddingsForChunks(allChunks);

    const output = {
      model: EMBEDDING_MODEL,
      createdAt: new Date().toISOString(),
      chunkSizeLines: CHUNK_SIZE_LINES,
      entries: allChunks,
    };

    await fs.writeFile(OUTPUT_PATH, JSON.stringify(output), 'utf8');

    // eslint-disable-next-line no-console
    console.log('Saved embeddings to', OUTPUT_PATH);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error('Embedding script failed:', err);
    process.exit(1);
  }
}

void main();


