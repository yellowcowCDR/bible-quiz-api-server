import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import OpenAI from 'openai';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';
import { CreateQuizDto } from './dto/create-quiz.dto';
import {
  QuizDifficulty,
  QuizQuestion,
  QuizResponse,
} from './types/quiz-question.interface';
import { QdrantService, BibleSearchResult } from './qdrant.service';

interface QuizTemplate {
  id: string;
  reference: string;
  summary: string;
  prompt: string;
  correctAnswer: string;
  distractors: string[];
  keywords: string[];
}

const QUIZ_TEMPLATES: QuizTemplate[] = [
  {
    id: 'love-patience',
    reference: '고전 13:4-5',
    summary:
      '사랑은 오래 참고 친절하며, 질투나 자기 유익을 구하지 않는 태도를 강조한다.',
    prompt: '바울이 설명한 사랑의 태도에 대한 묘사',
    correctAnswer: '사랑은 오래 참고 친절하며 무례히 행하지 않는다.',
    distractors: [
      '사랑은 감정이 식으면 쉽게 포기해도 된다.',
      '사랑은 자신의 유익을 최우선으로 삼는다.',
      '사랑은 상대를 변화시키려고 강요해야 한다.',
    ],
    keywords: ['사랑', '우정', '헌신'],
  },
  {
    id: 'companionship',
    reference: '전 4:9-10',
    summary:
      '두 사람이 함께할 때 서로를 일으켜 세울 수 있으니 동역의 힘을 강조한다.',
    prompt: '서로를 일으켜 세워 주는 동행의 가치에 대한 교훈',
    correctAnswer: '동행은 넘어질 때 서로를 붙잡아 다시 일으켜 세운다.',
    distractors: [
      '혼자 있는 것이 가장 안전하다.',
      '동역은 자신의 수고를 빼앗긴다고 경계해야 한다.',
      '친구는 실수할 때 거리를 둬야 한다.',
    ],
    keywords: ['우정', '동역', '연합'],
  },
  {
    id: 'sacrifice',
    reference: '요 15:13',
    summary:
      '친구를 위해 목숨을 내어놓을 때처럼 희생적인 사랑이 가장 큰 사랑임을 강조한다.',
    prompt: '희생을 통한 우정과 사랑의 깊이에 대한 가르침',
    correctAnswer: '친구를 위해 자신의 생명까지 내어놓는 사랑이 가장 크다.',
    distractors: [
      '희생은 연약함의 표시이므로 피해야 한다.',
      '사랑은 상대에게서 먼저 보호받는 것이다.',
      '우정은 감정이 식으면 끊어도 무방하다.',
    ],
    keywords: ['우정', '사랑', '희생'],
  },
  {
    id: 'encouragement',
    reference: '살전 5:11',
    summary: '서로를 위로하고 덕을 세우는 공동체 문화를 강조한다.',
    prompt: '공동체 안에서 서로를 세워 주는 방법에 대한 권면',
    correctAnswer: '서로를 격려하고 덕을 세우라.',
    distractors: [
      '경쟁심을 통해 공동체가 성장한다.',
      '개인의 성장은 공동체와 무관하다.',
      '위로는 약한 사람에게만 필요하다.',
    ],
    keywords: ['위로', '격려', '공동체'],
  },
  {
    id: 'humility',
    reference: '빌 2:3-4',
    summary:
      '겸손히 서로를 자신보다 낫게 여기고, 각자의 일을 넘어 다른 이의 일을 돌아보라고 요청한다.',
    prompt: '겸손과 배려를 실천하는 태도에 대한 촉구',
    correctAnswer:
      '겸손히 서로를 자신보다 낫게 여기며 다른 사람들의 유익을 돌보라.',
    distractors: [
      '자신의 일을 우선 해결한 뒤 다른 사람을 본다.',
      '겸손은 스스로의 가치를 낮추는 일이다.',
      '서로를 경쟁 상대로 삼아야 성장한다.',
    ],
    keywords: ['겸손', '배려', '공동체'],
  },
  {
    id: 'perseverance',
    reference: '약 1:12',
    summary:
      '시험을 견디는 자가 생명의 면류관을 받는다는 약속으로 인내의 가치를 일깨운다.',
    prompt: '믿음의 시험을 견디는 사람에게 주시는 약속',
    correctAnswer: '시험을 견디는 자가 생명의 면류관을 받는다.',
    distractors: [
      '시험은 하나님의 관심이 없다는 증거다.',
      '고난을 피하는 것이 믿음의 증거다.',
      '인내는 상황을 방치하는 태도다.',
    ],
    keywords: ['인내', '믿음', '소망'],
  },
  {
    id: 'hope',
    reference: '렘 29:11',
    summary:
      '하나님의 평안과 미래에 대한 선한 계획을 신뢰하라는 소망의 메시지다.',
    prompt: '하나님의 선한 계획을 신뢰하는 태도에 대한 메시지',
    correctAnswer: '하나님은 재앙이 아닌 평안을 주며 미래와 희망을 예비하신다.',
    distractors: [
      '하나님의 계획은 인간이 통제할 수 있다.',
      '소망은 현재의 성과에만 달려 있다.',
      '하나님은 종종 선한 계획을 숨기신다.',
    ],
    keywords: ['소망', '미래', '신뢰'],
  },
  {
    id: 'trust',
    reference: '잠 3:5-6',
    summary:
      '자신의 명철을 의지하지 말고 범사에 하나님을 인정하면 그분이 인도하신다는 약속이다.',
    prompt: '하나님을 신뢰하며 길을 맡기는 삶의 태도',
    correctAnswer: '자기 명철을 의지하지 말고 범사에 하나님을 인정하라.',
    distractors: [
      '지혜는 철저히 자신의 경험에서만 얻어진다.',
      '하나님은 중요한 결정에만 관여하신다.',
      '믿음은 감정에 따라 움직인다.',
    ],
    keywords: ['믿음', '신뢰', '인도'],
  },
];

type LlmQuizQuestionPayload = {
  id?: unknown;
  prompt?: unknown;
  question?: unknown;
  options?: unknown;
  answer?: unknown;
  correctAnswer?: unknown;
  explanation?: unknown;
  reference?: unknown;
};

type LlmQuizPayload = {
  topic?: unknown;
  difficulty?: unknown;
  count?: unknown;
  questions: LlmQuizQuestionPayload[];
};

@Injectable()
export class QuizService {
  private readonly logger = new Logger(QuizService.name);
  private readonly openai?: OpenAI;
  private readonly openaiModel?: string;

  constructor(
    private readonly configService: ConfigService,
    private readonly qdrantService: QdrantService,
  ) {
    const apiKey = this.configService.get<string>('OPENAI_API_KEY');
    const model =
      this.configService.get<string>('OPENAI_MODEL') ?? 'gpt-4o-mini';

    if (apiKey) {
      this.openai = new OpenAI({ apiKey });
      this.openaiModel = model;
      this.logger.log(
        `ChatGPT integration enabled with model "${this.openaiModel}"`,
      );
    } else {
      this.logger.warn(
        'OPENAI_API_KEY is missing. Falling back to template-based quiz generation.',
      );
    }
  }

  async generateQuiz(dto: CreateQuizDto): Promise<QuizResponse> {
    const { topic, count, difficulty } = dto;
    const normalizedTopic = topic.trim();
    const effectiveTopic =
      normalizedTopic.length > 0 ? normalizedTopic : '전체 성경';

    const llmResult = await this.generateQuizViaChatGPT(
      effectiveTopic,
      count,
      difficulty,
    );
    if (llmResult) {
      return llmResult;
    }

    return this.generateFallbackQuiz(effectiveTopic, count, difficulty);
  }

  private async generateQuizViaChatGPT(
    topic: string,
    count: number,
    difficulty: QuizDifficulty,
  ): Promise<QuizResponse | null> {
    if (!this.openai || !this.openaiModel) {
      return null;
    }

    const context = await this.buildBibleContext(topic);
    const messages = this.buildMessages(topic, count, difficulty, context);

    try {
      const temperature = this.resolveTemperature(difficulty);
      const baseParams: OpenAI.Chat.Completions.ChatCompletionCreateParams = {
        model: this.openaiModel,
        messages,
        response_format: { type: 'json_object' },
      };

      // 일부 최신 모델(gpt-5 계열 등)은 temperature 커스텀을 지원하지 않고
      // 기본값(1)만 허용하므로, 그런 모델에서는 temperature를 보내지 않는다.
      const params = this.shouldUseCustomTemperature(this.openaiModel)
        ? { ...baseParams, temperature }
        : baseParams;

      const completion = await this.openai.chat.completions.create(params);

      const content = completion.choices[0]?.message?.content;
      if (!content) {
        this.logger.warn('ChatGPT 응답에서 내용을 찾을 수 없습니다.');
        return null;
      }

      const parsed = this.parseChatGptResponse(content, {
        topic,
        count,
        difficulty,
      });
      if (!parsed) {
        this.logger.warn(
          'ChatGPT 응답을 파싱하지 못해 템플릿 모드로 전환합니다.',
        );
      }
      return parsed;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(
        'ChatGPT 호출 실패, 템플릿 모드로 전환합니다.',
        message,
      );
      return null;
    }
  }

  private buildMessages(
    topic: string,
    count: number,
    difficulty: QuizDifficulty,
    context: BibleSearchResult[] | null,
  ): ChatCompletionMessageParam[] {
    const safeTopic = topic || '임의 주제';
    const systemPrompt =
      '당신은 한국어로 성경 기반 객관식 퀴즈를 만드는 교사입니다. ' +
      '각 문제는 4지선다로 구성하고, 보기와 정답, 근거가 명확해야 합니다. ' +
      '정답은 보기 중 하나여야 하며, 모든 설명과 문장은 자연스러운 한국어를 사용하십시오. ' +
      '가능하다면 제공된 성경 본문 컨텍스트를 최대한 활용하여 문제를 출제하십시오.';

    const schemaDescription = `
다음 JSON 스키마를 반드시 따르세요:
{
  "topic": "요청 주제 그대로",
  "difficulty": "easy | medium | hard 중 하나",
  "count": ${count},
  "questions": [
    {
      "id": "문제 식별자 (선택)",
      "prompt": "문제 질문",
      "options": ["보기1", "보기2", "보기3", "보기4"],
      "answer": "options 중 정답",
      "explanation": "정답 해설 (1~2문장)",
      "reference": "성경 책 및 장절 예: 고전 13:4"
    }
  ]
}
`;

    const contextBlock =
      context && context.length
        ? `성경 본문 컨텍스트:
${context
  .map(
    (c, idx) =>
      `[${idx + 1}] (${c.bookFile} ${c.startLine}-${c.endLine})\n${c.text}`,
  )
  .join('\n\n')}`
        : '성경 본문 컨텍스트는 제공되지 않았습니다. 가능한 한 일반적인 성경 지식을 사용하되, 과도한 추측은 피하세요.';

    const userPrompt = `
주제: ${safeTopic}
난이도: ${difficulty}
문제 수: ${count}
요청: 위 스키마 그대로 JSON만 출력하고, 코드블록이나 추가 설명은 넣지 마세요.

${contextBlock}
`;

    return [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: schemaDescription },
      { role: 'user', content: userPrompt },
    ];
  }

  private async buildBibleContext(
    topic: string,
  ): Promise<BibleSearchResult[] | null> {
    // 1차 시도: LangChain + Qdrant 벡터스토어 사용
    try {
      const langchainResults = await this.buildBibleContextWithLangChain(topic);
      if (langchainResults && langchainResults.length > 0) {
        return langchainResults;
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(
        `LangChain 기반 컨텍스트 빌드 실패: ${message} (기존 방식으로 재시도합니다.)`,
      );
    }

    // 2차 시도: 기존 OpenAI 임베딩 + QdrantService 방식 (fallback)
    if (!this.qdrantService.isEnabled || !this.openai) {
      return null;
    }

    try {
      const embedding = await this.openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: topic || '일반적인 성경 퀴즈 주제',
      });

      const vector = embedding.data[0]?.embedding;
      if (!vector) {
        this.logger.warn('임베딩 생성에 실패했습니다. (context 없음)');
        return null;
      }

      const results = await this.qdrantService.searchBibleChunks(vector, 8);
      if (!results.length) {
        this.logger.log(
          `Qdrant에서 "${topic}" 주제에 대한 관련 본문을 찾지 못했습니다.`,
        );
        return null;
      }

      const preview = results.slice(0, 5);
      const logBody = preview
        .map(
          (r, idx) =>
            `[${idx + 1}] score=${r.score.toFixed(3)} ` +
            `(${r.bookFile} ${r.startLine}-${r.endLine})\n${r.text}`,
        )
        .join('\n\n');

      this.logger.debug(
        `Qdrant 검색 결과 ${results.length}개 수신 (상위 ${
          preview.length
        }개 미리보기):\n${logBody}`,
      );

      return results;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Qdrant 컨텍스트 빌드 실패: ${message}`);
      return null;
    }
  }

  // LangChain + Qdrant 벡터스토어를 사용해 성경 컨텍스트를 검색한다.
  // 문제가 생기면 호출 측에서 기존 방식으로 자동 fallback 된다.
  private async buildBibleContextWithLangChain(
    topic: string,
  ): Promise<BibleSearchResult[] | null> {
    const openaiApiKey = this.configService.get<string>('OPENAI_API_KEY');
    const qdrantUrl = this.configService.get<string>('QDRANT_URL');
    const qdrantApiKey = this.configService.get<string>('QDRANT_API_KEY');
    const collectionName =
      this.configService.get<string>('QDRANT_COLLECTION') ??
      'bible_embeddings';

    if (!openaiApiKey || !qdrantUrl || !qdrantApiKey) {
      this.logger.debug(
        'LangChain 컨텍스트 빌드에 필요한 환경변수(OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY)가 부족하여 기존 방식으로 진행합니다.',
      );
      return null;
    }

    // ESM 전용 패키지이므로 dynamic import 사용
    const [{ QdrantVectorStore }, { OpenAIEmbeddings }, { QdrantClient }] =
      await Promise.all([
        import('@langchain/community/vectorstores/qdrant'),
        import('@langchain/openai'),
        import('@qdrant/js-client-rest'),
      ]);

    const client = new QdrantClient({ url: qdrantUrl, apiKey: qdrantApiKey });
    const embeddings = new OpenAIEmbeddings({
      apiKey: openaiApiKey,
      model: 'text-embedding-3-small',
    });

    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        client,
        collectionName,
        // seed 시 payload.text에 본문을 넣었으므로, 해당 키를 pageContent로 사용
        contentPayloadKey: 'text',
      },
    );

    const query = topic || '일반적인 성경 퀴즈 주제';
    this.logger.debug(
      `LangChain Qdrant 검색 시작: "${query}" (collection=${collectionName})`,
    );

    const docsWithScore = await vectorStore.similaritySearchWithScore(
      query,
      8,
    );

    if (!docsWithScore.length) {
      this.logger.log(
        `LangChain/Qdrant에서 "${topic}" 주제에 대한 관련 본문을 찾지 못했습니다.`,
      );
      return null;
    }

    const results: BibleSearchResult[] = docsWithScore.map(
      ([doc, score], idx) => {
        const metadata = (doc.metadata ?? {}) as Record<string, unknown>;
        const bookFile =
          typeof metadata.bookFile === 'string'
            ? metadata.bookFile
            : 'unknown';

        return {
          id: String(metadata.originalId ?? idx),
          text: doc.pageContent,
          bookFile,
          chunkIndex:
            typeof metadata.chunkIndex === 'number' ? metadata.chunkIndex : -1,
          startLine:
            typeof metadata.startLine === 'number' ? metadata.startLine : -1,
          endLine:
            typeof metadata.endLine === 'number' ? metadata.endLine : -1,
          score,
        };
      },
    );

    const preview = results.slice(0, 5);
    const logBody = preview
      .map(
        (r, idx) =>
          `[LC ${idx + 1}] score=${r.score.toFixed(3)} ` +
          `(${r.bookFile} ${r.startLine}-${r.endLine})\n${r.text}`,
      )
      .join('\n\n');

    this.logger.debug(
      `LangChain Qdrant 검색 결과 ${results.length}개 수신 (상위 ${
        preview.length
      }개 미리보기):\n${logBody}`,
    );

    return results;
  }

  private parseChatGptResponse(
    content: string,
    meta: { topic: string; count: number; difficulty: QuizDifficulty },
  ): QuizResponse | null {
    try {
      const cleaned = content.replace(/```json|```/gi, '').trim();
      const parsedUnknown = JSON.parse(cleaned) as unknown;
      if (!this.isLlmQuizPayload(parsedUnknown)) {
        return null;
      }
      const parsed = parsedUnknown;

      const questions = parsed.questions
        .slice(0, meta.count)
        .map((question, index) => this.normalizeQuestion(question, index, meta))
        .filter((question): question is QuizQuestion => Boolean(question));

      if (!questions.length) {
        return null;
      }

      return {
        topic:
          this.coerceString(parsed.topic) ?? meta.topic ?? '사용자 정의 주제',
        difficulty: this.coerceDifficulty(parsed.difficulty) ?? meta.difficulty,
        count:
          typeof parsed.count === 'number' && parsed.count > 0
            ? Math.min(parsed.count, questions.length)
            : questions.length,
        questions,
        generatedAt: new Date().toISOString(),
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`ChatGPT JSON 파싱 실패: ${message}`);
      return null;
    }
  }

  private normalizeQuestion(
    raw: LlmQuizQuestionPayload,
    index: number,
    meta: { topic: string; difficulty: QuizDifficulty },
  ): QuizQuestion | null {
    const prompt =
      this.coerceString(raw.prompt) ?? this.coerceString(raw.question) ?? '';
    const answer = this.extractAnswer(raw);
    const options = this.sanitizeOptions(raw.options, answer);

    if (!prompt || !answer || options.length < 2) {
      return null;
    }

    const reference = this.coerceString(raw.reference) ?? '참고 구절 미상';
    const explanation =
      this.coerceString(raw.explanation) ??
      '정답의 근거를 성경 구절 맥락에서 설명해주세요.';

    return {
      id: this.coerceString(raw.id) ?? `chatgpt-${index + 1}`,
      prompt,
      options,
      answer,
      difficulty: meta.difficulty,
      reference,
      explanation,
    };
  }

  private extractAnswer(raw: LlmQuizQuestionPayload): string | null {
    const answer = this.coerceString(raw.answer);
    if (answer) {
      return answer;
    }
    const correctAnswer = this.coerceString(raw.correctAnswer);
    return correctAnswer ?? null;
  }

  private sanitizeOptions(
    optionsInput: unknown,
    answer: string | null,
  ): string[] {
    const baseOptions = Array.isArray(optionsInput)
      ? optionsInput
          .map((option) => (typeof option === 'string' ? option.trim() : ''))
          .filter((option) => option.length > 0)
      : [];

    if (answer && !baseOptions.includes(answer)) {
      baseOptions.push(answer);
    }

    const deduped = Array.from(new Set(baseOptions));
    return deduped.slice(0, 4);
  }

  private coerceString(value: unknown): string | null {
    return typeof value === 'string' && value.trim().length > 0
      ? value.trim()
      : null;
  }

  private coerceDifficulty(value: unknown): QuizDifficulty | null {
    return this.isDifficulty(value) ? value : null;
  }

  private isDifficulty(value: unknown): value is QuizDifficulty {
    return value === 'easy' || value === 'medium' || value === 'hard';
  }

  private isLlmQuizPayload(value: unknown): value is LlmQuizPayload {
    if (typeof value !== 'object' || value === null) {
      return false;
    }

    const candidate = value as Record<string, unknown>;
    if (!Array.isArray(candidate.questions)) {
      return false;
    }

    return candidate.questions.every((item) =>
      this.isLlmQuizQuestionPayload(item),
    );
  }

  private isLlmQuizQuestionPayload(
    value: unknown,
  ): value is LlmQuizQuestionPayload {
    return typeof value === 'object' && value !== null;
  }

  private resolveTemperature(difficulty: QuizDifficulty): number {
    switch (difficulty) {
      case 'easy':
        return 0.4;
      case 'hard':
        return 0.8;
      default:
        return 0.6;
    }
  }

  /**
   * 일부 최신 모델(예: gpt-5-mini 계열)은 temperature를 고정값(1)으로만 허용한다.
   * 이런 모델에서는 temperature를 아예 보내지 않고, OpenAI 기본값을 사용한다.
   */
  private shouldUseCustomTemperature(model: string | undefined): boolean {
    if (!model) return true;
    const lower = model.toLowerCase();
    // gpt-5 계열은 temperature를 고정값만 허용하는 경우가 있어 제외
    if (lower.startsWith('gpt-5')) {
      return false;
    }
    return true;
  }

  private generateFallbackQuiz(
    topic: string,
    count: number,
    difficulty: QuizDifficulty,
  ): QuizResponse {
    const pool = this.pickTemplates(topic);
    const questions = this.composeQuestions(pool, topic, count, difficulty);

    return {
      topic,
      difficulty,
      count: questions.length,
      questions,
      generatedAt: new Date().toISOString(),
    };
  }

  private pickTemplates(topic: string): QuizTemplate[] {
    const lowered = topic.toLowerCase();
    const matches = QUIZ_TEMPLATES.filter((template) =>
      template.keywords.some(
        (keyword) =>
          keyword.toLowerCase().includes(lowered) ||
          lowered.includes(keyword.toLowerCase()),
      ),
    );

    return matches.length ? matches : QUIZ_TEMPLATES;
  }

  private composeQuestions(
    templates: QuizTemplate[],
    topic: string,
    count: number,
    difficulty: QuizDifficulty,
  ): QuizQuestion[] {
    const expanded: QuizQuestion[] = [];
    let index = 0;

    while (expanded.length < count) {
      const template = templates[index % templates.length];
      expanded.push(
        this.buildQuestion(template, topic, difficulty, expanded.length),
      );
      index++;
    }

    return expanded;
  }

  private buildQuestion(
    template: QuizTemplate,
    topic: string,
    difficulty: QuizDifficulty,
    order: number,
  ): QuizQuestion {
    const difficultyLabel = this.getDifficultyLabel(difficulty);
    const focus = topic || template.keywords[0];
    const prompt = `[${difficultyLabel}] ${template.prompt} — “${focus}” 관점에서 가장 알맞은 답을 고르세요.`;
    const options = this.composeOptions(template);

    return {
      id: `${template.id}-${order}`,
      prompt,
      options,
      answer: template.correctAnswer,
      difficulty,
      reference: template.reference,
      explanation: template.summary,
    };
  }

  private composeOptions(template: QuizTemplate): string[] {
    const options = this.shuffle([
      template.correctAnswer,
      ...template.distractors,
    ]);

    return options.slice(0, 4);
  }

  private getDifficultyLabel(difficulty: QuizDifficulty): string {
    switch (difficulty) {
      case 'easy':
        return '기초';
      case 'hard':
        return '심화';
      default:
        return '표준';
    }
  }

  private shuffle<T>(input: T[]): T[] {
    const array = [...input];
    for (let i = array.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }
}
