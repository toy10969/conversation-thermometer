import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(express.json({ limit: '20mb' }));
app.use(express.static(join(__dirname, 'public')));

const API_KEY = process.env.ANTHROPIC_API_KEY;
if (!API_KEY) {
  console.error('ERROR: ANTHROPIC_API_KEY environment variable is not set.');
  process.exit(1);
}
const client = new Anthropic({ apiKey: API_KEY });

function buildContext(options = {}) {
  const { partnerProfile, pastSummary, isPersonalized } = options;
  let ctx = '';
  if (partnerProfile) ctx += `【相手のプロフィール・特徴】\n${partnerProfile}\n\n`;
  if (pastSummary)    ctx += `【過去の傾向・フィードバック履歴】\n${pastSummary}\n\n`;
  return ctx;
}

const JSON_SCHEMA = `{
  "temperature": 0から100の整数（0=冷たい対立、50=中立、100=情熱的・温かい）,
  "mood": "2〜5文字の日本語でムード表現",
  "emotions": [{"name": "感情名", "intensity": 0から100の整数, "emoji": "絵文字"}],
  "positive_points": ["良い点1", "良い点2"],
  "concern_points": ["懸念点1"],
  "summary": "2〜3文での感情分析サマリー",
  "advice": "関係改善や継続のための具体的なアドバイス",
  "reply_suggestions": ["返信案1（相手の温度感に合ったトーン）", "返信案2（別アプローチ）", "返信案3（さらに別角度）"],
  "trend_memo": "この会話の傾向を次回分析に活かすための1〜2文メモ（過去メモがあれば統合更新）"
}`;

function buildPrompt(conversation, options = {}) {
  const { isPersonalized } = options;
  const ctx = buildContext(options);
  const personNote = isPersonalized ? '※ 上記のプロフィールと過去傾向を十分考慮してパーソナライズ分析してください。\n' : '';
  return `${ctx}以下の会話を分析して、感情温度と詳細な感情分析をJSON形式で返してください。
${personNote}
会話:
${conversation}

以下のJSON形式で返してください（コードブロックなし、純粋なJSONのみ）:
${JSON_SCHEMA}

emotionsは最も強い感情を3〜5個。reply_suggestionsは実際に使える返信文を3パターン提示。`;
}

function buildImagePrompt(options = {}) {
  const { isPersonalized } = options;
  const ctx = buildContext(options);
  const personNote = isPersonalized ? '※ 上記のプロフィールと過去傾向を十分考慮してパーソナライズ分析してください。\n' : '';
  return `${ctx}この画像はLINEのトーク画面のスクリーンショットです。会話内容を正確に読み取り、分析をJSON形式で返してください。
${personNote}
以下のJSON形式で返してください（コードブロックなし、純粋なJSONのみ）:
{
  "extracted_text": "画像から読み取った会話テキスト（発言者名: メッセージ の形式で改行区切り）",
  ${JSON_SCHEMA.slice(1)}

emotionsは最も強い感情を3〜5個。reply_suggestionsは実際に使える返信文を3パターン提示。
画像から会話が読み取れない場合は extracted_text に「会話が読み取れませんでした」と記載し、他フィールドは推測で埋めないでください。`;
}

function parseAnalysis(rawText) {
  try { return JSON.parse(rawText); }
  catch {
    const match = rawText.match(/\{[\s\S]*\}/);
    if (!match) throw new Error('JSONの解析に失敗しました。');
    return JSON.parse(match[0]);
  }
}

function handleApiError(err, res) {
  if (err instanceof Anthropic.AuthenticationError)
    return res.status(401).json({ error: 'APIキーが無効です。' });
  if (err instanceof Anthropic.RateLimitError)
    return res.status(429).json({ error: 'APIのレート制限に達しました。しばらく待ってから再試行してください。' });
  if (err instanceof Anthropic.BadRequestError)
    return res.status(400).json({ error: 'リクエストが無効です: ' + err.message });
  console.error('API error:', err);
  res.status(500).json({ error: 'サーバーエラーが発生しました: ' + err.message });
}

app.post('/api/analyze', async (req, res) => {
  const { conversation, partnerProfile, pastSummary, isPersonalized } = req.body;

  if (!conversation || typeof conversation !== 'string')
    return res.status(400).json({ error: '会話テキストが必要です。' });
  const trimmed = conversation.trim();
  if (trimmed.length < 5)
    return res.status(400).json({ error: '会話が短すぎます（最低5文字）。' });
  if (trimmed.length > 20000)
    return res.status(400).json({ error: '会話が長すぎます（最大20000文字）。' });

  try {
    const message = await client.messages.create({
      model: 'claude-opus-4-7',
      max_tokens: 1500,
      messages: [{ role: 'user', content: buildPrompt(trimmed, { partnerProfile, pastSummary, isPersonalized }) }],
    });
    const analysis = parseAnalysis(message.content[0].text.trim());
    analysis.temperature = Math.max(0, Math.min(100, Math.round(Number(analysis.temperature))));
    res.json(analysis);
  } catch (err) { handleApiError(err, res); }
});

app.post('/api/analyze-image', async (req, res) => {
  const { imageData, mediaType, partnerProfile, pastSummary, isPersonalized } = req.body;

  if (!imageData || typeof imageData !== 'string')
    return res.status(400).json({ error: '画像データが必要です。' });
  const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
  if (!validTypes.includes(mediaType))
    return res.status(400).json({ error: '対応形式はJPEG、PNG、GIF、WebPのみです。' });
  if (Math.ceil(imageData.length * 0.75) > 10 * 1024 * 1024)
    return res.status(400).json({ error: '画像サイズは10MB以下にしてください。' });

  try {
    const message = await client.messages.create({
      model: 'claude-opus-4-7',
      max_tokens: 2500,
      messages: [{
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: mediaType, data: imageData } },
          { type: 'text', text: buildImagePrompt({ partnerProfile, pastSummary, isPersonalized }) },
        ],
      }],
    });
    const analysis = parseAnalysis(message.content[0].text.trim());
    analysis.temperature = Math.max(0, Math.min(100, Math.round(Number(analysis.temperature))));
    res.json(analysis);
  } catch (err) { handleApiError(err, res); }
});

app.listen(3000, () => console.log('サーバー起動: http://localhost:3000'));
