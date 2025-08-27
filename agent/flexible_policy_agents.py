from typing import Dict, List, Optional, TypedDict, Any
from datetime import datetime
import re
import asyncio

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

class SectionRepo:
    async def get_sections(self, project_id: str, step_key: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

class CrudSectionRepo(SectionRepo):
    def __init__(self, crud_module):
        self.crud = crud_module

    async def get_sections(self, project_id: str, step_key: str) -> List[Dict[str, Any]]:
        return await self.crud.fetch_project_step_sections(project_id, step_key)

class ChatRepo:
    async def get_recent_messages(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError

class CrudChatRepo(ChatRepo):
    def __init__(self, crud_module):
        self.crud = crud_module

    async def get_recent_messages(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return await self.crud.get_recent_chat_messages(session_id, limit)

STEP_ORDER = ["analysis", "objective", "concept", "plan", "proposal"]

STEP_NAME_ALIASES = {
    "analysis": ["現状分析", "課題整理"],
    "objective": ["目的整理"],
    "concept": ["コンセプト"],
    "plan": ["施策検討", "施策立案"],
    "proposal": ["提案資料", "資料作成"],
}

STEP_SECTIONS = {
    "analysis": [
        {"key": "problem_evidence",     "label": "課題と裏付け（定量・定性）"},
        {"key": "background_structure", "label": "課題の背景にある構造（制度・市場など）"},
        {"key": "priority_reason",      "label": "解決すべき課題の優先度と理由"},
    ],
    "objective": [
        {"key": "final_goal",  "label": "最終的に達成したいゴール"},
        {"key": "kpi_target",  "label": "KPI・目標値（いつまでに・どれだけ）"},
        {"key": "constraints", "label": "前提条件・制約（予算・人員・期間など）"},
    ],
    "concept": [
        {"key": "policy_direction", "label": "基本方針（価値/誰に/どう）"},
        {"key": "evidence",         "label": "方針の根拠・示唆"},
        {"key": "risks_actions",    "label": "主要リスクと打ち手"},
    ],
    "plan": [
        {"key": "main_actions", "label": "主な施策（3〜5）と狙い"},
        {"key": "org_schedule", "label": "体制・役割分担・スケジュール"},
        {"key": "cost_effect",  "label": "概算コスト・効果見込み"},
    ],
    "proposal": [
        {"key": "exec_summary",    "label": "提案サマリー（背景→課題→解決→効果→体制）"},
        {"key": "decision_points", "label": "意思決定者の関心（費用対効果・リスク・責任）"},
        {"key": "next_actions",    "label": "次のアクション（承認/説明/PoC）"},
    ],
}

STEP_CONFIG = {
    "analysis": {
        "title": "現状分析・課題整理",
        "goals": [
            "課題と裏付け（定量/定性）が示されている",
            "背景にある構造（制度・市場・慣行など）が仮説化されている",
            "優先度と着手順が説明できる",
        ],
        "persona": (
            "課題整理と現状分析の専門家。自然体の敬体で伴走。断定しすぎない。"
            " このフェーズでは施策/解決案の詳細には入らず、現状の把握と課題の特定・裏付けに専念する。"
            " 解決策に言及する場合は“仮に”レベルでとどめ、詳細検討は後工程へ明示的に保留する。"
        ),
        "stay_in_lane": True,
    },
    "objective": {
        "title": "目的整理",
        "goals": [
            "最終ゴール（誰がどう変わるか）が具体",
            "KPI/目標値（いつまでに・どれだけ）が根拠付きで定義",
            "前提条件・制約が列挙され、柔軟性が検討済み",
        ],
        "persona": "目的とKPI設定の専門家。まず“誰がどう変わるか”の言語化を支援。",
        "stay_in_lane": True,
    },
    "concept": {
        "title": "コンセプト策定",
        "goals": [
            "どんな価値を誰にどう届けるかが明確",
            "根拠（調査/事例/専門家）がある",
            "主要リスクと軽減策（代替案・小規模実験）がある",
        ],
        "persona": "政策コンセプトとリスク評価の専門家。反証も扱う。",
        "stay_in_lane": True,
    },
    "plan": {
        "title": "施策立案",
        "goals": [
            "主な施策（3〜5）が目的と整合",
            "体制/役割/スケジュールが現実的",
            "概算コストと効果見込み、評価方法がある",
        ],
        "persona": "実行計画とコスト/リソース管理の専門家。小さく速く検証。",
        "stay_in_lane": True,
    },
    "proposal": {
        "title": "資料作成",
        "goals": [
            "背景→課題→解決→効果→体制の物語が滑らか",
            "意思決定者の関心（費用対効果・リスク・責任）をカバー",
            "承認後の次アクション（PoC等）が明示",
        ],
        "persona": "提案資料と意思決定支援の専門家。“要は何か”を短く。",
        "stay_in_lane": True,
    },
}

EXPLICIT_REFERENCE = True

MIN_TURNS_FOR_STRUCTURE = 3
MIN_TURNS_FOR_CAPTURE = 4
MIN_TURNS_FOR_BRIDGE = 6
MIN_CHARS_FOR_CAPTURE = 120
MIN_RECENT_CHARS = 200

FACT_GUIDANCE = {
    "analysis": [
        "現場オペレーター（例：受発注・窓口・経理補助など）",
        "現場のリーダー/班長・係長クラス（業務の詰まり所の把握者）",
        "情報システム/総務・庶務（帳票/ツール/権限の実務を知る人）",
        "主要な取引先/委託先の担当（FAX/紙依存の理由を持つ相手側）",
        "商工会・業界団体/地域金融機関（横断的な事情に詳しい組織）",
        "処理時間・処理件数・エラー/差し戻し件数（業務ログ/紙台帳）",
        "紙・FAX比率、再入力件数、二重入力の発生頻度",
        "在庫回転・受発注リードタイム・納期遅延率",
        "問い合わせチャネル別件数（電話/メール/紙/オンライン）",
        "人員構成・経験年数・研修受講状況（基礎スキル把握）",
    ],
    "objective": [
        "KPIのオーナー/管理者（部門長・経営企画）、データ管理者",
        "評価を受ける現場側（KPIが業務に与える負荷を確認）",
        "KPI候補：紙書類比率、再入力率、処理リードタイム、エラー率、教育受講率、システム定着率（MAU/週次利用率）",
        "計測可能性：取得経路（業務ログ/在庫・会計/CSツール/勤怠）と更新頻度・欠測",
        "ベンチマーク：類似施策の達成水準、業界統計の中央値",
    ],
    "concept": [
        "想定ターゲットの利用者/中小企業の担当者（価値仮説への共感度）",
        "専門家/先行事例の担当者（成功・失敗要因の裏取り）",
        "ユーザー課題の頻度と深刻度、代替手段の満足度/費用",
        "国内外の先行事例（導入条件・スイッチングコスト・リスク）",
    ],
    "plan": [
        "パイロットに協力可能な現場/地域・IT担当",
        "外部ベンダ/ツール提供者（見積・PoC条件）",
        "パイロット設計：対象セグメント、期間、成功指標（例：処理時間/再入力率/定着率）",
        "概算コスト（初期/運用）と想定効果（時間削減・事故削減・満足度）",
    ],
    "proposal": [
        "意思決定者/財務、リスク管理、現場代表（関心点の擦り合わせ）",
        "費用対効果（投資回収期間/ROI）、主要リスクの発生確率と影響、体制稼働（FTE）",
    ],
}

_SOLUTION_WORDS = re.compile(
    r"(施策|ソリューション|導入|実装|PoC|ロードマップ|KPI改善案|予算配分|体制案|稟議|スケジュール|ツール選定|システム|ツール|IT化|RPA|SaaS|パッケージ)"
)

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _two_bullets(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "・（検討中）"
    outs = []
    for ln in lines[:2]:
        outs.append(ln if ln.startswith(("・","-","*","—")) else f"・{ln}")
    return "\n".join(outs)

def _bullets_any(text: str, limit=3) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    bulls = [ln for ln in lines if ln.startswith(("・","-","*","—"))]
    if not bulls:
        bulls = [f"・{ln}" for ln in lines if not ln.startswith(("#","##"))]
    return "\n".join(bulls[:limit]) if bulls else ""

def _score(known: str, gaps: str) -> int:
    sc = 30
    sc += 30 if len(known) >= 120 else 15
    sc += 40 if (0 < len(gaps) <= 120) else 20
    return max(0, min(100, sc))

def _has_solution_leak(text: str) -> bool:
    return bool(_SOLUTION_WORDS.search(text or ""))

def _scope_guard(step: str, text: str) -> str:
    if step != "analysis":
        return text
    if not _has_solution_leak(text):
        return text
    softened = re.sub(r"(導入|実装|施策|ソリューション|システム|ツール)", r"仮に\1", text)
    note = "（※ 今は現状と課題の裏付けに専念し、解決策の細部は後工程で扱います）"
    if note not in softened:
        softened += "\n" + note
    return softened

class FlexiblePolicyState(TypedDict):
    session_id: str
    project_id: Optional[str]
    last_updated_step: Optional[str]
    step_timestamps: Dict[str, str]
    conversation_history: List[Dict[str, str]]
    ask_cooldown: int
    last_move: str
    step_completion: Dict[str, int]
    last_result: Optional[str]

class FlexiblePolicyAgentSystem:

    def __init__(self, chat_model: ChatOpenAI, section_repo: Optional[SectionRepo]=None, chat_repo: Optional[ChatRepo]=None):
        self.chat = chat_model
        self.section_repo = section_repo
        self.chat_repo = chat_repo
        self.state_storage: Dict[str, FlexiblePolicyState] = {}

    def _get_state(self, session_id: str, project_id: Optional[str]) -> FlexiblePolicyState:
        st = self.state_storage.get(session_id)
        if not st:
            st = FlexiblePolicyState(
                session_id=session_id,
                project_id=project_id,
                last_updated_step=None,
                step_timestamps={},
                conversation_history=[],
                ask_cooldown=0,
                last_move="advise",
                step_completion={k: 0 for k in STEP_ORDER},
                last_result=None,
            )
            self.state_storage[session_id] = st
        if project_id and not st.get("project_id"):
            st["project_id"] = project_id
        return st

    def _turns(self, st: FlexiblePolicyState) -> int:
        return max(0, len(st["conversation_history"]) // 2)

    def _recent_user_chars(self, st: FlexiblePolicyState, last_n_turns: int) -> int:
        users = [m["content"] for m in st["conversation_history"] if m.get("role")=="user"]
        return sum(len(t or "") for t in users[-last_n_turns:])

    def _enough(self, st: FlexiblePolicyState, min_turns: int, min_chars: int) -> bool:
        return (self._turns(st) >= min_turns) and (self._recent_user_chars(st, min_turns) >= min_chars)

    def _log_exchange(self, st: FlexiblePolicyState, step: str, user_input: str, result: str):
        st["last_updated_step"] = step
        st["step_timestamps"][step] = datetime.now().isoformat()
        st["conversation_history"].append({"role":"user","content":user_input})
        st["conversation_history"].append({"role":"assistant","content":result})
        st["conversation_history"] = st["conversation_history"][-20:]
        st["last_result"] = result
        last_move = "ask" if result.strip().endswith(("？","?")) else "advise"
        st["last_move"] = last_move
        st["ask_cooldown"] = 2 if last_move=="ask" else max(0, st["ask_cooldown"]-1)

    async def _cross_context(self, st: FlexiblePolicyState) -> str:
        if not (self.section_repo and st.get("project_id")):
            return ""
        blocks = []
        for step in STEP_ORDER:
            rows = await self.section_repo.get_sections(st["project_id"], step)
            if not rows:
                continue
            ex = []
            for r in rows:
                txt = (r.get("content_text") or r.get("content") or "").strip()
                if not txt: continue
                ex.append(_clean(txt)[:140])
            if ex:
                blocks.append(f"◆{STEP_CONFIG[step]['title']}\n・" + "\n・".join(ex[:3]))
        return "\n\n".join(blocks)

    async def _step_context(self, st: FlexiblePolicyState, step: str) -> str:
        if not (self.section_repo and st.get("project_id")):
            return ""
        rows = await self.section_repo.get_sections(st["project_id"], step)
        if not rows: return ""
        bl = []
        for r in rows:
            txt = (r.get("content_text") or r.get("content") or "").strip()
            if not txt: continue
            label = r.get("label") or r.get("section_key","")
            bl.append(f"■{label}\n{txt[:280]}")
        return "\n\n".join(bl)

    async def _build_context(self, st: FlexiblePolicyState, step: str) -> str:
        chunks = []
        hist = st["conversation_history"][-8:]
        if hist:
            talk = "\n".join([f"【{m['role']}】{_clean(m['content'])}" for m in hist if m.get("content")])
            chunks.append("【直近の会話】\n" + talk)
        cross = await self._cross_context(st)
        if cross:
            chunks.append("【保存済みの横断前提】\n" + cross)
        now = await self._step_context(st, step)
        if now:
            chunks.append("【現在ステップの整理】\n" + now)
        return "\n\n".join(chunks)

    async def _chat_reply(self, step: str, user_input: str, st: FlexiblePolicyState) -> str:
        cfg = STEP_CONFIG[step]
        lane_note = " ※このステップの範囲から逸脱しないでください。" if cfg.get("stay_in_lane") else ""
        analysis_guard = (
            " 特にこの analysis（現状分析・課題整理）では、システムやツール等の導入・実装・施策の詳細には入らないでください。"
            " 要望があっても丁寧に保留し、現状の事実・課題・根拠（誰が/何が/どれくらい）に集中してください。"
        ) if step == "analysis" else ""

        sys = (
            f"あなたは{cfg['persona']}{lane_note}{analysis_guard}"
            " 出力は自然な会話文のみ（見出し・箇条書きは禁止）。"
            " 構成は『要約＋共感→示唆/観点を1つ→軽い確認 or 提案を1つ』で合計2〜5文。"
            " 連続質問は避け、クールダウン中は質問しない。"
            + (" 可能なら【保存済みの横断前提】または【現在ステップの整理】から該当事項を1点だけ参照し、"
            "整合しているか/差分がないかをひと言添えてください。矛盾が見つかる場合は、"
            "『後で確認』ではなく“小さく確認する提案”を短く添えてください。保存内容が無ければ参照は省略します。"
            if EXPLICIT_REFERENCE else "")
        )

        prompt = PromptTemplate(
            template=(
                "【ステップ】{title}\n"
                "【このステップのゴール】\n- " + "\n- ".join(cfg["goals"]) + "\n\n"
                "【文脈】\n{context}\n\n"
                "【ユーザーの発話】\n{user_input}\n\n"
                "会話文だけで返してください。"
            ),
            input_variables=["title","context","user_input"],
        )
        msgs = [
            SystemMessage(content=sys),
            HumanMessage(content=prompt.format(title=cfg["title"], context=await self._build_context(st, step), user_input=user_input)),
        ]
        out = await self.chat.ainvoke(msgs)
        return _scope_guard(step, out.content.strip())

    async def _mini_structure(self, step: str, user_input: str, st: FlexiblePolicyState) -> Dict[str,str]:
        cfg = STEP_CONFIG[step]
        guidance_lines = FACT_GUIDANCE.get(step, [])
        guidance_block = "・" + "\n・".join(guidance_lines) if guidance_lines else ""

        analysis_guard = (
            " 特にこの analysis では、システム/ツール導入などの解決策の詳細には入らず、"
            " 現状の事実と課題の裏付け（統計/業務ログ/証言/実地観察）に集中してください。"
        ) if step == "analysis" else ""

        tmpl = PromptTemplate(
            template=(
                "あなたは政策立案の壁打ちコーチです。短く要点だけ、箇条書き中心。"
                " このステップの範囲（スコープ）から逸脱しないでください。"
                + analysis_guard + "\n"
                "【ステップ】{title}\n"
                "【ゴール】\n- " + "\n- ".join(cfg["goals"]) + "\n\n"
                "【文脈】\n{context}\n\n"
                "【ユーザーの発話】\n{user_input}\n\n"
                "次の6ブロックをこの順で出力：\n"
                "1) TL;DR（1行）\n"
                "2) 今わかっていること（最大3点）\n"
                "3) 足りないこと（最大3点）\n"
                "4) 次の一歩（1つ/担当や期限は書かない/動詞で始める）\n"
                "5) 必要なファクト（裏付けに必要な客観情報/統計/ログ/文書/実地観察/誰の証言などを自由記述で具体）\n"
                "6) ファクト収集の提案（誰に何をどう聞くか、どのデータをどこからどう抽出するか/粒度/期間を自由記述で具体）\n"
                "参考カテゴリ（任意で使う。プレースホルダは禁止）：\n{guidance}\n"
                "禁止：{{who1}}/{{metric1}} 等のプレースホルダ表記、曖昧な『◯◯など』のみの回答。\n"
                + ("可能なら、いずれかのブロックに1行だけ『整合メモ：…』を付記して、"
                "【保存済みの横断前提】または【現在ステップの整理】との整合/差分を示してください。"
                "保存内容が無ければ付記は省略します。"
                if EXPLICIT_REFERENCE else "")
            ),
            input_variables=["title","context","user_input","guidance"],
        )
        msgs = [
            SystemMessage(content="出力は日本語。各ブロックは見出し＋本文（1〜3行）。このステップの範囲から逸脱しない。"),
            HumanMessage(content=tmpl.format(
                title=cfg["title"],
                context=await self._build_context(st, step),
                user_input=user_input,
                guidance=guidance_block
            )),
        ]
        raw = await self.chat.ainvoke(msgs)
        raw_content = _scope_guard(step, raw.content.strip())

        pats = {
            "tldr":  r"^\s*1\).*?$([\s\S]*?)(?=^\s*2\)|^\Z)",
            "known": r"^\s*2\).*?$([\s\S]*?)(?=^\s*3\)|^\Z)",
            "gaps":  r"^\s*3\).*?$([\s\S]*?)(?=^\s*4\)|^\Z)",
            "next":  r"^\s*4\).*?$([\s\S]*?)(?=^\s*5\)|^\Z)",
            "facts": r"^\s*5\).*?$([\s\S]*?)(?=^\s*6\)|^\Z)",
            "how":   r"^\s*6\).*?$([\s\S]*)\Z",
        }
        out: Dict[str, str] = {k: "" for k in pats}
        for k, pat in pats.items():
            m = re.search(pat, raw_content, flags=re.MULTILINE)
            if m:
                out[k] = m.group(1).strip()

        if not out["known"]:
            out["known"] = _bullets_any(raw_content, 3)
        if not out["gaps"]:
            out["gaps"] = _bullets_any(raw_content, 3)
        if not out["facts"]:
            out["facts"] = "関係者の証言・業務ログ・既存統計・実地観察など、裏付けファクトが未取得。"
        if not out["how"]:
            out["how"] = ("現場担当/管理職/情報システム/主要取引先に半構造化インタビュー。"
                          "業務ログ（処理時間/件数/エラー率）や在庫・受発注データを期間指定で抽出し、"
                          "チャネル別件数等を集計する。")
        qs = [ln.strip() for ln in raw_content.splitlines() if ln.strip().endswith(("？", "?"))]
        out["q"] = qs[-1] if qs else "いま決めるなら、最小の次の一歩は何にしますか？"

        if not out["next"]:
            out["next"] = self._synthesize_next(step, out)

        if not out["tldr"]:
            try:
                out["tldr"] = f"{out['known'].splitlines()[0].lstrip('・-*—').strip()} / {out['next'].splitlines()[0]}"
            except Exception:
                out["tldr"] = "要点を整理中。"

        return out
    
    def _synthesize_next(self, step: str, parsed: Dict[str, str]) -> str:
        def first_line(s: str) -> str:
            for ln in (s or "").splitlines():
                t = ln.lstrip("・-*—").strip()
                if t:
                    return t
            return ""

        f = first_line(parsed.get("facts", ""))
        h = first_line(parsed.get("how", ""))

        if f and h:
            return f"上位のファクト（例：{f}）を、提案手段（例：{h}）で小さく収集開始する。"
        if h:
            return f"提案手段（例：{h}）をスモールテストし、収集可能性と粒度を確認する。"
        if f:
            return f"最重要ファクト（例：{f}）の入手先と取得可否を関係者に確認する。"

        defaults = {
            "analysis":  "現場1部署を選び、業務フロー観察（半日）＋担当者ヒアリング（15分×2名）を実施する。",
            "objective": "最終ゴールの草案を1文で作り、関係者2名からフィードバックを得る。",
            "concept":   "価値提案の1文を作り、想定ターゲット2名に共感度を確認する。",
            "plan":      "施策候補を3つ列挙し、目的との因果と評価指標の対応表（ラフ）を作る。",
            "proposal":  "提案サマリー（3行）を作り、決裁者の関心点に沿って推敲する。",
        }
        return defaults.get(step, "最小の次の一歩を決める。")

    async def _capture_candidates(self, st: FlexiblePolicyState, step: str, parsed: Dict[str,str]) -> str:
        if not (self.section_repo and st.get("project_id")):
            return ""
        
        rows = await self.section_repo.get_sections(st["project_id"], step)
        existing = {r["section_key"]: (r.get("content_text") or r.get("content") or "") for r in rows}

        mapping = {
            "analysis": {
                "problem_evidence": f"{parsed['known']}\n\n【必要なファクト】\n{parsed['facts']}",
                "background_structure": parsed["gaps"],
                "priority_reason": parsed["next"],
            },
            "objective": {"final_goal": parsed["known"], "kpi_target": parsed["gaps"], "constraints": parsed["next"]},
            "concept":   {"policy_direction": parsed["known"], "evidence": parsed["gaps"], "risks_actions": parsed["next"]},
            "plan":      {"main_actions": parsed["known"], "org_schedule": parsed["gaps"], "cost_effect": parsed["next"]},
            "proposal":  {"exec_summary": parsed["known"], "decision_points": parsed["gaps"], "next_actions": parsed["next"]},
        }.get(step, {})

        blocks = []
        for sec in STEP_SECTIONS.get(step, []):
            key, label = sec["key"], sec["label"]
            already = (existing.get(key) or "").strip()
            if already:
                continue

            draft = (mapping.get(key) or "").strip()
            if not draft or len(draft) < MIN_CHARS_FOR_CAPTURE:
                draft = "関係者の証言・業務ログ・既存統計・実地観察など、裏付けファクトが未取得。"
            blocks.append(f"### {label}\n{draft}")

        if not blocks:
            return ""
        return "— 必要なら、下記を**内容整理**に貼り付けて保存してください。\n" + "\n\n".join(blocks)

    async def _bridge(self, current_step: str, score: int, st: FlexiblePolicyState) -> str:
        if score < 80:
            return ""
        if self._turns(st) < MIN_TURNS_FOR_BRIDGE:
            return ""
        idx = STEP_ORDER.index(current_step)
        if idx >= len(STEP_ORDER) - 1:
            return ""
        next_key = STEP_ORDER[idx + 1]
        next_title = STEP_CONFIG[next_key]["title"]
        return (
            f"ここまでの整理は十分に形になってきました。次は「{next_title}」に進みますか？\n"
            f"（『はい』『進みます』『お願いします』で移動します。続けてこのステップを深掘りする場合は『このまま』等で続行します）"
        )
    
    async def process_flexible(self, user_input: str, session_id: str, current_step: str, project_id: str=None) -> Dict[str, Any]:
        if current_step not in STEP_CONFIG:
            current_step = "analysis"
        st = self._get_state(session_id, project_id)

        target_step = await self._detect_step_switch_intent(user_input, st, current_step)
        if target_step and target_step != current_step:
            msg = f"了解しました。{STEP_CONFIG[target_step]['title']}に移ります。"
            st["conversation_history"] = []
            st["ask_cooldown"] = 0
            st["last_move"] = "advise"
            st["step_completion"][target_step] = 0
            
            self._log_exchange(st, current_step, user_input, msg)
            return {
                "result": msg,
                "type": "navigate",
                "step": current_step,
                "navigate_to": target_step,
                "progress": st["step_completion"].get(current_step, 0),
                "full_state": st,
            }

        t = (user_input or "").strip()
        if t in ["保存した","保存しました","保存完了"] or t.startswith("#保存"):
            msg = "ありがとうございます。内容が反映されました。この方針で続けますか？それとも修正しますか？"
            self._log_exchange(st, current_step, user_input, msg)
            return {"result": msg, "step": current_step, "type": "save_confirm",
                    "progress": st["step_completion"].get(current_step, 0), "full_state": st}

        reply = await self._chat_reply(current_step, user_input, st)

        tail = ""
        save_hint = ""
        bridge = ""
        progress = st["step_completion"].get(current_step, 0)

        if self._enough(st, MIN_TURNS_FOR_STRUCTURE, MIN_RECENT_CHARS):
            parsed = await self._mini_structure(current_step, user_input, st)

            body = (
                f"【{STEP_CONFIG[current_step]['title']}の要点】\n"
                f"●見えていること\n{_two_bullets(parsed['known'])}\n\n"
                f"●足りないこと\n{_two_bullets(parsed['gaps'])}\n\n"
                f"●必要なファクト\n{_two_bullets(parsed['facts'])}\n\n"
                f"●収集方法の提案\n{_two_bullets(parsed['how'])}\n\n"
                f"●次の一歩\n{_two_bullets(parsed['next'])}\n\n"
                f"確認ですが、{parsed.get('q','この方向でよいですか？')}"
            )
            tail += "\n\n" + body

            if self._enough(st, MIN_TURNS_FOR_CAPTURE, 300):
                save_hint = await asyncio.to_thread(self._capture_candidates, st, current_step, parsed)
                if save_hint:
                    tail += "\n\n" + save_hint

            if self._enough(st, MIN_TURNS_FOR_BRIDGE, 450):
                progress = _score(parsed["known"], parsed["gaps"])
                st["step_completion"][current_step] = progress
                bridge = await self._bridge(current_step, progress, st)
                if bridge:
                    tail += "\n\n" + bridge

        result = reply + (tail if tail else "")
        self._log_exchange(st, current_step, user_input, result)

        return {
            "result": result,
            "step": current_step,
            "type": "structure" if tail else "chat",
            "progress": progress,
            "save_suggestions": save_hint or "",
            "bridge_suggestion": bridge or "",
            "full_state": st,
        }
    
    async def _detect_step_switch_intent(self, user_input: str, st: FlexiblePolicyState, current_step: str) -> Optional[str]:
        t = (user_input or "").strip()
        if not t:
            return None
        tl = t.lower()
        m = re.search(r"#\s*(?:移動|move|戻る|back)\s+([a-z]+)", tl)
        if m:
            key = m.group(1)
            return key if key in STEP_CONFIG else None

        GO_CMDS   = r"(?:移る|移ります|移って(?:ください|下さい)?|進む|進める|進みたい|進んで(?:ください|下さい)?|行く|行きたい|行って(?:ください|下さい)?|切り替える|切替える|切替|切り替えて(?:ください|下さい)?|入る|入りたい|入って(?:ください|下さい)?|開始する|開始したい)"
        WANT_CMDS = r"(?:したい|やりたい|行いたい|始めたい|取り組みたい|進めたい)"
        ASK_CMDS  = r"(?:お願いします|お願い致します|お願いいたします|お願い)?"
        MAKE_NOUN = r"(?:作成|策定|検討)"
        BACK_CMDS = r"(?:戻る|戻ります|戻りたい|戻って(?:ください|下さい)?|戻して(?:ください|下さい)?|戻れます(?:か)?|戻してほしい)"

        if re.search(r"(?:前|ひとつ前|直前)のステップに?\s*" + BACK_CMDS, t):
            try:
                idx = STEP_ORDER.index(current_step)
                if idx > 0:
                    return STEP_ORDER[idx - 1]
            except ValueError:
                pass
        
        for key, aliases in STEP_NAME_ALIASES.items():
            for a in aliases:
                a_esc = re.escape(a)
                pat_back = fr"{a_esc}\s*(?:{MAKE_NOUN})?\s*(?:へ|に)?\s*{BACK_CMDS}"
                if re.search(pat_back, t):
                    return key

        matches: list[tuple[int, str]] = []
        for key, aliases in STEP_NAME_ALIASES.items():
            for a in aliases:
                a_esc = re.escape(a)
                pat1 = fr"{a_esc}\s*(?:{MAKE_NOUN})?\s*(?:へ|に)?\s*{GO_CMDS}"
                pat2 = fr"{a_esc}\s*(?:{MAKE_NOUN})?\s*(?:を|に)?\s*{WANT_CMDS}"
                pat3 = fr"(?:次は|次に)\s*{a_esc}"
                pat4 = fr"{a_esc}\s*(?:{MAKE_NOUN})?\s*(?:を)?\s*お願いします[。!！]?$"

                for pat in (pat1, pat2, pat3, pat4):
                    for m in re.finditer(pat, t):
                        matches.append((m.start(), key))

        if matches:
            matches.sort(key=lambda x: x[0])
            return matches[-1][1]
        
        if re.search(r"^(はい|了解|ok|お願いします|お願いいたします|お願い致します|進みます|次へ|進めよう|お願いします[。!！]?)$", t.strip(), re.IGNORECASE):
            last_assistant = ""
            for m in reversed(st["conversation_history"]):
                if m.get("role") == "assistant":
                    last_assistant = m.get("content", "")
                    break
            if last_assistant:
                for key in STEP_ORDER:
                    title = STEP_CONFIG[key]["title"]
                    if re.search(fr"{re.escape(title)}に進みますか", last_assistant):
                        return key
                for key in STEP_ORDER:
                    title = STEP_CONFIG[key]["title"]
                    if f"次は「{title}」" in last_assistant:
                        return key

        return None
    
    async def get_session_state(self, session_id: str) -> Dict[str, Any]:
        st = self.state_storage.get(session_id)
        if not st:
            return {"error": "セッションが見つかりません"}
        return {
            "session_id": st["session_id"],
            "project_id": st["project_id"],
            "last_updated_step": st["last_updated_step"],
            "step_timestamps": st["step_timestamps"],
            "step_completion": st["step_completion"],
        }