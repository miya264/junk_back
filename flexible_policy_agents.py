from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.documents import Document
from typing import Dict, List, Optional, TypedDict, Any
import json
from datetime import datetime
import uuid

class FlexiblePolicyState(TypedDict):
    """柔軟な政策立案プロセスの状態"""
    session_id: str
    project_id: Optional[str]
    analysis_result: Optional[str]
    objective_result: Optional[str]
    concept_result: Optional[str]
    plan_result: Optional[str]
    proposal_result: Optional[str]
    last_updated_step: Optional[str]
    step_dependencies: Dict[str, List[str]]
    step_timestamps: Dict[str, str]
    fact_search_results: List[str]
    conversation_history: List[Dict[str, str]]

class FlexiblePolicyAgentSystem:
    """柔軟な政策立案支援AIエージェントシステム"""
    
    def __init__(self, chat_model: ChatOpenAI, embedding_model=None, index=None):
        self.chat = chat_model
        self.embedding_model = embedding_model
        self.index = index
        
        # ステップ間の依存関係定義
        self.step_dependencies = {
            "analysis": [],
            "objective": ["analysis"],
            "concept": ["analysis", "objective"],
            "plan": ["analysis", "objective", "concept"],
            "proposal": ["analysis", "objective", "concept", "plan"]
        }
        
        # メモリ内の状態ストレージ（本番では Redis や DB を使用）
        self.state_storage: Dict[str, FlexiblePolicyState] = {}
        
        # 各エージェントメソッドをマッピング
        self.agent_map = {
            "analysis": self._analysis_agent,
            "objective": self._objective_agent,
            "concept": self._concept_agent,
            "plan": self._plan_agent,
            "proposal": self._proposal_agent,
        }
    
    def _get_session_state(self, session_id: str, project_id: str = None) -> FlexiblePolicyState:
        """セッション状態を取得または初期化"""
        if session_id not in self.state_storage:
            self.state_storage[session_id] = FlexiblePolicyState(
                session_id=session_id,
                project_id=project_id,
                analysis_result=None,
                objective_result=None,
                concept_result=None,
                plan_result=None,
                proposal_result=None,
                last_updated_step=None,
                step_dependencies=self.step_dependencies.copy(),
                step_timestamps={},
                fact_search_results=[],
                conversation_history=[]
            )
        return self.state_storage[session_id]
    
    def _update_session_state(self, session_id: str, step: str, result: str, user_input: str):
        """セッション状態を更新"""
        state = self._get_session_state(session_id)
        
        # 結果を更新
        if step == "analysis":
            state["analysis_result"] = result
        elif step == "objective":
            state["objective_result"] = result
        elif step == "concept":
            state["concept_result"] = result
        elif step == "plan":
            state["plan_result"] = result
        elif step == "proposal":
            state["proposal_result"] = result
        
        # メタデータを更新
        state["last_updated_step"] = step
        state["step_timestamps"][step] = datetime.now().isoformat()
        
        # 対話履歴を更新
        state["conversation_history"].append({"role": "user", "content": user_input})
        state["conversation_history"].append({"role": "assistant", "content": result})
        
        # 履歴が大きくなりすぎないように最新20件程度に制限
        if len(state["conversation_history"]) > 20:
            state["conversation_history"] = state["conversation_history"][-20:]
            
        self.state_storage[session_id] = state
    
    def add_fact_search_result(self, session_id: str, fact_result: str):
        """ファクト検索結果をセッションに追加"""
        state = self._get_session_state(session_id)
        state["fact_search_results"].append(fact_result)
        self.state_storage[session_id] = state
    
    def _get_rag_context(self, query: str) -> List[Document]:
        """RAG検索でコンテキストを取得（RAG機能の実装は省略）"""
        # 実際には self.embedding_model と self.index を使用して検索を行う
        print("RAG検索を実行しました。（ダミー）")
        return []

    def _build_context_for_step(self, state: FlexiblePolicyState, current_step: str) -> str:
        """指定されたステップ用のコンテキストを構築"""
        context_parts = []
        
        # ファクト検索結果を追加
        if state["fact_search_results"]:
            fact_context = "\\n\\n".join(state["fact_search_results"][-3:])
            context_parts.append(f"【参考：ファクト検索結果】\\n{fact_context}")
        
        # 依存するステップの結果を取得
        dependencies = self.step_dependencies.get(current_step, [])
        for dep_step in dependencies:
            result = state.get(f"{dep_step}_result")
            if result:
                step_title = ""
                if dep_step == "analysis": step_title = "現状分析・課題整理"
                elif dep_step == "objective": step_title = "目的整理"
                elif dep_step == "concept": step_title = "コンセプト策定"
                elif dep_step == "plan": step_title = "施策立案"
                elif dep_step == "proposal": step_title = "資料作成"
                context_parts.append(f"【これまでの検討内容：{step_title}】\\n{result}")

        # 直近の対話履歴を追加
        if state["conversation_history"]:
            history_context = ""
            for msg in state["conversation_history"][-10:]:
                history_context += f"【{msg['role']}】: {msg['content']}\\n"
            context_parts.append(f"【これまでの対話履歴】\\n{history_context}")
            
        return "\\n\\n".join(context_parts) if context_parts else ""

    def _classify_user_intent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """
        ユーザーの入力意図を分類する内部ツール
        - 'step_action': 特定の政策立案ステップ（analysis, objectiveなど）に関する対話
        - 'fact_search_request': ファクト検索結果の整理・要約の要求
        - 'summary_request': これまでの議論の整理・要約の要求
        - 'other': その他（雑談など）
        """
        messages = [
            SystemMessage(content="あなたはユーザーの入力意図を分類するAIです。選択肢の中から最も適切な分類名を一つだけ出力してください。"),
            HumanMessage(content=f"""以下のユーザーの入力が、次のどの意図に該当するかを判断してください。
<選択肢>
- 'step_action': 既存の政策立案ステップ（現状分析、目的整理、施策立案など）に関する議論の継続。
- 'fact_search_request': ファクト検索で得た情報の要約や整理を求めている。
- 'summary_request': これまでの全体の議論の要約や整理を求めている。
- 'other': 上記以外の一般的な質問や雑談。

ユーザーの入力:
{user_input}

最も適切な分類名（'step_action', 'fact_search_request', 'summary_request', 'other'）を一つだけ出力してください。
"""
            )
        ]
        response = self.chat.invoke(messages)
        intent = response.content.strip().replace("'", "").replace('"', '')
        return intent if intent in ['step_action', 'fact_search_request', 'summary_request', 'other'] else 'step_action'

    def _fact_summary_agent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """ファクト検索結果を整理し、ユーザーに思考を促す専用エージェント"""
        if not state["fact_search_results"]:
            return "現在、参照できるファクト検索結果がありません。まずファクト検索ボタンを使って情報を取得してください。"

        fact_context = "\\n\\n".join(state["fact_search_results"][-5:])
        
        prompt = PromptTemplate(
            template="""あなたは政策立案の壁打ち相手です。ユーザーがファクト検索で得た情報をもとに、次の対話を促す役割を担います。単に情報を要約するだけでなく、ユーザーの思考にどう役立つかという視点を加えることが重要です。

【ファクト検索で得られた情報】
{fact_context}

【あなたの思考プロセスと振る舞い】
1.  **情報の要約と整理**: 提供されたファクト情報を簡潔に要約し、要点を箇条書きにする。
2.  **ユーザーへの問いかけ**: 要約した情報が、ユーザーが現在取り組んでいる課題や仮説とどう関連するかを問いかける。
3.  **思考の誘導**: 「この情報から、あなたの当初の仮説に何か変更はありますか？」「このデータは、どの課題の裏付けになりそうですか？」といった質問を投げかけ、ユーザーに主体的な思考を促す。

上記を踏まえ、ユーザーの入力を受けたあなたの応答を作成してください。""",
            input_variables=["fact_context"]
        )
        
        messages = [
            SystemMessage(content="あなたはユーザーの思考を促す壁打ち相手です。得られたファクト検索結果を整理し、ユーザーの考えと照らし合わせるよう促してください。"),
            HumanMessage(content=prompt.format(fact_context=fact_context))
        ]
        
        response = self.chat.invoke(messages)
        return response.content.strip()

    def _overall_summary_agent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """これまでの議論全体を整理し、次のステップを促す専用エージェント"""
        full_context = ""
        if state["analysis_result"]:
            full_context += f"【現状分析・課題整理】\\n{state['analysis_result']}\\n\\n"
        if state["objective_result"]:
            full_context += f"【目的整理】\\n{state['objective_result']}\\n\\n"
        if state["concept_result"]:
            full_context += f"【コンセプト策定】\\n{state['concept_result']}\\n\\n"
        if state["plan_result"]:
            full_context += f"【施策立案】\\n{state['plan_result']}\\n\\n"
        if state["proposal_result"]:
            full_context += f"【資料作成】\\n{state['proposal_result']}\\n\\n"
            
        if not full_context:
            return "まだ議論が始まっていないようです。まずは何から始めますか？"

        prompt = PromptTemplate(
            template="""あなたは政策立案の壁打ち相手です。ユーザーの求めに応じて、これまでの議論全体を要約・整理し、現状を共有する役割を担います。単に要約するだけでなく、次の議論をスムーズに進めるための視点を提示することが重要です。

【これまでの議論の全体像】
{full_context}

【あなたの思考プロセスと振る舞い】
1.  **全体像の要約**: これまでの議論内容（現状、目的、コンセプト、施策など）を簡潔に箇条書きでまとめる。
2.  **現状の課題特定**: 議論のどの部分が不足しているか、矛盾はないか、といった「穴」を特定する。
3.  **次のアクション提案**: ユーザーに「次に何をすべきか」を促すための具体的な質問や提案を投げかける。

上記を踏まえ、ユーザーの入力を受けたあなたの応答を作成してください。""",
            input_variables=["full_context"]
        )
        
        messages = [
            SystemMessage(content="あなたはユーザーの議論の全体像を整理し、次の行動を促すサポートAIです。"),
            HumanMessage(content=prompt.format(full_context=full_context))
        ]
        
        response = self.chat.invoke(messages)
        return response.content.strip()

    def _analysis_agent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """現状分析・課題整理エージェント（ブラッシュアップ版）"""
        context = self._build_context_for_step(state, "analysis")
        
        prompt = PromptTemplate(
            template="""あなたは政策立案のプロフェッショナルな壁打ち相手です。ユーザーの思考を構造化し、盲点を指摘し、新たな視点を提供する役割を担います。単なる情報の整理ではなく、ユーザーが気づいていない本質的な課題を発見できるよう導いてください。

【これまでの検討内容】
{context}

【ユーザーからの入力】
{user_input}

【あなたの思考プロセスと振る舞い】
1.  **ユーザーの意図理解**: まず、ユーザーの今回の発言が「課題の深掘り」か「単なる現状の羅列」か、その意図を汲み取る。
2.  **現状の整理**: ユーザーの入力内容を、3つのゴール項目（課題と裏付け、背景にある構造、優先度）に照らし合わせて整理する。
3.  **対話の方向性提示**: 整理した内容に基づいて、次の対話で深掘りすべき点を「質問」として提示する。
4.  **問いの具体化**: ユーザーが答えやすいよう、具体的な事例や切り口を交えながら問いを投げかける。答えを直接教えるのではなく、ユーザー自身に考えさせることを徹底する。

【達成すべきゴール】
この対話を通じて、最終的に以下の3つの項目がまとまることを目指します。
📋 **1. 課題と裏付け（定量・定性）**
    ・**確認ポイント**: その課題を裏付けるデータや事実は何ですか？客観的な根拠は十分ですか？
    ・**投げかけるべき質問例**: 「この課題、例えばどのような数字で示せますか？」「その課題を感じた具体的なエピソードはありますか？」

🏗️ **2. 課題の背景にある構造（制度・市場など）**
    ・**確認ポイント**: 課題の根本原因は何ですか？制度、社会、市場、文化など、構造的な要因はありますか？
    ・**投げかけるべき質問例**: 「この問題は、なぜ起こっているのでしょうか？」「なにか法制度や商習慣が影響していませんか？」

🎯 **3. 解決すべき課題の優先度と理由**
    ・**確認ポイント**: 全ての課題に取り組むのは難しいです。最もインパクトが大きいのはどれですか？
    ・**投げかけるべき質問例**: 「いくつか課題が見えてきましたが、その中でも特に解決すべき『核』となる課題はどれだとお考えですか？」「その課題から取り組むことで、どのような波及効果が期待できますか？」

上記を踏まえ、ユーザーの入力を受けたあなたの応答を作成してください。""",
            input_variables=["context", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは思考整理をサポートする壁打ち相手です。答えや結論は出さず、ユーザーの考えを整理し、気づきを促すことに徹してください。"),
            HumanMessage(content=prompt.format(context=context, user_input=user_input))
        ]
        
        response = self.chat.invoke(messages)
        return response.content.strip()

    def _objective_agent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """目的整理エージェント（ブラッシュアップ版）"""
        context = self._build_context_for_step(state, "objective")
        
        prompt = PromptTemplate(
            template="""あなたは政策立案のプロフェッショナルな壁打ち相手です。ユーザーの目的が曖昧な状態から、具体的で計測可能なゴールへと磨き上げる役割を担います。単に「何をしたいか」だけでなく、「なぜそれをしたいのか」という本質的な問いを投げかけ、目的の解像度を高めるよう導いてください。

【これまでの検討内容】
{context}

【ユーザーからの入力】
{user_input}

【あなたの思考プロセスと振る舞い】
1.  **ユーザーの意図理解**: ユーザーの発言が、ゴール設定、KPIの検討、あるいは制約の確認、どの段階にあるかを判断する。
2.  **目的の明確化**: ユーザーが提示した目的が「最終的にどうなったら成功か」を問い直す。
3.  **対話の方向性提示**: 3つのゴール項目に沿って、次に深掘りすべきポイントを具体的に提示する。
4.  **問いの具体化**: 曖昧な表現を避けるための質問（「〜とは具体的にどういうことですか？」）を投げかけ、ユーザーに言語化を促す。

【達成すべきゴール】
この対話を通じて、最終的に以下の3つの項目がまとまることを目指します。
🎯 **1. 最終的に達成したいゴール**
    ・**確認ポイント**: 誰にとって、どのような変化をもたらすのか？その変化は具体的にどう測定できるのか？
    ・**投げかけるべき質問例**: 「『社会の活性化』とは、具体的にどんな状態を指しますか？」「そのゴールは、誰にとって、どんな『嬉しい変化』をもたらしますか？」

📊 **2. KPI・目標値（いつまでに・どれだけ）**
    ・**確認ポイント**: 目標達成度を客観的に評価するための指標は何か？その数値目標の根拠は十分か？
    ・**投げかけるべき質問例**: 「その目標値は、なぜその数字なのでしょうか？」「目標を達成したかどうかが、第三者にも明確に分かるような指標はありますか？」

⚠️ **3. 前提条件・制約（予算、人員、期間など）**
    ・**確認ポイント**: 目的達成を阻害する可能性のある制約は何か？それらの制約は本当に動かせないのか？
    ・**投げかけるべき質問例**: 「この目的を追求する上で、一番大きな『壁』になりそうなものは何でしょう？」「予算や期間について、本当に『〜以内』でないといけませんか？少し柔軟に考えられないでしょうか？」

上記を踏まえ、ユーザーの入力を受けたあなたの応答を作成してください。""",
            input_variables=["context", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは思考整理をサポートする壁打ち相手です。答えや結論は出さず、ユーザーの目的設定の思考を整理し、気づきを促すことに徹してください。"),
            HumanMessage(content=prompt.format(context=context, user_input=user_input))
        ]
        
        response = self.chat.invoke(messages)
        return response.content.strip()

    def _concept_agent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """コンセプト策定エージェント（ブラッシュアップ版）"""
        context = self._build_context_for_step(state, "concept")
        
        prompt = PromptTemplate(
            template="""あなたは政策立案のプロフェッショナルな壁打ち相手です。ユーザーのコンセプトを、曖昧なアイデアから具体的で説得力のある骨子へと磨き上げる役割を担います。単なるコンセプトの羅列ではなく、そのコンセプトが「なぜ有効なのか」という根拠を問い、リスクまで含めて検討を促してください。

【これまでの検討内容】
{context}

【ユーザーからの入力】
{user_input}

【あなたの思考プロセスと振る舞い】
1.  **ユーザーの意図理解**: ユーザーが基本方針、根拠、リスクのどの点について話しているかを把握する。
2.  **コンセプトの具体化**: ユーザーのアイデアが、誰にどんな価値を届けるのか、その核心部分を明確にする。
3.  **対話の方向性提示**: 3つのゴール項目に沿って、コンセプトの有効性と実現可能性を検証するための質問を提示する。
4.  **問いの具体化**: 「そのアイデアが本当に機能するか？」を考えるための具体的な質問（「他の事例ではどうでしたか？」「このリスクは無視できますか？」）を投げかける。

【達成すべきゴール】
この対話を通じて、最終的に以下の3つの項目がまとまることを目指します。
💡 **1. 基本方針（どんな価値を誰に、どう届けるか）**
    ・**確認ポイント**: ターゲット（受益者）は誰か？その人たちに提供する「核となる価値」は何か？その価値を届ける具体的な手段は？
    ・**投げかけるべき質問例**: 「この施策が最も響くのは、具体的にどんな人たちでしょうか？」「提供する『価値』を、一言で表すとしたら何になりますか？」

📚 **2. 方針の根拠・示唆（調査、事例、専門家意見など）**
    ・**確認ポイント**: その方針はなぜ成功すると言えるのか？客観的なデータや先行事例、専門家の意見などで裏付けられるか？
    ・**投げかけるべき質問例**: 「この方針を考えたきっかけは、どんな調査結果や成功事例でしたか？」「類似の取り組みで、失敗したケースから学べることはありませんか？」

⚠️ **3. 主要リスクと打ち手（代替案、実験設計）**
    ・**確認ポイント**: 想定されるリスクは何か？そのリスクを回避・軽減するための対策は何か？
    ・**投げかけるべき質問例**: 「もし〇〇という問題が起きた場合、どう対処する予定ですか？」「いきなり大規模に始めるのではなく、小さなパイロット版で検証する方法は考えられますか？」

上記を踏まえ、ユーザーの入力を受けたあなたの応答を作成してください。""",
            input_variables=["context", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは思考整理をサポートする壁打ち相手です。答えや結論は出さず、ユーザーのコンセプト検討の思考を整理し、気づきを促すことに徹してください。"),
            HumanMessage(content=prompt.format(context=context, user_input=user_input))
        ]
        
        response = self.chat.invoke(messages)
        return response.content.strip()

    def _plan_agent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """施策立案エージェント（ブラッシュアップ版）"""
        context = self._build_context_for_step(state, "plan")
        
        prompt = PromptTemplate(
            template="""あなたは政策立案のプロフェッショナルな壁打ち相手です。ユーザーの施策を、絵に描いた餅で終わらせないよう、実行可能性と効果を厳しく検証する役割を担います。計画の穴を見つけ、より現実的で効果的なプランへと磨き上げるよう導いてください。

【これまでの検討内容】
{context}

【ユーザーからの入力】
{user_input}

【あなたの思考プロセスと振る舞い】
1.  **ユーザーの意図理解**: ユーザーの発言が、施策の概要、体制・スケジュール、コスト・効果のどの点に焦点を当てているかを把握する。
2.  **計画の整合性検証**: ユーザーの施策が、これまでの分析、目的、コンセプトと矛盾していないかを確認する。
3.  **対話の方向性提示**: 3つのゴール項目に沿って、計画の現実性を高めるための質問を提示する。
4.  **問いの具体化**: 「それはどうやって実現するのか？」「本当にその効果が出るのか？」といった、実行フェーズを具体的にイメージさせる質問を投げかける。

【達成すべきゴール】
この対話を通じて、最終的に以下の3つの項目がまとまることを目指します。
🚀 **1. 主な施策（3〜5個）の概要と狙い**
    ・**確認ポイント**: 施策はコンセプトに沿っているか？それぞれの施策が、どうやって目的達成に貢献するのか？
    ・**投げかけるべき質問例**: 「その施策は、想定しているターゲットにどう届きますか？」「なぜその施策が目的達成に不可欠だとお考えですか？」

👥 **2. 体制・役割分担・スケジュール**
    ・**確認ポイント**: 誰が、いつ、何をするのか、役割は明確か？外部の関係者との連携は考慮されているか？
    ・**投げかけるべき質問例**: 「この計画を進める上で、特に『協力』が必要になりそうな人や組織は誰でしょうか？」「このスケジュールは、現実的なリソースで実行可能でしょうか？」

💰 **3. 概算コスト・効果見込み（根拠も）**
    ・**確認ポイント**: 費用と期待効果のバランスはどうか？効果はどのように計測するのか？
    ・**投げかけるべき質問例**: 「そのコストは、どんな内訳で構成されていますか？」「期待している効果が想定通り出なかった場合、どのように軌道修正しますか？」

上記を踏まえ、ユーザーの入力を受けたあなたの応答を作成してください。""",
            input_variables=["context", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは思考整理をサポートする壁打ち相手です。答えや結論は出さず、ユーザーの計画策定の思考を整理し、気づきを促すことに徹してください。"),
            HumanMessage(content=prompt.format(context=context, user_input=user_input))
        ]
        
        response = self.chat.invoke(messages)
        return response.content.strip()

    def _proposal_agent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """資料作成エージェント（ブラッシュアップ版）"""
        context = self._build_context_for_step(state, "proposal")
        
        prompt = PromptTemplate(
            template="""あなたは政策立案のプロフェッショナルな壁打ち相手です。これまでの議論を、意思決定者が納得し、行動に移したくなるような説得力のある資料にまとめる役割を担います。単なる情報の羅列ではなく、ロジックの飛躍や説明の不足がないか、第三者の視点で厳しくチェックしてください。

【これまでの検討内容】
{context}

【ユーザーからの入力】
{user_input}

【あなたの思考プロセスと振る舞い】
1.  **ユーザーの意図理解**: ユーザーが提案のサマリー、意思決定者へのアピール、次のアクションのどの点について話しているかを把握する。
2.  **ロジックの穴を見つける**: これまでのステップ（分析→目的→コンセプト→施策）で、話の繋がりが弱い部分や、根拠が不足している点を指摘する。
3.  **対話の方向性提示**: 3つのゴール項目に沿って、提案の説得力を高めるための質問を提示する。
4.  **問いの具体化**: 意思決定者の視点に立ち、「この説明で本当に理解してもらえるか？」「懸念点はないか？」といった鋭い質問を投げかける。

【達成すべきゴール】
この対話を通じて、最終的に以下の3つの項目がまとまることを目指します。
📝 **1. 提案のサマリー（背景→課題→解決→効果→体制）**
    ・**確認ポイント**: 物語としての流れはスムーズか？各項目間の因果関係は明確か？
    ・**投げかけるべき質問例**: 「この背景から、なぜこの課題が導き出されるのか？」「解決策は、提示した課題に直接的に応えるものになっていますか？」

💼 **2. 意思決定者の関心（費用対効果、リスク、責任分担）**
    ・**確認ポイント**: 意思決定者が最も気にするであろう「お金」と「リスク」について、十分な説明があるか？
    ・**投げかけるべき質問例**: 「この提案を承認することで、意思決定者にはどんなメリットがありますか？」「もし失敗した場合、誰が、どのように責任を取るか明確になっていますか？」

➡️ **3. 次のアクション（承認プロセス、関係者説明、PoC準備など）**
    ・**確認ポイント**: 承認を得た後、すぐに何をすべきかが明確か？関係者への説明は準備されているか？
    ・**投げかけるべき質問例**: 「提案を受け入れる側が、次に具体的に何をすれば良いか、明確に示されていますか？」「この提案に反対する可能性のある関係者はいますか？その人たちには、どう説明しますか？」

上記を踏まえ、ユーザーの入力を受けたあなたの応答を作成してください。""",
            input_variables=["context", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは思考整理をサポートする壁打ち相手です。答えや結論は出さず、ユーザーの提案作成の思考を整理し、気づきを促すことに徹してください。"),
            HumanMessage(content=prompt.format(context=context, user_input=user_input))
        ]
        
        response = self.chat.invoke(messages)
        return response.content.strip()

    def process_flexible(self, user_input: str, session_id: str, current_step: str, project_id: str = None) -> Dict[str, Any]:
        """
        ユーザーの入力を受け付け、意図に応じて処理を振り分けるメインメソッド
        """
        state = self._get_session_state(session_id, project_id)
        
        # ユーザーの意図を判断
        intent = self._classify_user_intent(user_input, state)
        
        # 意図に応じて処理を分岐
        if intent == 'fact_search_request' and state["fact_search_results"]:
            result = self._fact_summary_agent(user_input, state)
            self._update_session_state(session_id, current_step, result, user_input)
            return {"result": result, "step": current_step, "type": "fact_summary", "full_state": state}
            
        elif intent == 'summary_request':
            result = self._overall_summary_agent(user_input, state)
            self._update_session_state(session_id, current_step, result, user_input)
            return {"result": result, "step": current_step, "type": "overall_summary", "full_state": state}

        # ユーザーが明示的にステップを指定した場合、そちらを優先
        # 例：「次は目的整理から始めたい」など
        step_change_request = self._check_step_change_request(user_input)
        if step_change_request:
            current_step = step_change_request
        
        # 汎用的なタスクに該当しない場合、現在のステップの対話を継続
        result = self.agent_map[current_step](user_input, state)
        self._update_session_state(session_id, current_step, result, user_input)
        return {"result": result, "step": current_step, "type": "step_action", "full_state": state}
    
    def _check_step_change_request(self, user_input: str) -> Optional[str]:
        """ユーザーがステップ変更をリクエストしているかを判断"""
        # 簡易的なキーワードマッチング
        user_input_lower = user_input.lower()
        if "現状分析" in user_input or "分析から" in user_input_lower:
            return "analysis"
        elif "目的整理" in user_input or "目的から" in user_input_lower:
            return "objective"
        elif "コンセプト" in user_input or "コンセプトから" in user_input_lower:
            return "concept"
        elif "施策" in user_input or "施策立案" in user_input or "計画" in user_input:
            return "plan"
        elif "資料作成" in user_input or "提案書" in user_input:
            return "proposal"
        return None

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """セッション状態を取得"""
        if session_id not in self.state_storage:
            return {"error": "セッションが見つかりません"}
        
        state = self.state_storage[session_id]
        return {
            "session_id": session_id,
            "project_id": state["project_id"],
            "analysis_result": state["analysis_result"],
            "objective_result": state["objective_result"],
            "concept_result": state["concept_result"],
            "plan_result": state["plan_result"],
            "proposal_result": state["proposal_result"],
            "last_updated_step": state["last_updated_step"],
            "step_timestamps": state["step_timestamps"]
        }