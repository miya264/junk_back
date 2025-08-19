"""
柔軟な政策立案支援AIエージェントシステム
どのステップからでも開始可能で、システム全体に反映される仕組み
"""

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
    step_dependencies: Dict[str, List[str]]  # どのステップがどのステップに依存するか
    step_timestamps: Dict[str, str]  # 各ステップの最終更新時刻
    fact_search_results: List[str]  # ファクト検索結果の履歴

class FlexiblePolicyAgentSystem:
    """柔軟な政策立案支援AIエージェントシステム"""
    
    def __init__(self, chat_model: ChatOpenAI, embedding_model, index):
        self.chat = chat_model
        self.embedding_model = embedding_model
        self.index = index
        
        # ステップ間の依存関係定義
        self.step_dependencies = {
            "analysis": [],  # 分析は独立して実行可能
            "objective": ["analysis"],  # 目的整理は分析結果を参照
            "concept": ["analysis", "objective"],  # コンセプトは分析と目的を参照
            "plan": ["analysis", "objective", "concept"],  # 計画は前の3つを参照
            "proposal": ["analysis", "objective", "concept", "plan"]  # 提案書は全てを参照
        }
        
        # メモリ内の状態ストレージ（本番では Redis や DB を使用）
        self.state_storage: Dict[str, FlexiblePolicyState] = {}
    
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
                fact_search_results=[]
            )
        return self.state_storage[session_id]
    
    def _update_session_state(self, session_id: str, step: str, result: str):
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
        
        self.state_storage[session_id] = state
    
    def add_fact_search_result(self, session_id: str, fact_result: str):
        """ファクト検索結果をセッションに追加"""
        state = self._get_session_state(session_id)
        state["fact_search_results"].append(fact_result)
        self.state_storage[session_id] = state
    
    def _get_rag_context(self, query: str, cache_key: str = None) -> List[Document]:
        """RAG検索でコンテキストを取得（キャッシュ付き）"""
        try:
            query_embedding = self.embedding_model.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            
            docs = []
            for match in results['matches']:
                doc = Document(
                    page_content=match['metadata']['text'],
                    metadata={
                        'source': match['metadata']['source'],
                        'score': match['score']
                    }
                )
                docs.append(doc)
            return docs
        except Exception as e:
            print(f"RAG検索エラー: {e}")
            return []
    
    def _build_context_for_step(self, state: FlexiblePolicyState, current_step: str) -> str:
        """指定されたステップ用のコンテキストを構築"""
        context_parts = []
        
        # ファクト検索結果を追加
        if state["fact_search_results"]:
            fact_context = "\\n\\n".join(state["fact_search_results"][-3:])  # 最新3件
            context_parts.append(f"【参考：ファクト検索結果】\\n{fact_context}")
        
        # 依存するステップの結果を取得
        dependencies = self.step_dependencies.get(current_step, [])
        
        for dep_step in dependencies:
            if dep_step == "analysis" and state["analysis_result"]:
                context_parts.append(f"【現状分析・課題整理】\\n{state['analysis_result']}")
            elif dep_step == "objective" and state["objective_result"]:
                context_parts.append(f"【目的整理】\\n{state['objective_result']}")
            elif dep_step == "concept" and state["concept_result"]:
                context_parts.append(f"【コンセプト策定】\\n{state['concept_result']}")
            elif dep_step == "plan" and state["plan_result"]:
                context_parts.append(f"【施策立案】\\n{state['plan_result']}")
            elif dep_step == "proposal" and state["proposal_result"]:
                context_parts.append(f"【資料作成】\\n{state['proposal_result']}")
        
        return "\\n\\n".join(context_parts) if context_parts else ""
    
    def _analysis_agent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """現状分析・課題整理エージェント（改良版）"""
        # RAG検索は除去 - ファクト検索ボタン専用とする
        rag_context = ""
        
        # ファクト検索結果と他のステップの結果を参考にする
        related_context = ""
        if state["fact_search_results"]:
            fact_context = "\\n\\n".join(state["fact_search_results"][-2:])  # 最新2件
            related_context += f"\\n\\n【参考：ファクト検索結果】\\n{fact_context}"
        if state["objective_result"]:
            related_context += f"\\n\\n【参考：設定された目的】\\n{state['objective_result']}"
        if state["concept_result"]:
            related_context += f"\\n\\n【参考：策定されたコンセプト】\\n{state['concept_result']}"
        
        prompt = PromptTemplate(
            template="""あなたは思考整理をサポートする壁打ち相手です。ユーザーの考えを整理し、見落としを指摘し、気づきを促すのが役割です。答えは出しません。

これまでの検討内容:
{related_context}

ユーザーが考えていること: {user_input}

【あなたの役割】
- ユーザーの考えを整理して、構造化をサポートする
- 見落としがちな視点や抜けている要素を指摘する
- より深く考えるための質問を投げかける
- 判断や結論は出さず、ユーザー自身が気づけるよう導く

最終的には以下の3つがまとまると良いですね：

📋 **1. 課題と裏付け（定量・定性）**
ユーザーが認識している課題について：
・定量的なデータや数字で示せる部分はありますか？
・定性的に感じている課題も整理してみませんか？
・その裏付けとなる根拠は十分でしょうか？

🏗️ **2. 課題の背景にある構造（制度・市場など）**
課題の背景について：
・制度や仕組みの面で影響している要因はありませんか？
・市場環境や競争状況はいかがでしょう？
・他にも構造的な要因がありそうですね

🎯 **3. 解決すべき課題の優先度と理由**
どこから取り組むかについて：
・インパクトが大きそうなのはどれでしょう？
・実現しやすそうなものはありますか？
・優先順位をつけるとしたら、その理由は？

「この課題の根拠は十分ですか？」「背景の構造で見落としているものはありませんか？」「優先順位の理由を聞かせてください」といった形で、3つの観点での整理をサポートします。""",
            input_variables=["related_context", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは思考整理をサポートする壁打ち相手です。答えや結論は出さず、ユーザーの考えを整理し、気づきを促すことに徹してください。"),
            HumanMessage(content=prompt.format(
                related_context=related_context,
                user_input=user_input
            ))
        ]
        
        response = self.chat.invoke(messages)
        return response.content.strip()
    
    def _objective_agent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """目的整理エージェント（改良版）"""
        # 依存関係のコンテキストを構築
        context = self._build_context_for_step(state, "objective")
        
        prompt = PromptTemplate(
            template="""あなたは思考整理をサポートする壁打ち相手です。ユーザーが考えている目的を整理し、より明確にするためのサポートをします。答えは出しません。

これまでの検討内容:
{context}

ユーザーが考えていること: {user_input}

【あなたの役割】
- ユーザーの目的設定の思考を整理する
- 曖昧な部分や抜けている要素を指摘する
- より具体化するための質問を投げかける
- 判断や決定はユーザーに委ね、気づきを促す

最終的には以下の3つがまとまると良いですね：

🎯 **1. 最終的に達成したいゴール**
具体的な目標について：
・どんな状態になったら「成功」と言えそうでしょうか？
・誰が、どのように変わることを期待していますか？
・ゴールをもう少し具体的に表現できそうですか？

📊 **2. KPI・目標値（いつまでに・どれだけ）**
成果の測り方について：
・達成度をどうやって測りたいでしょうか？
・いつまでに、どの程度の成果を期待しますか？
・その目標値、現実的でしょうか？根拠はいかがですか？

⚠️ **3. 前提条件・制約（予算、人員、期間など）**
制限要因について：
・予算、人員、期間の制約はいかがでしょう？
・他にも制約になりそうなことはありませんか？
・これらの制約、本当に動かせないものでしょうか？

「ゴールをもう少し具体的に表現できそうですね」「この測定方法で本当に分かりそうですか？」「制約の部分で見落としはありませんか？」といった形で、3つの観点での整理をサポートします。""",
            input_variables=["context", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは思考整理をサポートする壁打ち相手です。答えや結論は出さず、ユーザーの目的設定の思考を整理し、気づきを促すことに徹してください。"),
            HumanMessage(content=prompt.format(context=context, user_input=user_input))
        ]
        
        response = self.chat.invoke(messages)
        return response.content.strip()
    
    def _concept_agent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """コンセプト策定エージェント（改良版）"""
        # 依存関係のコンテキストを構築
        context = self._build_context_for_step(state, "concept")
        
        # RAG検索は除去 - ファクト検索ボタン専用とする
        rag_context = ""
        
        prompt = PromptTemplate(
            template="""あなたは思考整理をサポートする壁打ち相手です。ユーザーが考えているコンセプトを整理し、ブラッシュアップするためのサポートをします。答えは出しません。

これまでの検討内容:
{context}

ユーザーが考えていること: {user_input}

【あなたの役割】
- ユーザーのコンセプト検討の思考を整理する
- 検討漏れや曖昧な部分を指摘する
- より深く考えるための視点を提供する
- 判断や結論は出さず、ユーザー自身が気づけるよう導く

最終的には以下の3つがまとまると良いですね：

💡 **1. 基本方針（どんな価値を誰に、どう届けるか）**
基本的な考え方について：
・誰に向けた取り組みにしたいでしょうか？
・どんな価値や効果を提供したいですか？
・どうやって届ける方法を考えていますか？

📚 **2. 方針の根拠・示唆（調査、事例、専門家意見など）**
アプローチの根拠について：
・なぜそのアプローチが良いと思われますか？
・参考にできそうな事例や研究はありますか？
・専門家の意見や調査結果で支える部分はありますか？

⚠️ **3. 主要リスクと打ち手（代替案、実験設計）**
リスクと対策について：
・どんなリスクや心配事がありそうでしょうか？
・うまくいかなかった場合の代替案は考えていますか？
・小さく試してみる方法はありそうですか？

「この方針の根拠を教えてください」「参考になる事例はご存知ですか？」「リスクへの備えはいかがですか？」といった形で、3つの観点での整理をサポートします。""",
            input_variables=["context", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは思考整理をサポートする壁打ち相手です。答えや結論は出さず、ユーザーのコンセプト検討の思考を整理し、気づきを促すことに徹してください。"),
            HumanMessage(content=prompt.format(
                context=context,
                user_input=user_input
            ))
        ]
        
        response = self.chat.invoke(messages)
        return response.content.strip()
    
    def _plan_agent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """施策立案エージェント（改良版）"""
        # 依存関係のコンテキストを構築
        context = self._build_context_for_step(state, "plan")
        
        prompt = PromptTemplate(
            template="""あなたは思考整理をサポートする壁打ち相手です。ユーザーが考えている計画を整理し、実現可能性を高めるためのサポートをします。答えは出しません。

これまでの検討内容:
{context}

ユーザーが考えていること: {user_input}

【あなたの役割】
- ユーザーの計画策定の思考を整理する
- 実現可能性の観点から気になる点を指摘する
- より具体化するための視点を提供する
- 判断や結論は出さず、ユーザー自身が考えを深められるよう導く

最終的には以下の3つがまとまると良いですね：

🚀 **1. 主な施策（3〜5個）の概要と狙い**
具体的な取り組みについて：
・メインでやりたい施策は何でしょうか？
・それぞれの施策の狙いや期待効果は？
・3〜5個程度に絞るとしたら、どれが重要でしょう？

👥 **2. 体制・役割分担・スケジュール**
実施の進め方について：
・誰がどんな役割を担うか決まっていますか？
・いつ頃から、どんなペースで進めたいでしょう？
・重要な節目やマイルストーンはいつ頃でしょう？

💰 **3. 概算コスト・効果見込み（根拠も）**
投資と効果について：
・ざっくりとした費用感はいかがでしょう？
・どの程度の効果を期待していますか？
・その見込みの根拠となるものはありますか？

「この施策の狙いを教えてください」「体制は現実的に組めそうですか？」「コストと効果の見込みはいかがですか？」といった形で、3つの観点での整理をサポートします。""",
            input_variables=["context", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは思考整理をサポートする壁打ち相手です。答えや結論は出さず、ユーザーの計画策定の思考を整理し、気づきを促すことに徹してください。"),
            HumanMessage(content=prompt.format(context=context, user_input=user_input))
        ]
        
        response = self.chat.invoke(messages)
        return response.content.strip()
    
    def _proposal_agent(self, user_input: str, state: FlexiblePolicyState) -> str:
        """資料作成エージェント（改良版）"""
        # 依存関係のコンテキストを構築
        context = self._build_context_for_step(state, "proposal")
        
        prompt = PromptTemplate(
            template="""あなたは思考整理をサポートする壁打ち相手です。ユーザーが考えている提案を整理し、説得力を高めるためのサポートをします。答えは出しません。

これまでの検討内容:
{context}

ユーザーが考えていること: {user_input}

【あなたの役割】
- ユーザーの提案作成の思考を整理する
- 説得力や論理性の観点から気になる点を指摘する
- より効果的に伝えるための視点を提供する
- 判断や結論は出さず、ユーザー自身が考えをまとめられるよう導く

最終的には以下の3つがまとまると良いですね：

📝 **1. 提案のサマリー（背景→課題→解決→効果→体制）**
提案の流れについて：
・背景から効果まで、ストーリーは繋がっていそうですか？
・相手にとって分かりやすい流れになっているでしょうか？
・体制の部分も含めて、現実的に見えそうですか？

💼 **2. 意思決定者の関心（費用対効果、リスク、責任分担）**
決定権者の視点について：
・投資に見合うリターンがありそうに見えるでしょうか？
・失敗時のリスクについて、どう説明しますか？
・責任分担は明確で、納得してもらえそうですか？

➡️ **3. 次のアクション（承認プロセス、関係者説明、PoC準備など）**
これからの進め方について：
・承認を得るために、どんな段取りが必要でしょう？
・関係者への説明で、特に気をつけたいことは？
・実証実験など、小さく始める準備はいかがですか？

「この説明で相手に伝わりそうですか？」「意思決定者が気にしそうなポイントは押さえていますか？」「次のステップは具体的でしょうか？」といった形で、3つの観点での整理をサポートします。""",
            input_variables=["context", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは思考整理をサポートする壁打ち相手です。答えや結論は出さず、ユーザーの提案作成の思考を整理し、気づきを促すことに徹してください。"),
            HumanMessage(content=prompt.format(context=context, user_input=user_input))
        ]
        
        response = self.chat.invoke(messages)
        return response.content.strip()
    
    def process_step_flexible(self, step: str, user_input: str, session_id: str, project_id: str = None) -> Dict[str, Any]:
        """柔軟なステップ処理（どのステップからでも開始可能）"""
        # セッション状態を取得
        state = self._get_session_state(session_id, project_id)
        
        # 指定されたステップを実行
        result = ""
        if step == "analysis":
            result = self._analysis_agent(user_input, state)
        elif step == "objective":
            result = self._objective_agent(user_input, state)
        elif step == "concept":
            result = self._concept_agent(user_input, state)
        elif step == "plan":
            result = self._plan_agent(user_input, state)
        elif step == "proposal":
            result = self._proposal_agent(user_input, state)
        else:
            return {"error": "指定されたステップが見つかりません。"}
        
        # セッション状態を更新
        self._update_session_state(session_id, step, result)
        
        # 更新された状態を取得
        updated_state = self._get_session_state(session_id)
        
        return {
            "result": result,
            "step": step,
            "session_id": session_id,
            "project_id": project_id,
            "full_state": {
                "analysis_result": updated_state["analysis_result"],
                "objective_result": updated_state["objective_result"],
                "concept_result": updated_state["concept_result"],
                "plan_result": updated_state["plan_result"],
                "proposal_result": updated_state["proposal_result"],
                "last_updated_step": updated_state["last_updated_step"],
                "step_timestamps": updated_state["step_timestamps"]
            }
        }
    
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