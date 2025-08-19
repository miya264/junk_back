"""
政策立案支援のためのAIエージェント実装
LangChainとLangGraphを使用したステップ別処理
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing import Dict, List, Optional, TypedDict
import json
from datetime import datetime

class PolicyState(TypedDict):
    """政策立案プロセスの状態を管理"""
    current_step: str
    user_input: str
    analysis_result: Optional[str]
    objective_result: Optional[str]
    concept_result: Optional[str]
    plan_result: Optional[str]
    proposal_result: Optional[str]
    context: Dict
    rag_documents: Optional[List[Document]]

class PolicyAgentSystem:
    """政策立案支援AIエージェントシステム"""
    
    def __init__(self, chat_model: ChatOpenAI, embedding_model, index):
        self.chat = chat_model
        self.embedding_model = embedding_model
        self.index = index
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """LangGraphでワークフローを構築"""
        workflow = StateGraph(PolicyState)
        
        # ノードを追加
        workflow.add_node("analysis", self._analysis_agent)
        workflow.add_node("objective", self._objective_agent)
        workflow.add_node("concept", self._concept_agent)
        workflow.add_node("plan", self._plan_agent)
        workflow.add_node("proposal", self._proposal_agent)
        
        # エントリーポイントを設定
        workflow.set_entry_point("analysis")
        
        # エッジを追加（条件分岐）
        workflow.add_conditional_edges(
            "analysis",
            self._route_next_step,
            {
                "objective": "objective",
                "concept": "concept", 
                "plan": "plan",
                "proposal": "proposal",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "objective",
            self._route_next_step,
            {
                "analysis": "analysis",
                "concept": "concept",
                "plan": "plan", 
                "proposal": "proposal",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "concept",
            self._route_next_step,
            {
                "analysis": "analysis",
                "objective": "objective",
                "plan": "plan",
                "proposal": "proposal", 
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "plan",
            self._route_next_step,
            {
                "analysis": "analysis",
                "objective": "objective",
                "concept": "concept",
                "proposal": "proposal",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "proposal",
            self._route_next_step,
            {
                "analysis": "analysis", 
                "objective": "objective",
                "concept": "concept",
                "plan": "plan",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _route_next_step(self, state: PolicyState) -> str:
        """次のステップを決定"""
        return state.get("current_step", "end")
    
    def _get_rag_context(self, query: str) -> List[Document]:
        """RAG検索でコンテキストを取得"""
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
    
    def _analysis_agent(self, state: PolicyState) -> PolicyState:
        """現状分析・課題整理エージェント"""
        user_input = state["user_input"]
        
        # RAG検索でコンテキストを取得
        rag_docs = self._get_rag_context(f"現状分析 課題 {user_input}")
        context_text = "\\n\\n".join([doc.page_content for doc in rag_docs[:3]])
        
        prompt = PromptTemplate(
            template="""あなたは政策分析の専門家です。以下の情報をもとに現状分析と課題整理を行ってください。

参考資料:
{context}

ユーザーの質問: {user_input}

以下の観点で分析してください：
1. 現状の把握（定量・定性データの整理）
2. 主要課題の特定
3. 課題の根本原因分析
4. 優先度の評価
5. 関連するステークホルダーの整理

分析結果:""",
            input_variables=["context", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは政策分析の専門家です。論理的で構造化された分析を提供してください。"),
            HumanMessage(content=prompt.format(context=context_text, user_input=user_input))
        ]
        
        response = self.chat.invoke(messages)
        
        state["analysis_result"] = response.content
        state["rag_documents"] = rag_docs
        return state
    
    def _objective_agent(self, state: PolicyState) -> PolicyState:
        """目的整理エージェント"""
        user_input = state["user_input"]
        analysis_result = state.get("analysis_result", "")
        
        prompt = PromptTemplate(
            template="""あなたは政策企画の専門家です。現状分析をもとに政策の目的を明確化してください。

現状分析結果:
{analysis_result}

ユーザーの要望: {user_input}

以下の観点で目的を整理してください：
1. 最終的に達成したいゴール（具体的な将来像）
2. 測定可能な成果指標（KPI）の設定
3. 達成期限の設定
4. 前提条件・制約条件の整理
5. 成功要因の特定

目的整理結果:""",
            input_variables=["analysis_result", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは政策企画の専門家です。実現可能で測定可能な目的設定を行ってください。"),
            HumanMessage(content=prompt.format(analysis_result=analysis_result, user_input=user_input))
        ]
        
        response = self.chat.invoke(messages)
        state["objective_result"] = response.content
        return state
    
    def _concept_agent(self, state: PolicyState) -> PolicyState:
        """コンセプト策定エージェント"""
        user_input = state["user_input"]
        analysis_result = state.get("analysis_result", "")
        objective_result = state.get("objective_result", "")
        
        # RAG検索で関連する政策事例を取得
        rag_docs = self._get_rag_context(f"政策 コンセプト 事例 {user_input}")
        context_text = "\\n\\n".join([doc.page_content for doc in rag_docs[:3]])
        
        prompt = PromptTemplate(
            template="""あなたは政策デザインの専門家です。分析と目的をもとに政策コンセプトを策定してください。

現状分析:
{analysis_result}

設定目的:
{objective_result}

参考事例:
{context}

ユーザーの要望: {user_input}

以下の観点でコンセプトを策定してください：
1. 基本方針（どんな価値を誰にどう提供するか）
2. 政策の独自性・差別化要因
3. 理論的根拠・エビデンス
4. 想定される効果・インパクト
5. 主要リスクと対策

コンセプト策定結果:""",
            input_variables=["analysis_result", "objective_result", "context", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは政策デザインの専門家です。革新的で実現可能なコンセプトを提案してください。"),
            HumanMessage(content=prompt.format(
                analysis_result=analysis_result,
                objective_result=objective_result,
                context=context_text,
                user_input=user_input
            ))
        ]
        
        response = self.chat.invoke(messages)
        state["concept_result"] = response.content
        return state
    
    def _plan_agent(self, state: PolicyState) -> PolicyState:
        """施策立案エージェント"""
        user_input = state["user_input"]
        analysis_result = state.get("analysis_result", "")
        objective_result = state.get("objective_result", "")
        concept_result = state.get("concept_result", "")
        
        prompt = PromptTemplate(
            template="""あなたは政策実行の専門家です。これまでの検討をもとに具体的な施策を立案してください。

現状分析:
{analysis_result}

設定目的:
{objective_result}

政策コンセプト:
{concept_result}

ユーザーの要望: {user_input}

以下の観点で施策を立案してください：
1. 主要施策の具体的内容（3-5個程度）
2. 実施体制・役割分担
3. 実施スケジュール・ロードマップ
4. 必要な予算・リソース
5. 評価・モニタリング方法
6. リスク管理・緊急時対応

施策立案結果:""",
            input_variables=["analysis_result", "objective_result", "concept_result", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは政策実行の専門家です。実現可能で効果的な施策を立案してください。"),
            HumanMessage(content=prompt.format(
                analysis_result=analysis_result,
                objective_result=objective_result,
                concept_result=concept_result,
                user_input=user_input
            ))
        ]
        
        response = self.chat.invoke(messages)
        state["plan_result"] = response.content
        return state
    
    def _proposal_agent(self, state: PolicyState) -> PolicyState:
        """資料作成エージェント"""
        user_input = state["user_input"]
        analysis_result = state.get("analysis_result", "")
        objective_result = state.get("objective_result", "")
        concept_result = state.get("concept_result", "")
        plan_result = state.get("plan_result", "")
        
        prompt = PromptTemplate(
            template="""あなたは政策提案書作成の専門家です。これまでの検討をもとに意思決定者向けの提案書を作成してください。

現状分析:
{analysis_result}

設定目的:
{objective_result}

政策コンセプト:
{concept_result}

施策案:
{plan_result}

ユーザーの要望: {user_input}

以下の構成で提案書を作成してください：
1. エグゼクティブサマリー
2. 背景・課題認識
3. 政策の目的・期待効果
4. 政策コンセプト・基本方針
5. 具体的施策・実施計画
6. 予算・体制
7. 想定リスク・対策
8. 次のアクション

提案書作成結果:""",
            input_variables=["analysis_result", "objective_result", "concept_result", "plan_result", "user_input"]
        )
        
        messages = [
            SystemMessage(content="あなたは政策提案書作成の専門家です。説得力があり理解しやすい提案書を作成してください。"),
            HumanMessage(content=prompt.format(
                analysis_result=analysis_result,
                objective_result=objective_result,
                concept_result=concept_result,
                plan_result=plan_result,
                user_input=user_input
            ))
        ]
        
        response = self.chat.invoke(messages)
        state["proposal_result"] = response.content
        return state
    
    def process_step(self, step: str, user_input: str, context: Dict = None) -> str:
        """指定されたステップを実行"""
        initial_state: PolicyState = {
            "current_step": "end",  # 単一ステップ実行なので終了フラグ
            "user_input": user_input,
            "analysis_result": context.get("analysis_result") if context else None,
            "objective_result": context.get("objective_result") if context else None,
            "concept_result": context.get("concept_result") if context else None,
            "plan_result": context.get("plan_result") if context else None,
            "proposal_result": context.get("proposal_result") if context else None,
            "context": context or {},
            "rag_documents": None
        }
        
        # 指定されたステップを直接実行
        if step == "analysis":
            result_state = self._analysis_agent(initial_state)
            return result_state["analysis_result"]
        elif step == "objective":
            result_state = self._objective_agent(initial_state)
            return result_state["objective_result"]
        elif step == "concept":
            result_state = self._concept_agent(initial_state)
            return result_state["concept_result"]
        elif step == "plan":
            result_state = self._plan_agent(initial_state)
            return result_state["plan_result"]
        elif step == "proposal":
            result_state = self._proposal_agent(initial_state)
            return result_state["proposal_result"]
        else:
            return "指定されたステップが見つかりません。"