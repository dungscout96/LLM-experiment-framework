"""Multi-agent orchestration using LangGraph."""

import logging
from typing import Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLLM
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from omegaconf import DictConfig

from ..models import BaseModel
from .react import LangChainModelWrapper

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State shared between agents."""

    messages: list[BaseMessage]
    current_agent: str
    task: str
    context: dict[str, Any]
    final_output: str


class MultiAgentOrchestrator:
    """Multi-agent system with configurable orchestration."""

    def __init__(
        self,
        cfg: DictConfig,
        model: BaseModel | None = None,
        llm: BaseLLM | None = None,
    ):
        """Initialize multi-agent orchestrator.

        Args:
            cfg: Agent configuration from Hydra.
            model: Optional BaseModel instance (shared by all agents).
            llm: Optional LangChain LLM (overrides model).
        """
        self.cfg = cfg
        self._model = model
        self._llm = llm
        self._agents: dict[str, AgentNode] = {}
        self._graph: StateGraph | None = None
        self._app = None

    def setup(self) -> None:
        """Set up agents and orchestration graph."""
        if self._llm is None:
            self._llm = self._create_llm()

        self._agents = self._create_agents()
        self._graph = self._create_graph()
        self._app = self._graph.compile()

        logger.info(
            f"Multi-agent system initialized with {len(self._agents)} agents: "
            f"{list(self._agents.keys())}"
        )

    def _create_llm(self) -> BaseLLM:
        """Create LangChain LLM from model."""
        if self._model is None:
            raise ValueError("Either model or llm must be provided")

        self._model.load()
        return LangChainModelWrapper(self._model)

    def _create_agents(self) -> dict[str, "AgentNode"]:
        """Create agent nodes from configuration."""
        agents = {}

        for agent_cfg in self.cfg.agents:
            agent = AgentNode(
                name=agent_cfg.name,
                role=agent_cfg.role,
                goal=agent_cfg.goal,
                backstory=agent_cfg.backstory,
                llm=self._llm,
                verbose=agent_cfg.get("verbose", True),
            )
            agents[agent_cfg.name] = agent

        return agents

    def _create_graph(self) -> StateGraph:
        """Create the orchestration graph."""
        orchestration_type = self.cfg.orchestration.type

        if orchestration_type == "sequential":
            return self._create_sequential_graph()
        elif orchestration_type == "hierarchical":
            return self._create_hierarchical_graph()
        else:
            return self._create_collaborative_graph()

    def _create_sequential_graph(self) -> StateGraph:
        """Create a sequential execution graph."""
        graph = StateGraph(AgentState)

        agent_names = list(self._agents.keys())

        for name, agent in self._agents.items():
            graph.add_node(name, agent.run)

        graph.set_entry_point(agent_names[0])

        for i in range(len(agent_names) - 1):
            graph.add_edge(agent_names[i], agent_names[i + 1])

        graph.add_edge(agent_names[-1], END)

        return graph

    def _create_hierarchical_graph(self) -> StateGraph:
        """Create a hierarchical graph with a manager agent."""
        graph = StateGraph(AgentState)

        manager = ManagerAgent(
            llm=self._llm,
            agents=list(self._agents.keys()),
        )

        graph.add_node("manager", manager.run)

        for name, agent in self._agents.items():
            graph.add_node(name, agent.run)

        graph.set_entry_point("manager")

        def route_from_manager(state: AgentState) -> str:
            next_agent = state.get("current_agent", "")
            if next_agent in self._agents:
                return next_agent
            return END

        graph.add_conditional_edges(
            "manager",
            route_from_manager,
            {name: name for name in self._agents} | {END: END},
        )

        for name in self._agents:
            graph.add_edge(name, "manager")

        return graph

    def _create_collaborative_graph(self) -> StateGraph:
        """Create a collaborative graph where agents can interact."""
        graph = StateGraph(AgentState)

        for name, agent in self._agents.items():
            graph.add_node(name, agent.run)

        router = CollaborativeRouter(
            llm=self._llm,
            agents=list(self._agents.keys()),
        )
        graph.add_node("router", router.run)

        graph.set_entry_point("router")

        for name in self._agents:
            graph.add_edge(name, "router")

        def route_decision(state: AgentState) -> str:
            next_agent = state.get("current_agent", "")
            if next_agent == "end":
                return END
            if next_agent in self._agents:
                return next_agent
            return END

        graph.add_conditional_edges(
            "router",
            route_decision,
            {name: name for name in self._agents} | {END: END},
        )

        return graph

    def run(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the multi-agent system on a task.

        Args:
            task: Task description.
            context: Optional initial context.

        Returns:
            Final output and conversation history.
        """
        if self._app is None:
            self.setup()

        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "current_agent": list(self._agents.keys())[0],
            "task": task,
            "context": context or {},
            "final_output": "",
        }

        try:
            result = self._app.invoke(initial_state)
            return {
                "output": result.get("final_output", ""),
                "messages": [
                    {
                        "role": "assistant" if isinstance(m, AIMessage) else "user",
                        "content": m.content,
                    }
                    for m in result.get("messages", [])
                ],
                "context": result.get("context", {}),
            }
        except Exception as e:
            logger.error(f"Multi-agent error: {e}")
            return {"output": f"Error: {str(e)}", "messages": [], "context": {}}


class AgentNode:
    """Individual agent node in the graph."""

    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        llm: BaseLLM,
        verbose: bool = True,
    ):
        """Initialize agent node.

        Args:
            name: Agent name.
            role: Agent role description.
            goal: Agent's goal.
            backstory: Agent backstory for context.
            llm: Language model.
            verbose: Whether to log verbose output.
        """
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm
        self.verbose = verbose

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.role}.
Your goal: {self.goal}
Background: {self.backstory}

Complete your assigned task based on the conversation history and any provided context."""),
            ("human", "{input}"),
        ])

    def run(self, state: AgentState) -> AgentState:
        """Execute the agent's task.

        Args:
            state: Current state.

        Returns:
            Updated state.
        """
        task = state["task"]
        context = state.get("context", {})

        context_str = "\n".join(f"{k}: {v}" for k, v in context.items()) if context else ""
        input_text = f"Task: {task}\n\nContext:\n{context_str}" if context_str else f"Task: {task}"

        chain = self.prompt | self.llm

        try:
            response = chain.invoke({"input": input_text})
            response_text = response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"Agent {self.name} error: {e}")
            response_text = f"Error: {str(e)}"

        if self.verbose:
            logger.info(f"[{self.name}] {response_text[:200]}...")

        new_messages = state["messages"] + [
            AIMessage(content=f"[{self.name}]: {response_text}")
        ]

        new_context = state["context"].copy()
        new_context[f"{self.name}_output"] = response_text

        return {
            **state,
            "messages": new_messages,
            "context": new_context,
            "final_output": response_text,
        }


class ManagerAgent:
    """Manager agent for hierarchical orchestration."""

    def __init__(self, llm: BaseLLM, agents: list[str]):
        """Initialize manager agent.

        Args:
            llm: Language model.
            agents: List of available agent names.
        """
        self.llm = llm
        self.agents = agents

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a manager coordinating a team of agents: {', '.join(agents)}.
Decide which agent should handle the next part of the task, or say 'done' if the task is complete.
Respond with just the agent name or 'done'."""),
            ("human", "{input}"),
        ])

    def run(self, state: AgentState) -> AgentState:
        """Decide next agent.

        Args:
            state: Current state.

        Returns:
            Updated state with next agent.
        """
        context_summary = "\n".join(
            f"{k}: {v[:200]}..." if len(str(v)) > 200 else f"{k}: {v}"
            for k, v in state["context"].items()
        )

        input_text = f"Task: {state['task']}\n\nProgress:\n{context_summary}"

        chain = self.prompt | self.llm

        try:
            response = chain.invoke({"input": input_text})
            decision = response.strip().lower() if isinstance(response, str) else str(response).strip().lower()
        except Exception as e:
            logger.error(f"Manager error: {e}")
            decision = "done"

        if decision == "done" or decision not in [a.lower() for a in self.agents]:
            next_agent = END
        else:
            next_agent = next(a for a in self.agents if a.lower() == decision)

        return {**state, "current_agent": next_agent}


class CollaborativeRouter:
    """Router for collaborative agent interaction."""

    def __init__(self, llm: BaseLLM, agents: list[str]):
        """Initialize router.

        Args:
            llm: Language model.
            agents: List of available agent names.
        """
        self.llm = llm
        self.agents = agents
        self.iteration = 0
        self.max_iterations = 5

    def run(self, state: AgentState) -> AgentState:
        """Route to next agent or end.

        Args:
            state: Current state.

        Returns:
            Updated state with next agent decision.
        """
        self.iteration += 1

        if self.iteration > self.max_iterations:
            return {**state, "current_agent": "end"}

        if len(state["context"]) >= len(self.agents):
            return {**state, "current_agent": "end"}

        remaining = [a for a in self.agents if f"{a}_output" not in state["context"]]
        if not remaining:
            return {**state, "current_agent": "end"}

        return {**state, "current_agent": remaining[0]}
