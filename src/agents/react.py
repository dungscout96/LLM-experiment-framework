"""ReAct (Reasoning + Acting) agent implementation using LangGraph."""

import logging
from typing import Any

from langchain_core.language_models import BaseLLM
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from omegaconf import DictConfig

from ..models import BaseModel

logger = logging.getLogger(__name__)


class ReActAgent:
    """ReAct agent with configurable tools using LangGraph."""

    def __init__(
        self,
        cfg: DictConfig,
        model: BaseModel | None = None,
        llm: BaseLLM | None = None,
    ):
        """Initialize ReAct agent.

        Args:
            cfg: Agent configuration from Hydra.
            model: Optional BaseModel instance.
            llm: Optional LangChain LLM (overrides model).
        """
        self.cfg = cfg
        self.agent_cfg = cfg.agent
        self._model = model
        self._llm = llm
        self._tools: list = []
        self._agent = None

    def setup(self) -> None:
        """Set up the agent with tools."""
        if self._llm is None:
            self._llm = self._create_llm()

        self._tools = self._create_tools()
        self._agent = create_react_agent(self._llm, self._tools)

        logger.info(
            f"ReAct agent initialized with {len(self._tools)} tools: "
            f"{[t.name for t in self._tools]}"
        )

    def _create_llm(self) -> BaseLLM:
        """Create LangChain LLM from model."""
        if self._model is None:
            raise ValueError("Either model or llm must be provided")

        self._model.load()

        # For API models, return the client directly (it's already a ChatModel)
        if hasattr(self._model, '_client') and self._model._client is not None:
            # Check if it's a LangChain ChatModel
            if hasattr(self._model._client, 'bind_tools'):
                return self._model._client

        # For local models, wrap them
        return LangChainModelWrapper(self._model)

    def _create_tools(self) -> list:
        """Create tools based on configuration."""
        tools = []
        tools_cfg = self.cfg.get("tools", [])

        for tool_cfg in tools_cfg:
            if not tool_cfg.get("enabled", True):
                continue

            if tool_cfg.name == "calculator":
                tool = self._create_calculator_tool()
                if tool:
                    tools.append(tool)
            elif tool_cfg.name == "search":
                tool = self._create_search_tool()
                if tool:
                    tools.append(tool)

        return tools

    def _create_calculator_tool(self):
        """Create calculator tool."""
        @tool
        def calculator(expression: str) -> str:
            """Perform mathematical calculations. Input should be a mathematical expression like '2 + 2' or '15 * 23'."""
            try:
                allowed_chars = set("0123456789+-*/.() ")
                if not all(c in allowed_chars for c in expression):
                    return "Error: Invalid characters in expression"
                result = eval(expression)
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
        return calculator

    def _create_search_tool(self):
        """Create web search tool."""
        try:
            from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
            search = DuckDuckGoSearchAPIWrapper()

            @tool
            def web_search(query: str) -> str:
                """Search the web for current information. Use this when you need to find up-to-date information."""
                return search.run(query)
            return web_search
        except ImportError:
            logger.warning("DuckDuckGo search not available")
            return None

    def run(self, query: str) -> dict[str, Any]:
        """Run the agent on a query.

        Args:
            query: User query.

        Returns:
            Agent response with output and messages.
        """
        if self._agent is None:
            self.setup()

        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            messages = result.get("messages", [])

            # Get the last AI message as the output
            output = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    output = msg.content
                    break

            return {
                "output": output,
                "messages": messages,
            }
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {"output": f"Error: {str(e)}", "messages": []}

    def clear_memory(self) -> None:
        """Clear conversation memory (recreate agent)."""
        if self._llm is not None:
            self._agent = create_react_agent(self._llm, self._tools)


class LangChainModelWrapper(BaseLLM):
    """Wrapper to use BaseModel with LangChain."""

    model: Any = None

    def __init__(self, model: BaseModel, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
        return self.model.generate(prompt, **kwargs)

    def _generate(self, prompts, stop=None, **kwargs):
        from langchain_core.outputs import Generation, LLMResult
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model.name}
