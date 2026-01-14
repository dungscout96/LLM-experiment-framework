#!/usr/bin/env python3
"""Agent runner script for agentic workflows."""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import MultiAgentOrchestrator, RAGPipeline, ReActAgent
from src.models import ModelFactory

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def run_react_agent(cfg: DictConfig, single_query: str = None) -> None:
    """Run ReAct agent interactively or with a single query.

    Args:
        cfg: Hydra configuration.
        single_query: Optional single query to run non-interactively.
    """
    console.print("\n[bold blue]ReAct Agent[/bold blue]")

    model = ModelFactory.create(cfg.model)
    agent = ReActAgent(cfg.agent, model=model)
    agent.setup()

    console.print(f"Tools: [green]{[t.name for t in agent._tools]}[/green]")

    # Single query mode
    if single_query:
        console.print(f"\n[bold blue]Query:[/bold blue] {single_query}\n")
        with console.status("[bold green]Agent thinking..."):
            result = agent.run(single_query)

        console.print(Panel(
            result["output"],
            title="[bold green]Response[/bold green]",
            border_style="green",
        ))

        if result.get("messages") and cfg.agent.get("verbose", False):
            console.print("\n[dim]Message history:[/dim]")
            for msg in result.get("messages", []):
                if hasattr(msg, "content"):
                    console.print(f"  {msg.__class__.__name__}: {msg.content[:200]}...")
        return

    # Interactive mode
    console.print("\nType 'quit' to exit, 'clear' to clear memory\n")

    while True:
        try:
            query = console.input("[bold blue]Query:[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            break

        if query.lower() == "quit":
            break
        if query.lower() == "clear":
            agent.clear_memory()
            console.print("[dim]Memory cleared[/dim]")
            continue

        with console.status("[bold green]Agent thinking..."):
            result = agent.run(query)

        console.print(Panel(
            result["output"],
            title="[bold green]Response[/bold green]",
            border_style="green",
        ))

        if result.get("messages") and cfg.agent.get("verbose", False):
            console.print("\n[dim]Message history:[/dim]")
            for msg in result.get("messages", []):
                if hasattr(msg, "content"):
                    console.print(f"  {msg.__class__.__name__}: {msg.content[:200]}...")


def run_rag_pipeline(cfg: DictConfig) -> None:
    """Run RAG pipeline interactively.

    Args:
        cfg: Hydra configuration.
    """
    console.print("\n[bold blue]RAG Pipeline[/bold blue]")

    model = ModelFactory.create(cfg.model)
    rag = RAGPipeline(cfg.agent, model=model)
    rag.setup()

    console.print("\nCommands:")
    console.print("  'ingest <path>' - Ingest documents from path")
    console.print("  'search <query>' - Similarity search without LLM")
    console.print("  'clear' - Clear vector store")
    console.print("  'quit' - Exit\n")

    while True:
        try:
            user_input = console.input("[bold blue]Query:[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "clear":
            rag.clear_vectorstore()
            console.print("[dim]Vector store cleared[/dim]")
            continue

        if user_input.lower().startswith("ingest "):
            path = user_input[7:].strip()
            try:
                count = rag.ingest_documents(path)
                console.print(f"[green]Ingested {count} chunks[/green]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            continue

        if user_input.lower().startswith("search "):
            query = user_input[7:].strip()
            results = rag.similarity_search(query)
            for i, result in enumerate(results, 1):
                console.print(f"\n[bold]Result {i}[/bold] (score: {result['score']:.4f})")
                console.print(result["content"][:200] + "...")
            continue

        with console.status("[bold green]Querying..."):
            result = rag.query(user_input)

        console.print(Panel(
            result["answer"],
            title="[bold green]Answer[/bold green]",
            border_style="green",
        ))

        if result.get("source_documents"):
            console.print("\n[dim]Sources:[/dim]")
            for i, doc in enumerate(result["source_documents"][:3], 1):
                console.print(f"  {i}. {doc['content'][:100]}...")


def run_multi_agent(cfg: DictConfig) -> None:
    """Run multi-agent system interactively.

    Args:
        cfg: Hydra configuration.
    """
    console.print("\n[bold blue]Multi-Agent System[/bold blue]")

    model = ModelFactory.create(cfg.model)
    orchestrator = MultiAgentOrchestrator(cfg.agent, model=model)
    orchestrator.setup()

    agent_names = list(orchestrator._agents.keys())
    console.print(f"Agents: [green]{agent_names}[/green]")
    console.print(f"Orchestration: [green]{cfg.agent.orchestration.type}[/green]")
    console.print("\nType 'quit' to exit\n")

    while True:
        try:
            task = console.input("[bold blue]Task:[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            break

        if task.lower() == "quit":
            break

        with console.status("[bold green]Agents working..."):
            result = orchestrator.run(task)

        console.print(Panel(
            result["output"],
            title="[bold green]Final Output[/bold green]",
            border_style="green",
        ))

        if result.get("messages") and cfg.agent.verbose:
            console.print("\n[dim]Agent conversation:[/dim]")
            for msg in result["messages"][-5:]:
                role = msg["role"]
                content = msg["content"][:200]
                console.print(f"  [{role}]: {content}...")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run agent with Hydra configuration.

    Args:
        cfg: Hydra configuration.
    """
    setup_logging(cfg.get("verbose", False))

    console.print("\n[bold blue]LLM Agent Runner[/bold blue]")
    console.print(f"Model: [green]{cfg.model.name}[/green]")
    console.print(f"Agent Type: [green]{cfg.agent.type}[/green]")

    agent_type = cfg.agent.type
    query = cfg.get("query", None)

    if agent_type == "react":
        run_react_agent(cfg, single_query=query)
    elif agent_type == "rag":
        if query:
            console.print(f"\n[bold blue]Query:[/bold blue] {query}\n")
            model = ModelFactory.create(cfg.model)
            rag = RAGPipeline(cfg.agent, model=model)
            rag.setup()
            with console.status("[bold green]Querying..."):
                result = rag.query(query)
            console.print(Panel(
                result["answer"],
                title="[bold green]Answer[/bold green]",
                border_style="green",
            ))
            if result.get("source_documents"):
                console.print("\n[dim]Sources:[/dim]")
                for i, doc in enumerate(result["source_documents"][:3], 1):
                    console.print(f"  {i}. {doc['content'][:100]}...")
        else:
            run_rag_pipeline(cfg)
    elif agent_type == "multi_agent":
        if query:
            console.print(f"\n[bold blue]Task:[/bold blue] {query}\n")
            model = ModelFactory.create(cfg.model)
            orchestrator = MultiAgentOrchestrator(cfg.agent, model=model)
            orchestrator.setup()
            with console.status("[bold green]Agents working..."):
                result = orchestrator.run(query)
            console.print(Panel(
                result["output"],
                title="[bold green]Final Output[/bold green]",
                border_style="green",
            ))
        else:
            run_multi_agent(cfg)
    else:
        console.print(f"[red]Unknown agent type: {agent_type}[/red]")
        console.print("Available types: react, rag, multi_agent")
        return

    console.print("\n[bold green]Session ended.[/bold green]")


if __name__ == "__main__":
    main()
