#!/usr/bin/env python3
"""Autonomous Decision-Making Agent — Entry Point.

This is the main entry point for the autonomous agent.  It initializes all
components, starts the PCA loop, and handles CLI arguments.

Usage:
    python main.py                  # Run with default config
    python main.py --dry-run        # Run without executing actions
    python main.py --verbose        # Print full reasoning traces
    python main.py --config path    # Use a custom config file
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.action import ActionLayer
from core.cognition import CognitionLayer
from core.loop import PCALoop
from core.perception import PerceptionLayer
from core.state import SharedState
from environment.simulated import SimulatedEnvironment
from llm.client import LLMClient
from memory.memory_manager import MemoryManager
from utils.config_loader import ConfigLoader
from utils.logger import AgentLogger

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.markdown import Markdown


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous Decision-Making Agent with PCA Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --dry-run --verbose\n"
            "  python main.py --config my_config.yaml\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run cognition but do not execute actions",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full reasoning traces to console",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max_iterations from config",
    )
    return parser.parse_args()


def print_banner(console: Console, config: dict) -> None:
    """Print an attractive startup banner using Rich."""
    agent_name = config.get("agent", {}).get("name", "AutonomousAgent")
    model = config.get("llm", {}).get("model", "unknown")
    base_url = config.get("llm", {}).get("base_url", "unknown")
    scenario = config.get("environment", {}).get("scenario", "unknown")
    max_iter = config.get("agent", {}).get("max_iterations", "?")
    interval = config.get("agent", {}).get("loop_interval_seconds", "?")
    confidence = config.get("agent", {}).get("confidence_threshold", "?")

    banner_text = Text()
    banner_text.append("AUTONOMOUS DECISION-MAKING AGENT\n", style="bold cyan")
    banner_text.append("Perception → Cognition → Action Loop\n\n", style="dim")
    banner_text.append("Based on ", style="white")

    console.print(Panel(banner_text, border_style="cyan", padding=(1, 2)))

    # Configuration table
    table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Agent Name", agent_name)
    table.add_row("LLM Model", model)
    table.add_row("LLM Endpoint", base_url)
    table.add_row("Scenario", scenario)
    table.add_row("Max Iterations", str(max_iter))
    table.add_row("Loop Interval", f"{interval}s")
    table.add_row("Confidence Threshold", str(confidence))

    console.print(table)
    console.print()


def print_cycle_header(
    console: Console,
    iteration: int,
    max_iterations: int,
    decision,
    verbose: bool,
) -> None:
    """Print a formatted header for each PCA cycle."""
    console.print()
    console.rule(
        f"[bold blue]Cycle {iteration}/{max_iterations}[/bold blue]",
        style="blue",
    )

    # Decision summary
    if decision:
        action_color = "green" if decision.confidence >= 0.6 else "yellow"
        console.print(
            f"  [bold]Decision:[/bold] [{action_color}]"
            f"{decision.selected_action}[/{action_color}] "
            f"(confidence: {decision.confidence:.2f})"
        )
        if verbose and decision.reasoning:
            console.print(
                Panel(
                    decision.reasoning[:500],
                    title="Reasoning",
                    border_style="dim",
                    padding=(0, 1),
                )
            )


def main() -> None:
    """Main entry point for the autonomous agent."""
    console = Console()
    args = parse_args()

    # ── Load Configuration ────────────────────────────────────────
    try:
        config = ConfigLoader.load(args.config)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        sys.exit(1)

    # Apply CLI overrides
    if args.max_iterations is not None:
        config["agent"]["max_iterations"] = args.max_iterations

    # ── Print Banner ──────────────────────────────────────────────
    print_banner(console, config)

    # ── Initialize LLM Client ─────────────────────────────────────
    console.print("[bold]Initializing LLM client...[/bold]")
    try:
        llm_client = LLMClient(config["llm"])
        console.print(
            f"  [green]OK[/green] — Connected to {config['llm']['base_url']} "
            f"(model: {config['llm']['model']})"
        )
    except Exception as exc:
        console.print(f"  [red]FAILED[/red] — {exc}")
        console.print(
            "\n[yellow]Make sure LM Studio is running with a model loaded.[/yellow]"
        )
        console.print(
            "[yellow]Start LM Studio, load a model (Qwen 3.5 or Gemma 4), "
            "and ensure the server is started on the configured port.[/yellow]"
        )
        sys.exit(1)

    # ── Initialize Logger ─────────────────────────────────────────
    logger = AgentLogger(
        name=config["agent"]["name"],
        log_file=config["logging"]["file"],
        level=config["logging"]["level"],
    )

    # ── Initialize Memory ─────────────────────────────────────────
    console.print("[bold]Initializing memory systems...[/bold]")
    memory = MemoryManager(config)
    console.print(
        f"  [green]OK[/green] — Working memory capacity: "
        f"{memory.working.capacity}, "
        f"Episodic memory: {memory.episodic.file_path}"
    )

    # ── Initialize Environment ────────────────────────────────────
    console.print("[bold]Initializing environment...[/bold]")
    environment = SimulatedEnvironment(config)
    console.print(
        f"  [green]OK[/green] — Scenario: "
        f"{config['environment']['scenario']}"
    )

    # ── Initialize Shared State ───────────────────────────────────
    state = SharedState()

    # ── Initialize PCA Layers ─────────────────────────────────────
    console.print("[bold]Initializing PCA layers...[/bold]")

    perception = PerceptionLayer(
        environment=environment,
        memory=memory,
        state=state,
        config=config,
        logger=logger,
    )

    cognition = CognitionLayer(
        llm=llm_client,
        memory=memory,
        state=state,
        config=config,
        logger=logger,
    )
    cognition.set_environment_ref(environment)

    action = ActionLayer(
        environment=environment,
        state=state,
        config=config,
        logger=logger,
        dry_run=args.dry_run,
    )

    console.print("  [green]OK[/green] — Perception, Cognition, Action layers ready")
    if args.dry_run:
        console.print("  [yellow]DRY RUN MODE[/yellow] — Actions will not be executed")

    # ── Create and Start the Loop ─────────────────────────────────
    console.print()
    console.print("[bold green]Starting PCA Loop...[/bold green]")
    console.print("[dim]Press Ctrl+C to stop gracefully.[/dim]")
    console.print()

    loop = PCALoop(
        perception=perception,
        cognition=cognition,
        action=action,
        state=state,
        config=config,
        logger=logger,
    )

    # Monkey-patch the loop to add Rich console output per cycle
    original_run_cycle = loop._run_cycle

    def enhanced_run_cycle(iteration: int) -> None:
        """Run a cycle with Rich console output."""
        original_run_cycle(iteration)
        decision = state.get_decision()
        print_cycle_header(
            console, iteration, loop.max_iterations, decision, args.verbose
        )

        # Show observations
        observations = state.get_observations()
        if observations:
            latest = observations[-1] if observations else None
            if latest:
                console.print(
                    f"  [bold]Latest observation:[/bold] {latest.content[:120]}"
                )

        # Show action result
        feedback = state.get_feedback()
        if feedback:
            icon = "[green]OK[/green]" if feedback.success else "[red]FAIL[/red]"
            console.print(
                f"  [bold]Feedback:[/bold] {icon} — {feedback.message[:120]}"
            )

        # Show metrics
        metrics = state.get_metrics()
        console.print(
            f"  [dim]Decisions: {metrics['decisions_made']} | "
            f"Avg Confidence: {metrics.get('avg_confidence', 0):.2f} | "
            f"LLM Calls: {llm_client.call_count} | "
            f"Tokens: {llm_client.total_tokens_used}[/dim]"
        )

    loop._run_cycle = enhanced_run_cycle

    # Run the loop
    try:
        metrics = loop.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        metrics = state.get_metrics()

    # ── Print Final Summary ───────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Session Complete[/bold cyan]")

    summary_table = Table(title="Final Metrics", show_header=True, header_style="bold")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    start_time = metrics.get("start_time", time.time())
    elapsed = time.time() - start_time if start_time else 0.0

    summary_table.add_row("Total Iterations", str(metrics.get("iteration", 0)))
    summary_table.add_row("Decisions Made", str(metrics.get("decisions_made", 0)))
    summary_table.add_row("Actions Executed", str(metrics.get("actions_executed", 0)))
    summary_table.add_row("Avg Confidence", f"{metrics.get('avg_confidence', 0):.2f}")
    summary_table.add_row("Errors", str(metrics.get("errors", 0)))
    summary_table.add_row("Elapsed Time", f"{elapsed:.1f}s")
    summary_table.add_row("LLM Calls", str(llm_client.call_count))
    summary_table.add_row("Total Tokens", str(llm_client.total_tokens_used))
    summary_table.add_row(
        "Action Distribution", str(metrics.get("action_distribution", {}))
    )

    console.print(summary_table)

    # Environment summary
    env_summary = environment.get_state_summary()
    scenario_summary = env_summary.get("scenario_summary", {})
    console.print(
        f"\n[bold]Environment:[/bold] "
        f"{scenario_summary.get('completed', 0)}/{scenario_summary.get('total_tasks', 0)} "
        f"tasks completed"
    )


if __name__ == "__main__":
    main()
