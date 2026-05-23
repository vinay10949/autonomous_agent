"""Task Orchestration Scenario — agent manages tasks with dependencies.

This is the default simulated scenario.  Tasks arrive over time with
priorities, dependencies, and deadlines.  The agent must assess which
tasks to start, respect dependency ordering, manage priorities, and
handle deadline warnings and resource constraints.

Events include:
- task_arrival: A new task has been added to the queue.
- deadline_warning: A task deadline is approaching.
- task_completed: A previously started task has finished.
- resource_alert: A resource constraint has changed.
- dependency_resolved: A blocking dependency has been completed.

The agent can take these actions:
- start_task(task_id): Begin working on a task.
- defer_task(task_id): Postpone a task for later.
- request_info(task_id): Request more information about a task.
- escalate_task(task_id): Escalate a task to a human.
- respond(message): Send a general response or status update.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from models.action import Action


class Task:
    """Represents a task in the orchestration scenario.

    Attributes:
        task_id: Unique task identifier (e.g. 'T-001').
        description: Human-readable task description.
        priority: Priority level (1=lowest, 5=highest).
        dependencies: List of task IDs that must be completed first.
        deadline: When the task is due.
        status: Current status — 'pending', 'in_progress', 'completed', 'deferred', 'escalated'.
        assigned_tick: The tick when the task was assigned.
    """

    def __init__(
        self,
        task_id: str,
        description: str,
        priority: int = 3,
        dependencies: list[str] | None = None,
        deadline: datetime | None = None,
        assigned_tick: int = 0,
    ) -> None:
        self.task_id = task_id
        self.description = description
        self.priority = max(1, min(5, priority))
        self.dependencies = dependencies or []
        self.deadline = deadline or datetime.now() + timedelta(hours=24)
        self.status: str = "pending"
        self.assigned_tick = assigned_tick

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "deadline": self.deadline.isoformat(),
            "status": self.status,
        }


# Pre-defined task templates for realistic simulation
TASK_TEMPLATES: list[dict[str, Any]] = [
    {
        "description": "Deploy authentication service to staging",
        "priority": 5,
        "dependencies": [],
    },
    {
        "description": "Write integration tests for payment module",
        "priority": 4,
        "dependencies": [],
    },
    {
        "description": "Optimize database query performance",
        "priority": 3,
        "dependencies": [],
    },
    {
        "description": "Update API documentation for v2.1",
        "priority": 2,
        "dependencies": [],
    },
    {
        "description": "Configure CI/CD pipeline for new microservice",
        "priority": 4,
        "dependencies": [],
    },
    {
        "description": "Review and merge PR #247 (security patch)",
        "priority": 5,
        "dependencies": [],
    },
    {
        "description": "Set up monitoring alerts for production cluster",
        "priority": 3,
        "dependencies": [],
    },
    {
        "description": "Refactor user profile data model",
        "priority": 2,
        "dependencies": [],
    },
    {
        "description": "Implement rate limiting on public API endpoints",
        "priority": 4,
        "dependencies": [],
    },
    {
        "description": "Migrate legacy storage to cloud blob storage",
        "priority": 3,
        "dependencies": [],
    },
]


class TaskOrchestrationScenario:
    """Simulated task orchestration environment.

    Generates tasks over time with varying priorities and dependencies.
    Responds to agent actions with realistic outcomes including success,
    failure, and partial completion.

    Attributes:
        tasks: Dictionary of all tasks by task_id.
        completed_ids: Set of completed task IDs.
        _task_counter: Auto-incrementing counter for task IDs.
        _max_ticks: Maximum number of ticks before the scenario ends.
        _arrival_schedule: Ticks at which new tasks arrive.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the scenario.

        Args:
            config: Full agent configuration (used for future extensibility).
        """
        self.tasks: dict[str, Task] = {}
        self.completed_ids: set[str] = set()
        self._task_counter: int = 0
        self._max_ticks: int = config.get("agent", {}).get("max_iterations", 50)
        self._arrival_schedule: list[int] = []
        self._in_progress_task: str | None = None
        self._in_progress_remaining_ticks: int = 0

        self._generate_arrival_schedule()

    def _generate_arrival_schedule(self) -> None:
        """Determine at which ticks new tasks will arrive."""
        # Tasks arrive in batches, roughly every 3-5 ticks
        tick = 1
        while tick < self._max_ticks:
            self._arrival_schedule.append(tick)
            tick += random.randint(3, 6)

    def _create_task(self, tick: int) -> Task:
        """Create a new task from the template pool."""
        self._task_counter += 1
        template = random.choice(TASK_TEMPLATES)
        task_id = f"T-{self._task_counter:03d}"

        # Some tasks depend on previously completed or pending tasks
        dependencies: list[str] = []
        if self.completed_ids and random.random() < 0.4:
            dep_choices = list(self.completed_ids | set(self.tasks.keys()))
            if dep_choices:
                dep_id = random.choice(dep_choices)
                dependencies = [dep_id]

        deadline = datetime.now() + timedelta(
            hours=random.randint(2, 48)
        )

        task = Task(
            task_id=task_id,
            description=template["description"],
            priority=template["priority"],
            dependencies=dependencies,
            deadline=deadline,
            assigned_tick=tick,
        )
        self.tasks[task_id] = task
        return task

    def generate_events(self, tick: int) -> list[dict[str, Any]]:
        """Generate events for the given tick.

        This method is called by the SimulatedEnvironment on each observe()
        cycle.  It produces task arrivals, deadline warnings, and
        task completion events.

        Args:
            tick: The current tick number.

        Returns:
            A list of event dictionaries.
        """
        events: list[dict[str, Any]] = []

        # Task arrivals
        if tick in self._arrival_schedule:
            num_new = random.randint(1, 2)
            for _ in range(num_new):
                task = self._create_task(tick)
                events.append({
                    "source": "task_arrival",
                    "content": (
                        f"New task {task.task_id}: '{task.description}' "
                        f"(priority={task.priority}, "
                        f"dependencies={task.dependencies}, "
                        f"deadline={task.deadline.strftime('%H:%M')})"
                    ),
                    "relevance": 0.8 + task.priority * 0.04,
                    "metadata": {"task": task.to_dict(), "event_type": "task_arrival"},
                })

        # In-progress task completion
        if self._in_progress_task and self._in_progress_remaining_ticks > 0:
            self._in_progress_remaining_ticks -= 1
            if self._in_progress_remaining_ticks == 0:
                task_id = self._in_progress_task
                if task_id in self.tasks:
                    self.tasks[task_id].status = "completed"
                    self.completed_ids.add(task_id)
                    events.append({
                        "source": "task_completed",
                        "content": (
                            f"Task {task_id} has been completed successfully. "
                            f"'{self.tasks[task_id].description}'"
                        ),
                        "relevance": 0.9,
                        "metadata": {"task_id": task_id, "event_type": "task_completed"},
                    })
                self._in_progress_task = None

        # Deadline warnings for pending tasks
        for task in self.tasks.values():
            if task.status == "pending":
                time_remaining = task.deadline - datetime.now()
                if timedelta(hours=0) < time_remaining < timedelta(hours=2):
                    events.append({
                        "source": "deadline_warning",
                        "content": (
                            f"DEADLINE APPROACHING: Task {task.task_id} "
                            f"'{task.description}' is due soon "
                            f"(priority={task.priority})"
                        ),
                        "relevance": 0.95,
                        "metadata": {
                            "task_id": task.task_id,
                            "event_type": "deadline_warning",
                        },
                    })

        # Resource alerts (occasional)
        if tick % 7 == 0 and random.random() < 0.5:
            events.append({
                "source": "resource_alert",
                "content": "Server CPU usage at 85% — consider deferring low-priority tasks.",
                "relevance": 0.6,
                "metadata": {"event_type": "resource_alert", "cpu_usage": 85},
            })

        # Dependency resolved notifications
        newly_resolved = []
        for task in self.tasks.values():
            if task.status == "pending" and task.dependencies:
                if all(dep in self.completed_ids for dep in task.dependencies):
                    newly_resolved.append(task)

        for task in newly_resolved:
            # Only notify once — check if we already sent this event
            events.append({
                "source": "dependency_resolved",
                "content": (
                    f"All dependencies resolved for task {task.task_id}: "
                    f"'{task.description}' is now ready to start."
                ),
                "relevance": 0.75,
                "metadata": {
                    "task_id": task.task_id,
                    "event_type": "dependency_resolved",
                },
            })

        # If no events this tick, add a heartbeat
        if not events:
            events.append({
                "source": "system",
                "content": f"Heartbeat tick {tick}. No new events.",
                "relevance": 0.1,
                "metadata": {"tick": tick, "event_type": "heartbeat"},
            })

        return events

    def handle_action(self, action: Action) -> dict[str, Any]:
        """Process an agent action and return the result.

        Args:
            action: The Action to handle.

        Returns:
            A result dictionary with 'success', 'signal', 'message', and
            'new_observations' keys.
        """
        action_type = action.action_type
        params = action.params

        if action_type == "respond":
            return {
                "success": True,
                "signal": "neutral",
                "message": f"Response recorded: {params.get('message', '')}",
                "new_observations": [],
            }

        elif action_type == "query_environment":
            query = params.get("query", "general")
            # Provide additional information about the environment
            pending = [
                t.to_dict() for t in self.tasks.values() if t.status == "pending"
            ]
            in_progress = [
                t.to_dict() for t in self.tasks.values() if t.status == "in_progress"
            ]
            return {
                "success": True,
                "signal": "positive",
                "message": f"Environment query: {query}",
                "new_observations": [
                    f"Query result: {len(pending)} pending tasks, "
                    f"{len(in_progress)} in progress, "
                    f"{len(self.completed_ids)} completed."
                ],
            }

        elif action_type == "wait":
            return {
                "success": True,
                "signal": "neutral",
                "message": "Agent chose to wait for more information.",
                "new_observations": [],
            }

        elif action_type == "escalate":
            task_id = params.get("task_id", "")
            if task_id in self.tasks:
                self.tasks[task_id].status = "escalated"
                return {
                    "success": True,
                    "signal": "positive",
                    "message": f"Task {task_id} has been escalated to human operator.",
                    "new_observations": [
                        f"Task {task_id} escalated: '{self.tasks[task_id].description}'"
                    ],
                }
            return {
                "success": False,
                "signal": "negative",
                "message": f"Cannot escalate: task {task_id} not found.",
                "new_observations": [],
            }

        elif action_type == "tool_use":
            tool = params.get("tool", "unknown")
            task_id = params.get("task_id", "")

            if tool == "start_task":
                return self._handle_start_task(task_id)
            elif tool == "defer_task":
                return self._handle_defer_task(task_id)
            elif tool == "request_info":
                return self._handle_request_info(task_id)
            else:
                return {
                    "success": False,
                    "signal": "negative",
                    "message": f"Unknown tool: {tool}",
                    "new_observations": [],
                }

        return {
            "success": False,
            "signal": "negative",
            "message": f"Unknown action type: {action_type}",
            "new_observations": [],
        }

    def _handle_start_task(self, task_id: str) -> dict[str, Any]:
        """Handle a start_task action.

        Validates that the task exists, dependencies are met, and no other
        task is currently in progress.
        """
        if self._in_progress_task:
            return {
                "success": False,
                "signal": "negative",
                "message": f"Cannot start task {task_id}: another task "
                           f"({self._in_progress_task}) is already in progress.",
                "new_observations": [],
            }

        if task_id not in self.tasks:
            return {
                "success": False,
                "signal": "negative",
                "message": f"Cannot start task {task_id}: task not found.",
                "new_observations": [],
            }

        task = self.tasks[task_id]

        if task.status != "pending":
            return {
                "success": False,
                "signal": "negative",
                "message": f"Cannot start task {task_id}: status is '{task.status}', not 'pending'.",
                "new_observations": [],
            }

        # Check dependencies
        unmet = [d for d in task.dependencies if d not in self.completed_ids]
        if unmet:
            return {
                "success": False,
                "signal": "negative",
                "message": f"Cannot start task {task_id}: unmet dependencies: {unmet}",
                "new_observations": [
                    f"Task {task_id} blocked by: {', '.join(unmet)}"
                ],
            }

        # Start the task
        task.status = "in_progress"
        self._in_progress_task = task_id
        self._in_progress_remaining_ticks = random.randint(2, 4)

        return {
            "success": True,
            "signal": "positive",
            "message": f"Started task {task_id}: '{task.description}'. "
                       f"Estimated completion in {self._in_progress_remaining_ticks} cycles.",
            "new_observations": [
                f"Task {task_id} is now in progress."
            ],
        }

    def _handle_defer_task(self, task_id: str) -> dict[str, Any]:
        """Handle a defer_task action."""
        if task_id not in self.tasks:
            return {
                "success": False,
                "signal": "negative",
                "message": f"Cannot defer task {task_id}: task not found.",
                "new_observations": [],
            }

        task = self.tasks[task_id]
        if task.status != "pending":
            return {
                "success": False,
                "signal": "negative",
                "message": f"Cannot defer task {task_id}: status is '{task.status}'.",
                "new_observations": [],
            }

        task.status = "deferred"
        return {
            "success": True,
            "signal": "neutral",
            "message": f"Task {task_id} has been deferred.",
            "new_observations": [f"Task {task_id} deferred: '{task.description}'"],
        }

    def _handle_request_info(self, task_id: str) -> dict[str, Any]:
        """Handle a request_info action — returns detailed task information."""
        if task_id not in self.tasks:
            return {
                "success": False,
                "signal": "negative",
                "message": f"Task {task_id} not found.",
                "new_observations": [],
            }

        task = self.tasks[task_id]
        dep_status = {}
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            dep_status[dep_id] = dep_task.status if dep_task else "unknown"

        return {
            "success": True,
            "signal": "positive",
            "message": f"Information about task {task_id}.",
            "new_observations": [
                f"Task {task_id} details: priority={task.priority}, "
                f"status={task.status}, dependencies={dep_status}, "
                f"deadline={task.deadline.strftime('%H:%M')}"
            ],
        }

    def is_complete(self) -> bool:
        """Check if all tasks have been processed."""
        if not self.tasks:
            return False
        return all(
            t.status in ("completed", "escalated", "deferred")
            for t in self.tasks.values()
        )

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of the current scenario state."""
        status_counts: dict[str, int] = {}
        for task in self.tasks.values():
            status_counts[task.status] = status_counts.get(task.status, 0) + 1

        return {
            "total_tasks": len(self.tasks),
            "completed": len(self.completed_ids),
            "status_breakdown": status_counts,
            "in_progress": self._in_progress_task,
        }

    def reset(self) -> None:
        """Reset the scenario to its initial state."""
        self.tasks = {}
        self.completed_ids = set()
        self._task_counter = 0
        self._arrival_schedule = []
        self._in_progress_task = None
        self._in_progress_remaining_ticks = 0
        self._generate_arrival_schedule()

    def get_available_tasks_text(self) -> str:
        """Format available (startable) tasks as text for LLM prompts.

        Returns:
            A formatted string listing tasks that can be started.
        """
        lines = []
        for task in self.tasks.values():
            if task.status == "pending":
                unmet = [d for d in task.dependencies if d not in self.completed_ids]
                deps_str = (
                    f"BLOCKED by {unmet}" if unmet else "READY"
                )
                lines.append(
                    f"- {task.task_id}: '{task.description}' "
                    f"(P{task.priority}, {deps_str}, "
                    f"deadline={task.deadline.strftime('%H:%M')})"
                )
        if not lines:
            return "No pending tasks available."
        return "\n".join(lines)
