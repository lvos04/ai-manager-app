"""
Smart Resource Scheduling for AI Project Manager.
"""

import logging
import asyncio
import threading
import heapq
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ScheduledTask:
    id: str
    priority: TaskPriority
    execute_func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    estimated_duration: float = 1.0
    resource_requirements: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at

class ResourceScheduler:
    """Intelligent resource scheduling."""
    
    def __init__(self):
        self.task_queue = []
        self.resource_monitor = ResourceMonitor()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.lock = threading.Lock()
        self.scheduler_running = False
        
    def schedule_task(self, task: ScheduledTask) -> str:
        """Schedule a task for execution."""
        with self.lock:
            heapq.heappush(self.task_queue, task)
            logger.info(f"Scheduled task {task.id} with priority {task.priority.name}")
            
        if not self.scheduler_running:
            asyncio.create_task(self._run_scheduler())
            
        return task.id
        
    async def _run_scheduler(self):
        """Main scheduler loop."""
        self.scheduler_running = True
        
        try:
            while True:
                await self._process_next_task()
                await asyncio.sleep(0.1)
                
                with self.lock:
                    if not self.task_queue and not self.active_tasks:
                        break
                        
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
        finally:
            self.scheduler_running = False
            
    async def _process_next_task(self):
        """Process the next available task."""
        task = None
        
        with self.lock:
            if self.task_queue:
                available_resources = self.resource_monitor.get_available_resources()
                
                for i, candidate_task in enumerate(self.task_queue):
                    if self._can_execute_task(candidate_task, available_resources):
                        task = self.task_queue.pop(i)
                        heapq.heapify(self.task_queue)
                        break
                        
        if task:
            await self._execute_task(task)
            
    def _can_execute_task(self, task: ScheduledTask, available_resources: Dict[str, float]) -> bool:
        """Check if task can be executed with available resources."""
        requirements = task.resource_requirements
        
        for resource, required_amount in requirements.items():
            if available_resources.get(resource, 0) < required_amount:
                return False
                
        return True
        
    async def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        task_id = task.id
        
        try:
            self.active_tasks[task_id] = {
                "task": task,
                "start_time": time.time(),
                "status": "running"
            }
            
            logger.info(f"Executing task {task_id}")
            
            if asyncio.iscoroutinefunction(task.execute_func):
                result = await task.execute_func(*task.args, **task.kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, task.execute_func, *task.args, **task.kwargs)
                
            self.completed_tasks[task_id] = {
                "task": task,
                "result": result,
                "completion_time": time.time(),
                "status": "completed"
            }
            
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self.completed_tasks[task_id] = {
                "task": task,
                "error": str(e),
                "completion_time": time.time(),
                "status": "failed"
            }
        finally:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
                
    def schedule_tasks_optimally(self, tasks: List[ScheduledTask]):
        """Schedule multiple tasks based on resource availability."""
        for task in tasks:
            self.schedule_task(task)
            
    def implement_background_processing(self, background_tasks: List[ScheduledTask]):
        """Process tasks in background when resources available."""
        for task in background_tasks:
            task.priority = TaskPriority.LOW
            self.schedule_task(task)
            
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        else:
            return None
            
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled or running task."""
        with self.lock:
            for i, task in enumerate(self.task_queue):
                if task.id == task_id:
                    self.task_queue.pop(i)
                    heapq.heapify(self.task_queue)
                    logger.info(f"Cancelled queued task {task_id}")
                    return True
                    
        if task_id in self.active_tasks:
            logger.warning(f"Cannot cancel running task {task_id}")
            return False
            
        return False

class ResourceMonitor:
    """Monitor system resource availability."""
    
    def __init__(self):
        self.resource_cache = {}
        self.cache_timeout = 1.0
        self.last_update = 0
        
    def get_available_resources(self) -> Dict[str, float]:
        """Get current available system resources."""
        current_time = time.time()
        
        if current_time - self.last_update > self.cache_timeout:
            self._update_resource_cache()
            self.last_update = current_time
            
        return self.resource_cache.copy()
        
    def _update_resource_cache(self):
        """Update cached resource information."""
        try:
            import psutil
            import torch
            
            cpu_percent = 100 - psutil.cpu_percent(interval=0.1)
            memory_percent = 100 - psutil.virtual_memory().percent
            
            gpu_memory_percent = 100
            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated()
                    total = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_percent = 100 - (allocated / total * 100)
                except:
                    pass
                    
            self.resource_cache = {
                "cpu": cpu_percent,
                "memory": memory_percent,
                "gpu_memory": gpu_memory_percent,
                "disk_io": 50.0,
                "network": 80.0
            }
            
        except Exception as e:
            logger.warning(f"Resource monitoring failed: {e}")
            self.resource_cache = {
                "cpu": 50.0,
                "memory": 50.0,
                "gpu_memory": 50.0,
                "disk_io": 50.0,
                "network": 50.0
            }

_resource_scheduler = None

def get_resource_scheduler():
    """Get global resource scheduler instance."""
    global _resource_scheduler
    if _resource_scheduler is None:
        _resource_scheduler = ResourceScheduler()
    return _resource_scheduler

def create_scheduled_task(task_id: str, priority: TaskPriority, execute_func: Callable, 
                         *args, estimated_duration: float = 1.0, 
                         resource_requirements: Dict[str, float] = None, **kwargs) -> ScheduledTask:
    """Create a new scheduled task."""
    if resource_requirements is None:
        resource_requirements = {}
        
    return ScheduledTask(
        id=task_id,
        priority=priority,
        execute_func=execute_func,
        args=args,
        kwargs=kwargs,
        estimated_duration=estimated_duration,
        resource_requirements=resource_requirements
    )
