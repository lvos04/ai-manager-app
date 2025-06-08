"""
Database Performance Optimization for AI Project Manager.
"""

import logging
import sqlite3
import threading
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseOptimizer:
    """Database performance enhancements."""
    
    def __init__(self, db_path: str = "ai_project_manager.db"):
        self.db_path = db_path
        self.connection_pool = self._create_connection_pool()
        self.query_cache = QueryCache()
        self.lock = threading.Lock()
        
    def _create_connection_pool(self, pool_size: int = 5):
        """Create database connection pool."""
        pool = []
        for _ in range(pool_size):
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                pool.append(conn)
            except Exception as e:
                logger.error(f"Failed to create database connection: {e}")
        return pool
        
    def get_connection(self):
        """Get connection from pool."""
        with self.lock:
            if self.connection_pool:
                return self.connection_pool.pop()
            else:
                try:
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    return conn
                except Exception as e:
                    logger.error(f"Failed to create new database connection: {e}")
                    return None
                    
    def return_connection(self, conn):
        """Return connection to pool."""
        with self.lock:
            if len(self.connection_pool) < 5:
                self.connection_pool.append(conn)
            else:
                conn.close()
                
    def optimize_queries(self):
        """Implement advanced database optimizations."""
        conn = self.get_connection()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_projects_created_at 
                ON projects(created_at)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_projects_status 
                ON projects(status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_type_downloaded 
                ON models(type, downloaded)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pipeline_runs_project_id 
                ON pipeline_runs(project_id)
            """)
            
            cursor.execute("PRAGMA optimize")
            
            conn.commit()
            logger.info("Database optimization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return False
        finally:
            self.return_connection(conn)
            
    def bulk_insert(self, table: str, data: List[Dict[str, Any]]) -> bool:
        """Perform bulk insert operations."""
        if not data:
            return True
            
        conn = self.get_connection()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            
            columns = list(data[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            values = [tuple(row[col] for col in columns) for row in data]
            cursor.executemany(query, values)
            
            conn.commit()
            logger.info(f"Bulk inserted {len(data)} rows into {table}")
            return True
            
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            return False
        finally:
            self.return_connection(conn)
            
    def execute_cached_query(self, query: str, params: tuple = ()) -> Optional[List[Dict]]:
        """Execute query with caching."""
        cache_key = f"{query}_{params}"
        
        cached_result = self.query_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
            
        conn = self.get_connection()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            results = [dict(row) for row in cursor.fetchall()]
            self.query_cache.set(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Cached query execution failed: {e}")
            return None
        finally:
            self.return_connection(conn)

class QueryCache:
    """Simple query result cache with TTL."""
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        with self.lock:
            if key in self.cache:
                import time
                if time.time() - self.access_times[key] < self.ttl:
                    return self.cache[key]
                else:
                    del self.cache[key]
                    del self.access_times[key]
            return None
            
    def set(self, key: str, value: Any):
        """Set cached value."""
        with self.lock:
            import time
            
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                
            self.cache[key] = value
            self.access_times[key] = time.time()

_database_optimizer = None

def get_database_optimizer():
    """Get global database optimizer instance."""
    global _database_optimizer
    if _database_optimizer is None:
        _database_optimizer = DatabaseOptimizer()
    return _database_optimizer
