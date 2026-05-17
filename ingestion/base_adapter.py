import asyncio
from abc import ABC, abstractmethod

class DataFeedAdapter(ABC):
    """Abstract Base Class defining the interface contract for all market data ingestion adapters."""
    
    @abstractmethod
    async def connect_and_stream(self, queue: asyncio.Queue):
        """Establishes connection to the data provider and pipes normalized InternalTick objects to the queue."""
        pass
        
    @abstractmethod
    async def close(self):
        """Cleanly severs network connections and stops monitoring daemons."""
        pass
