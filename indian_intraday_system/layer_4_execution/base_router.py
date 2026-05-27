"""Abstract base class for indian_intraday_system execution routers."""

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseRouter(ABC):
    """Standard contract for order routing, position tracking, and fund queries."""

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        action: str,
        qty: int,
        order_type: str = "MARKET",
        price: float = None,
    ) -> Dict:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_funds(self) -> Dict:
        pass

    @abstractmethod
    def emergency_square_off(self) -> Dict:
        pass
