from typing import List
from pydantic import BaseModel, Field

class InternalTick(BaseModel):
    """Unified internal tick structure representing normalized market data across all exchanges."""
    symbol: str = Field(..., description="Trading pair symbol, e.g., BTCUSDT")
    exchange: str = Field(..., description="Exchange identifier, e.g., BINANCE")
    bid: float = Field(..., description="Best bid price")
    ask: float = Field(..., description="Best ask price")
    bid_size: float = Field(..., description="Best bid volume")
    ask_size: float = Field(..., description="Best ask volume")
    timestamp_ns: int = Field(..., description="Epoch timestamp in nanoseconds")

from pydantic import BaseModel, Field, model_validator
from typing import List, Any

class BinanceDepthEntry(BaseModel):
    price: str
    quantity: str

    @model_validator(mode='before')
    @classmethod
    def parse_list(cls, value: Any) -> Any:
        if isinstance(value, list) and len(value) >= 2:
            return {"price": value[0], "quantity": value[1]}
        return value

class BinanceDepthPayload(BaseModel):
    """Validates incoming Binance Level 2 order book snapshots/updates."""
    e: str  # Event type
    E: int  # Event time (ms)
    s: str  # Symbol
    b: List[BinanceDepthEntry]  # Bids
    a: List[BinanceDepthEntry]  # Asks

class BinanceBookTickerPayload(BaseModel):
    """Validates incoming Binance Best Bid/Offer updates."""
    u: int  # order book updateId
    s: str  # symbol
    b: str  # best bid price
    B: str  # best bid qty
    a: str  # best ask price
    A: str  # best ask qty
