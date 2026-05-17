class TradingBaseException(Exception):
    """Base exception for all custom trading system errors."""
    pass

class DataStallException(TradingBaseException):
    """Raised when a WebSocket remains open but stops receiving market ticks."""
    pass

class RateLimitException(TradingBaseException):
    """Raised when an exchange HTTP endpoint returns a 429 Rate Limited status."""
    pass

class SchemaValidationException(TradingBaseException):
    """Raised when incoming exchange payloads fail Pydantic validation."""
    pass

class InsufficientFundsException(TradingBaseException):
    """Raised when the Binance account has insufficient balance for an order."""
    pass

class CriticalExecutionException(TradingBaseException):
    """Raised when a critical error (such as invalid API key, IP, or permissions) is encountered."""
    pass

