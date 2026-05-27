"""Manages localized market schedules and intraday momentum execution phases."""

import asyncio
from datetime import datetime
import pytz
from indian_intraday_system.config import (
    REGIME_ARM_START,
    REGIME_LOCK_START,
    REGIME_MARKET_CLOSE,
    REGIME_MOMENTUM_START,
    REGIME_KILL_START,
    TIMEZONE,
)


class TimeManager:
    """Locks strategy states according to exact Indian Standard Time (IST) regimes."""

    def __init__(self):
        self.tz = pytz.timezone(TIMEZONE)
        self.virtual_time = None

    def set_virtual_time(self, dt: datetime):
        """Registers a virtual simulated time to override system clocks."""
        self.virtual_time = dt

    def get_current_time(self) -> datetime:
        if self.virtual_time is not None:
            return self.virtual_time
        return datetime.now(self.tz)

    def get_minutes_to_kill(self) -> float:
        """Calculates minutes remaining until 3:15 PM liquidation triggers."""
        now = self.get_current_time()
        kill_dt = datetime.combine(now.date(), REGIME_KILL_START)
        kill_dt = self.tz.localize(kill_dt)

        if kill_dt < now:
            return 0.0
        return (kill_dt - now).total_seconds() / 60.0

    def get_current_regime(self) -> str:
        """Maps current time to execution states: LOCK, MEAN_REVERSION, MOMENTUM, KILL."""
        current_time = self.get_current_time().time()

        if current_time < REGIME_LOCK_START or current_time >= REGIME_MARKET_CLOSE:
            return "LOCK"
        
        if REGIME_LOCK_START <= current_time < REGIME_ARM_START:
            return "LOCK"  # Reconnaissance settle-down phase

        if REGIME_ARM_START <= current_time < REGIME_MOMENTUM_START:
            return "MEAN_REVERSION"  # Hunt pinning walls

        if REGIME_MOMENTUM_START <= current_time < REGIME_KILL_START:
            return "MOMENTUM"  # Ride breakouts (0DTE squeeze)

        if current_time >= REGIME_KILL_START:
            return "KILL"  # Flat all positions

        return "LOCK"

    def get_seconds_until(self, target_time) -> float:
        now = self.get_current_time()
        target_dt = datetime.combine(now.date(), target_time)
        target_dt = self.tz.localize(target_dt)

        if target_dt < now:
            from datetime import timedelta
            target_dt += timedelta(days=1)
            target_dt = self.tz.localize(target_dt)

        return (target_dt - now).total_seconds()
