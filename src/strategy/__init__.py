"""Trading strategy: signal generation and backtesting."""

from src.strategy.backtest import backtest
from src.strategy.signals import predictions_to_signal, states_to_signal

__all__ = ["predictions_to_signal", "states_to_signal", "backtest"]
