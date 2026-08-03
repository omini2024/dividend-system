#!/usr/bin/env python3
# ==============================
# 月末監視ガード：本日が「当月の最終取引日(JPX)」なら exit 0、それ以外は exit 1
# systemd .service の ExecCondition= から呼ぶ想定。
# JPX休場 = 土日 + 日本の祝日 + 年末年始(12/31, 1/2, 1/3)
# 依存: jpholiday  (venv内に pip install jpholiday)
# ==============================
import sys
from datetime import date, timedelta
import jpholiday


def is_jpx_trading_day(d: date) -> bool:
    if d.weekday() >= 5:                  # 土(5)・日(6)
        return False
    if jpholiday.is_holiday(d):           # 日本の祝日
        return False
    if d.month == 12 and d.day == 31:     # 大納会翌日（休場）
        return False
    if d.month == 1 and d.day in (2, 3):  # 年始休場
        return False
    return True


def is_last_trading_day_of_month(d: date) -> bool:
    if not is_jpx_trading_day(d):
        return False
    nxt = d + timedelta(days=1)
    while nxt.month == d.month:
        if is_jpx_trading_day(nxt):       # 当月にまだ取引日が残っている
            return False
        nxt += timedelta(days=1)
    return True


if __name__ == "__main__":
    sys.exit(0 if is_last_trading_day_of_month(date.today()) else 1)
