# -*- coding: utf-8 -*-
"""
===================================
YfinanceFetcher - 兜底数据源 (Priority 4)
===================================

数据来源：Yahoo Finance（通过 yfinance 库）
特点：国际数据源、可能有延迟或缺失
定位：当所有国内数据源都失败时的最后保障

关键策略：
1. 自动将 A 股代码转换为 yfinance 格式（.SS / .SZ）
2. 处理 Yahoo Finance 的数据格式差异
3. 失败后指数退避重试
4. 修复 yfinance 新版多级索引导致的兼容性问题
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# 确保引用的是你的 base.py 中定义的类
from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS

logger = logging.getLogger(__name__)


class YfinanceFetcher(BaseFetcher):
    """
    Yahoo Finance 数据源实现
    
    优先级：4（最低，作为兜底，或用于美股/加密货币）
    """
    
    name = "YfinanceFetcher"
    priority = 4
    
    def _convert_stock_code(self, stock_code: str) -> str:
        """
        转换股票代码为 Yahoo Finance 格式
        支持 A股、港股(hk)、美股(us 或 纯字母)、加密货币(BTC-USD)
        """
        code = stock_code.strip()
        
        # 1. 处理美股：如果是以 us 开头 (如 usAAPL)
        if code.lower().startswith('us'):
            return code[2:].upper()
        
        # 2. 纯字母(美股) 或 带横杠(加密货币，如 BTC-USD)
        # 注意：A股代码都是数字，港股代码是数字(hk前缀在外部处理了)
        if code.isalpha() or '-' in code:
            return code.upper()

        # 3. 已经包含后缀的情况 (A股/港股)
        if '.SS' in code.upper() or '.SZ' in code.upper() or '.HK' in code.upper():
            return code.upper()
        
        # 4. 港股处理 (兼容 hk00700 格式)
        if code.lower().startswith('hk'):
            clean_code = code[2:]
            # 移除可能的前导0，Yahoo 港股通常是 0700.HK
            return f"{int(clean_code):04d}.HK"
        
        # 5. A 股逻辑
        code = code.replace('.SH', '').replace('.sh', '')
        if code.startswith(('600', '601', '603', '688')):
            return f"{code}.SS"
        elif code.startswith(('000', '002', '300')):
            return f"{code}.SZ"
        
        # 6. 无法识别的情况，直接返回原代码尝试
        return code
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)), # 捕获常规异常以便重试
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Yahoo Finance 获取原始数据
        """
        # 转换代码格式
        yf_code = self._convert_stock_code(stock_code)
        
        logger.debug(f"[{self.name}] 调用 yfinance.download({yf_code}, {start_date}, {end_date})")
        
        try:
            # 使用 yfinance 下载数据
            # 🛠️ 关键修复：multi_level_index=False
            df = yf.download(
                tickers=yf_code,
                start=start_date,
                end=end_date,
                progress=False,       # 禁止进度条
                auto_adjust=True,     # 自动调整价格（复权）
                multi_level_index=False # 🔴 必须加这个，否则新版 yfinance 会报错 "arg must be a list"
            )
            
            if df is None or df.empty:
                raise DataFetchError(f"Yahoo Finance 未查询到 {stock_code} ({yf_code}) 的数据")
            
            return df
            
        except Exception as e:
            # 如果已经是 DataFetchError，直接抛出
            if isinstance(e, DataFetchError):
                raise
            # 包装其他异常
            raise DataFetchError(f"Yahoo Finance 获取 {yf_code} 失败: {e}") from e
    
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Yahoo Finance 数据
        """
        df = df.copy()
        
        # 重置索引，将日期从索引变为列
        df = df.reset_index()
        
        # 统一列名（将 yfinance 的首字母大写转换为小写）
        # yfinance 通常返回: Date, Open, High, Low, Close, Volume
        df.columns = [c.lower() for c in df.columns]
        
        # 确保包含标准列
        # 计算涨跌幅（yfinance 不直接提供）
        if 'close' in df.columns:
            # 填充涨跌额和涨跌幅
            df['pct_chg'] = df['close'].pct_change() * 100
            df['pct_chg'] = df['pct_chg'].fillna(0).round(2)
        
        # 计算成交额（yfinance 不提供，使用估算值）
        if 'volume' in df.columns and 'close' in df.columns:
            df['amount'] = df['volume'] * df['close']
        else:
            df['amount'] = 0.0
            
        # 添加股票代码列
        df['code'] = stock_code
        
        # 确保日期列格式正确
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 筛选并排序最终列
        final_cols = ['code'] + STANDARD_COLUMNS
        # 填充缺失列
        for col in final_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        return df[final_cols]
