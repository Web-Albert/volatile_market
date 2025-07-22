#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BTC震荡市场交易策略配置文件
在这里修改策略参数，无需修改主程序代码
"""

# ==================== 策略参数配置 ====================

# 基础策略参数
MIN_BTC_RATIO = 0.0         # 最低BTC仓位比例 (30%)
MAX_BTC_RATIO = 0.9         # 最大BTC仓位比例 (80%)
INITIAL_CASH = 100000       # 初始资金 (100,000 USDT)
TRADING_FEE = 0.0005        # 交易手续费 (万分之五)

# 新闻信号阈值参数
NEWS_HIGH_THRESHOLD = 1.4   # 新闻信号高阈值（看多）
NEWS_LOW_THRESHOLD = 0.55    # 新闻信号低阈值（看空）

# ADX相关参数
ADX_PERIOD = 14             # ADX计算周期
ADX_LOW_THRESHOLD = 20      # ADX低阈值（趋势较弱）
ADX_HIGH_THRESHOLD = 35     # ADX高阈值（趋势较强）

# 合成信号阈值参数（20 <= ADX <= 40时 - 中等趋势）
MEDIUM_TREND_BUY_THRESHOLD = 1.5    # 中等趋势买入阈值
MEDIUM_TREND_SELL_THRESHOLD = 0.7   # 中等趋势卖出阈值

# 合成信号阈值参数（ADX > 40时 - 强趋势）
STRONG_TREND_BUY_THRESHOLD = 1.3    # 强趋势买入阈值
STRONG_TREND_SELL_THRESHOLD = 0.7   # 强趋势卖出阈值

# 合成信号阈值参数（ADX < 20时 - 弱趋势）
WEAK_TREND_SELL_THRESHOLD = 0.7     # 弱趋势卖出阈值（只卖出，不买入）
WEAK_TREND_SELL_RATIO = 0.3         # 弱趋势减仓比例（30%）

# 仓位参数
MEDIUM_TREND_BASE_RATIO = 0.5      # 中等趋势基础仓位比例 (25%)
STRONG_TREND_BASE_RATIO = 0.4      # 强趋势基础仓位比例 (15%)

# 止盈止损参数
MEDIUM_TREND_STOP_PROFIT = 0.08     # 中等趋势止盈比例 (6%)
MEDIUM_TREND_STOP_LOSS = 0.06     # 中等趋势止损比例 (3%)
STRONG_TREND_STOP_PROFIT = 0.06     # 强趋势止盈比例 (4%)
STRONG_TREND_STOP_LOSS = 0.06       # 强趋势止损比例 (3%)

# 肯特纳通道参数
KELTNER_PERIOD = 50                 # 肯特纳通道中线周期（50天EMA）
KELTNER_ATR_PERIOD = 50             # ATR计算周期（50天）
KELTNER_MULTIPLIER = 3.75           # ATR倍数（3.75倍）

# 中等趋势 (20 <= ADX <= 40) 肯特纳通道交易参数
MEDIUM_TREND_UPPER_SELL_RATIO = 0.20       # 中等趋势上穿上线减仓比例
MEDIUM_TREND_MIDDLE_SELL_RATIO = 0.15      # 中等趋势上穿中线减仓比例
MEDIUM_TREND_LOWER_BUY_RATIO = 0.15        # 中等趋势下穿下线加仓比例
MEDIUM_TREND_MIDDLE_BUY_RATIO = 0.10       # 中等趋势下穿中线加仓比例

# 强趋势 (ADX > 40) 肯特纳通道交易参数
STRONG_TREND_UPPER_SELL_RATIO = 0.25       # 强趋势上穿上线减仓比例（更大）
STRONG_TREND_MIDDLE_SELL_RATIO = 0.20      # 强趋势上穿中线减仓比例（更大）
STRONG_TREND_LOWER_BUY_RATIO = 0.10        # 强趋势下穿下线加仓比例（更小）
STRONG_TREND_MIDDLE_BUY_RATIO = 0.08       # 强趋势下穿中线加仓比例（更小）

# 周跌幅加仓参数
WEEKLY_DROP_THRESHOLD = 0.15        # 周跌幅阈值（15%）
WEEKLY_DROP_BUY_RATIO = 0.1         # 周跌幅加仓比例 (10%)
WEEKLY_DROP_COOLDOWN_DAYS = 7       # 周跌幅加仓冷却期（天数）

# 技术指标参数
MACD_PARAMS = {
    'fast': 12,
    'slow': 26,
    'signal': 9
}

BOLL_PARAMS = {
    'period': 20,
    'std': 2
}

RSI_PARAMS = {
    'period': 14,
    'buy_threshold': 30,
    'sell_threshold': 75
}

# ==================== 数据文件路径配置 ====================

# 日线K线数据路径
DAILY_KLINES_PATH = r'E:\Jupyter_Notebook_files\_crypto实习\舆情CTA增强模型\klines_daily\btc_daily_bt.csv'

# 15分钟K线数据路径
MIN15_KLINES_PATH = r'E:\Jupyter_Notebook_files\_crypto实习\舆情CTA增强模型\klines_features_bt\klines_features_bt.pkl'

# 新闻信号数据路径
NEWS_SIGNALS_PATH = r'E:\Jupyter_Notebook_files\_crypto实习\舆情CTA增强模型\_news_phase\important_news_signal_alpha0.2.csv'

# ==================== 结果保存配置 ====================

# 结果保存目录
RESULTS_DIR = r".\volatile_strategy_results"

# ==================== 获取配置函数 ====================

def get_volatile_strategy_params():
    """获取震荡市场策略参数字典"""
    return {
        'min_btc_ratio': MIN_BTC_RATIO,
        'max_btc_ratio': MAX_BTC_RATIO,
        'initial_cash': INITIAL_CASH,
        'trading_fee': TRADING_FEE,
        'news_high_threshold': NEWS_HIGH_THRESHOLD,
        'news_low_threshold': NEWS_LOW_THRESHOLD,
        'adx_period': ADX_PERIOD,
        'adx_low_threshold': ADX_LOW_THRESHOLD,
        'adx_high_threshold': ADX_HIGH_THRESHOLD,
        'medium_trend_buy_threshold': MEDIUM_TREND_BUY_THRESHOLD,
        'medium_trend_sell_threshold': MEDIUM_TREND_SELL_THRESHOLD,
        'strong_trend_buy_threshold': STRONG_TREND_BUY_THRESHOLD,
        'strong_trend_sell_threshold': STRONG_TREND_SELL_THRESHOLD,
        'weak_trend_sell_threshold': WEAK_TREND_SELL_THRESHOLD,
        'weak_trend_sell_ratio': WEAK_TREND_SELL_RATIO,
        'medium_trend_base_ratio': MEDIUM_TREND_BASE_RATIO,
        'strong_trend_base_ratio': STRONG_TREND_BASE_RATIO,
        'medium_trend_stop_profit': MEDIUM_TREND_STOP_PROFIT,
        'medium_trend_stop_loss': MEDIUM_TREND_STOP_LOSS,
        'strong_trend_stop_profit': STRONG_TREND_STOP_PROFIT,
        'strong_trend_stop_loss': STRONG_TREND_STOP_LOSS,
        'keltner_period': KELTNER_PERIOD,
        'keltner_atr_period': KELTNER_ATR_PERIOD,
        'keltner_multiplier': KELTNER_MULTIPLIER,
        'medium_trend_upper_sell_ratio': MEDIUM_TREND_UPPER_SELL_RATIO,
        'medium_trend_middle_sell_ratio': MEDIUM_TREND_MIDDLE_SELL_RATIO,
        'medium_trend_lower_buy_ratio': MEDIUM_TREND_LOWER_BUY_RATIO,
        'medium_trend_middle_buy_ratio': MEDIUM_TREND_MIDDLE_BUY_RATIO,
        'strong_trend_upper_sell_ratio': STRONG_TREND_UPPER_SELL_RATIO,
        'strong_trend_middle_sell_ratio': STRONG_TREND_MIDDLE_SELL_RATIO,
        'strong_trend_lower_buy_ratio': STRONG_TREND_LOWER_BUY_RATIO,
        'strong_trend_middle_buy_ratio': STRONG_TREND_MIDDLE_BUY_RATIO,
        'weekly_drop_threshold': WEEKLY_DROP_THRESHOLD,
        'weekly_drop_buy_ratio': WEEKLY_DROP_BUY_RATIO,
        'weekly_drop_cooldown_days': WEEKLY_DROP_COOLDOWN_DAYS,
        'macd_params': MACD_PARAMS,
        'boll_params': BOLL_PARAMS,
        'rsi_params': RSI_PARAMS
    }

def get_volatile_data_paths():
    """获取数据文件路径字典"""
    return {
        'daily_klines': DAILY_KLINES_PATH,
        'min15_klines': MIN15_KLINES_PATH,
        'news_signals': NEWS_SIGNALS_PATH
    }

def print_volatile_config():
    """打印当前配置"""
    print("📋 震荡市场策略配置:")
    print("=" * 60)
    print("🎯 基础策略参数:")
    print(f"  最低BTC仓位比例: {MIN_BTC_RATIO:.2%}")
    print(f"  最大BTC仓位比例: {MAX_BTC_RATIO:.2%}")
    print(f"  初始资金: {INITIAL_CASH:,.2f} USDT")
    print(f"  交易手续费: {TRADING_FEE:.4%} (万分之{TRADING_FEE*10000:.0f})")
    
    print("\n📰 新闻信号阈值:")
    print(f"  高阈值 (看多): {NEWS_HIGH_THRESHOLD:.2f}")
    print(f"  低阈值 (看空): {NEWS_LOW_THRESHOLD:.2f}")
    
    print("\n📊 ADX参数:")
    print(f"  ADX计算周期: {ADX_PERIOD}")
    print(f"  ADX低阈值 (弱趋势): {ADX_LOW_THRESHOLD}")
    print(f"  ADX高阈值 (强趋势): {ADX_HIGH_THRESHOLD}")
    
    print("\n🔄 合成信号阈值 (中等趋势 20 <= ADX <= 40):")
    print(f"  买入阈值: {MEDIUM_TREND_BUY_THRESHOLD:.2f}")
    print(f"  卖出阈值: {MEDIUM_TREND_SELL_THRESHOLD:.2f}")
    print(f"  基础仓位比例: {MEDIUM_TREND_BASE_RATIO:.2%}")
    print(f"  止盈比例: {MEDIUM_TREND_STOP_PROFIT:.2%}")
    print(f"  止损比例: {MEDIUM_TREND_STOP_LOSS:.2%}")
    
    print("\n🔄 合成信号阈值 (强趋势 ADX > 40):")
    print(f"  买入阈值: {STRONG_TREND_BUY_THRESHOLD:.2f}")
    print(f"  卖出阈值: {STRONG_TREND_SELL_THRESHOLD:.2f}")
    print(f"  基础仓位比例: {STRONG_TREND_BASE_RATIO:.2%}")
    print(f"  止盈比例: {STRONG_TREND_STOP_PROFIT:.2%}")
    print(f"  止损比例: {STRONG_TREND_STOP_LOSS:.2%}")
    
    print("\n📈 肯特纳通道参数:")
    print(f"  中线周期: {KELTNER_PERIOD}天EMA")
    print(f"  ATR周期: {KELTNER_ATR_PERIOD}天")
    print(f"  ATR倍数: {KELTNER_MULTIPLIER}倍")
    print(f"  中等趋势上穿上线减仓比例: {MEDIUM_TREND_UPPER_SELL_RATIO:.2%}")
    print(f"  中等趋势上穿中线减仓比例: {MEDIUM_TREND_MIDDLE_SELL_RATIO:.2%}")
    print(f"  中等趋势下穿下线加仓比例: {MEDIUM_TREND_LOWER_BUY_RATIO:.2%}")
    print(f"  中等趋势下穿中线加仓比例: {MEDIUM_TREND_MIDDLE_BUY_RATIO:.2%}")
    print(f"  强趋势上穿上线减仓比例: {STRONG_TREND_UPPER_SELL_RATIO:.2%}")
    print(f"  强趋势上穿中线减仓比例: {STRONG_TREND_MIDDLE_SELL_RATIO:.2%}")
    print(f"  强趋势下穿下线加仓比例: {STRONG_TREND_LOWER_BUY_RATIO:.2%}")
    print(f"  强趋势下穿中线加仓比例: {STRONG_TREND_MIDDLE_BUY_RATIO:.2%}")
    print(f"  弱趋势卖出阈值: {WEAK_TREND_SELL_THRESHOLD:.2f}")
    print(f"  弱趋势减仓比例: {WEAK_TREND_SELL_RATIO:.2%}")
    print(f"  周跌幅阈值: {WEEKLY_DROP_THRESHOLD:.2%}")
    print(f"  周跌幅加仓比例: {WEEKLY_DROP_BUY_RATIO:.2%}")
    print(f"  周跌幅加仓冷却期: {WEEKLY_DROP_COOLDOWN_DAYS}天")
    
    print("\n⚙️ 技术指标参数:")
    print(f"  MACD参数: 快线{MACD_PARAMS['fast']}, 慢线{MACD_PARAMS['slow']}, 信号线{MACD_PARAMS['signal']}")
    print(f"  BOLL参数: 周期{BOLL_PARAMS['period']}, 标准差{BOLL_PARAMS['std']}")
    print(f"  RSI参数: 周期{RSI_PARAMS['period']}, 买入阈值{RSI_PARAMS['buy_threshold']}, 卖出阈值{RSI_PARAMS['sell_threshold']}")
    
    print("\n📁 数据文件路径:")
    print(f"  日线数据: {DAILY_KLINES_PATH}")
    print(f"  15分钟数据: {MIN15_KLINES_PATH}")
    print(f"  新闻信号: {NEWS_SIGNALS_PATH}")
    
    print(f"\n💾 结果保存目录: {RESULTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    print_volatile_config() 