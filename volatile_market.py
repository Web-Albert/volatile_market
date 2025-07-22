import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入信号生成器
from signal_generator import SignalGenerator

class VolatileMarketStrategy:
    """
    震荡市场交易策略：基于日线结合15分钟K线数据和ADX指标的震荡市场策略
    使用MACD、RSI、BOLL合成信号，结合多个MA作为支撑压力位
    只在震荡阶段运行，根据新闻信号判断市场阶段
    """
    
    def __init__(self,
                 # 基础策略参数
                 min_btc_ratio: float = 0.1,          # 最低BTC仓位比例
                 max_btc_ratio: float = 0.8,          # 最大BTC仓位比例
                 initial_cash: float = 100000,        # 初始资金
                 trading_fee: float = 0.0005,         # 交易手续费
                 
                 # 新闻信号阈值参数
                 news_high_threshold: float = 1.5,    # 新闻信号高阈值（看多）
                 news_low_threshold: float = 0.5,     # 新闻信号低阈值（看空）
                 
                 # ADX相关参数
                 adx_period: int = 14,                # ADX计算周期
                 adx_low_threshold: float = 20,       # ADX低阈值（趋势较弱）
                 adx_high_threshold: float = 40,      # ADX高阈值（趋势较强）
                 
                 # 合成信号阈值参数（20 <= ADX <= 40时）
                 medium_trend_buy_threshold: float = 1.5,   # 中等趋势买入阈值
                 medium_trend_sell_threshold: float = 0.5,  # 中等趋势卖出阈值
                 
                 # 合成信号阈值参数（ADX > 40时）
                 strong_trend_buy_threshold: float = 1.6,   # 强趋势买入阈值
                 strong_trend_sell_threshold: float = 0.4,  # 强趋势卖出阈值
                 
                 # 合成信号阈值参数（ADX < 20时）
                 weak_trend_sell_threshold: float = 0.6,    # 弱趋势卖出阈值（只卖出，不买入）
                 weak_trend_sell_ratio: float = 0.3,        # 弱趋势减仓比例
                 
                 # 仓位参数
                 medium_trend_base_ratio: float = 0.25,     # 中等趋势基础仓位比例
                 strong_trend_base_ratio: float = 0.15,     # 强趋势基础仓位比例
                 
                 # 止盈止损参数
                 medium_trend_stop_profit: float = 0.08,    # 中等趋势止盈比例
                 medium_trend_stop_loss: float = 0.05,      # 中等趋势止损比例
                 strong_trend_stop_profit: float = 0.04,    # 强趋势止盈比例
                 strong_trend_stop_loss: float = 0.03,      # 强趋势止损比例
                 
                 # 肯特纳通道参数
                 keltner_period: int = 50,                  # 肯特纳通道中线周期（50天）
                 keltner_atr_period: int = 50,              # ATR计算周期（50天）
                 keltner_multiplier: float = 3.75,          # ATR倍数（3.75倍）
                 
                 # 中等趋势肯特纳通道交易参数
                 medium_trend_upper_sell_ratio: float = 0.20,      # 中等趋势上穿上线减仓比例
                 medium_trend_middle_sell_ratio: float = 0.15,     # 中等趋势上穿中线减仓比例
                 medium_trend_lower_buy_ratio: float = 0.15,       # 中等趋势下穿下线加仓比例
                 medium_trend_middle_buy_ratio: float = 0.10,      # 中等趋势下穿中线加仓比例
                 
                 # 强趋势肯特纳通道交易参数
                 strong_trend_upper_sell_ratio: float = 0.25,      # 强趋势上穿上线减仓比例
                 strong_trend_middle_sell_ratio: float = 0.20,     # 强趋势上穿中线减仓比例
                 strong_trend_lower_buy_ratio: float = 0.10,       # 强趋势下穿下线加仓比例
                 strong_trend_middle_buy_ratio: float = 0.08,      # 强趋势下穿中线加仓比例
                 
                 # 周跌幅加仓参数
                 weekly_drop_threshold: float = 0.15,       # 周跌幅阈值（15%）
                 weekly_drop_buy_ratio: float = 0.1,        # 周跌幅加仓比例
                 weekly_drop_cooldown_days: int = 7,        # 周跌幅加仓冷却期（天数）
                 
                 # 信号生成器参数
                 macd_params: Dict = None,
                 boll_params: Dict = None,
                 rsi_params: Dict = None):
        
        # 基础参数
        self.min_btc_ratio = min_btc_ratio
        self.max_btc_ratio = max_btc_ratio
        self.initial_cash = initial_cash
        self.trading_fee = trading_fee
        
        # 新闻信号阈值
        self.news_high_threshold = news_high_threshold
        self.news_low_threshold = news_low_threshold
        
        # ADX参数
        self.adx_period = adx_period
        self.adx_low_threshold = adx_low_threshold
        self.adx_high_threshold = adx_high_threshold
        
        # 合成信号阈值
        self.medium_trend_buy_threshold = medium_trend_buy_threshold
        self.medium_trend_sell_threshold = medium_trend_sell_threshold
        self.strong_trend_buy_threshold = strong_trend_buy_threshold
        self.strong_trend_sell_threshold = strong_trend_sell_threshold
        self.weak_trend_sell_threshold = weak_trend_sell_threshold
        self.weak_trend_sell_ratio = weak_trend_sell_ratio
        
        # 仓位参数
        self.medium_trend_base_ratio = medium_trend_base_ratio
        self.strong_trend_base_ratio = strong_trend_base_ratio
        
        # 止盈止损参数
        self.medium_trend_stop_profit = medium_trend_stop_profit
        self.medium_trend_stop_loss = medium_trend_stop_loss
        self.strong_trend_stop_profit = strong_trend_stop_profit
        self.strong_trend_stop_loss = strong_trend_stop_loss
        
        # 肯特纳通道参数
        self.keltner_period = keltner_period
        self.keltner_atr_period = keltner_atr_period
        self.keltner_multiplier = keltner_multiplier
        self.medium_trend_upper_sell_ratio = medium_trend_upper_sell_ratio
        self.medium_trend_middle_sell_ratio = medium_trend_middle_sell_ratio
        self.medium_trend_lower_buy_ratio = medium_trend_lower_buy_ratio
        self.medium_trend_middle_buy_ratio = medium_trend_middle_buy_ratio
        self.strong_trend_upper_sell_ratio = strong_trend_upper_sell_ratio
        self.strong_trend_middle_sell_ratio = strong_trend_middle_sell_ratio
        self.strong_trend_lower_buy_ratio = strong_trend_lower_buy_ratio
        self.strong_trend_middle_buy_ratio = strong_trend_middle_buy_ratio
        
        # 周跌幅加仓参数
        self.weekly_drop_threshold = weekly_drop_threshold
        self.weekly_drop_buy_ratio = weekly_drop_buy_ratio
        self.weekly_drop_cooldown_days = weekly_drop_cooldown_days
        
        # 账户状态
        self.cash = initial_cash
        self.btc_amount = 0
        self.btc_value = 0
        self.total_value = initial_cash
        
        # 交易记录
        self.trades = []
        self.positions = []
        self.records = []
        
        # 市场状态
        self.market_phase = 'oscillation'  # 当前市场阶段
        self.current_news_signal = 1.0
        
        # 交易约束状态
        self.today_bought = False           # 当天是否已买入
        self.today_added_position = False   # 当天是否已加仓
        self.today_sold = False             # 当天是否已卖出或减仓
        self.last_trade_date = None         # 最后交易日期
        
        # 周跌幅加仓状态跟踪
        self.last_weekly_drop_buy_date = None  # 最后一次周跌幅加仓日期
        
        # 信号生成器
        self.signal_generator = SignalGenerator()
        
        # 如果提供了技术指标参数，更新信号生成器
        if macd_params or boll_params or rsi_params:
            self.signal_generator.update_technical_params(
                macd_params=macd_params,
                boll_params=boll_params, 
                rsi_params=rsi_params
            )
    
    def prepare_data(self, 
                    daily_klines: pd.DataFrame, 
                    min15_klines: pd.DataFrame, 
                    news_signals) -> pd.DataFrame:
        """
        准备和处理数据（参考trend_market.py的方法）
        
        Args:
            daily_klines: 日频K线数据
            min15_klines: 15分钟K线数据
            news_signals: 新闻信号数据
            
        Returns:
            pd.DataFrame: 合并后的数据
        """
        print("正在准备数据...")
        
        # 确保时间索引格式正确
        daily_klines.index = pd.to_datetime(daily_klines.index)
        min15_klines.index = pd.to_datetime(min15_klines.index)
        news_signals.index = pd.to_datetime(news_signals.index)
        
        # 为15分钟K线数据添加日期列
        min15_klines['date'] = min15_klines.index.date
        
        # 计算每个15分钟K线的临时日频技术指标信号和ADX
        from tqdm import tqdm
        
        tech_signals = []
        adx_values = []
        keltner_upper = []
        keltner_middle = []
        keltner_lower = []
        
        # 技术指标所需的最大回看期数
        max_lookback = max(
            self.signal_generator.macd_params['slow'] + self.signal_generator.macd_params['signal'],
            self.signal_generator.boll_params['period'],
            self.signal_generator.rsi_params['period'],
            self.adx_period,
            self.keltner_period,
            self.keltner_atr_period
        ) + 10  # 额外缓冲期
        
        print(f"正在计算 {len(min15_klines)} 个15分钟K线的技术指标信号...")
        
        for timestamp, row in tqdm(min15_klines.iterrows(), 
                                  total=len(min15_klines), 
                                  desc="计算技术指标"):
            current_date = timestamp.date()
            current_close = row['close']
            current_high = row['high']
            current_low = row['low']
            
            # 获取历史日线数据
            start_date = pd.Timestamp(current_date) - timedelta(days=max_lookback)
            historical_daily = daily_klines[
                (daily_klines.index >= start_date) & 
                (daily_klines.index.date <= current_date)
            ].copy()
            
            if len(historical_daily) > 0:
                # 更新当天的数据为当前15分钟的数据
                if historical_daily.index[-1].date() == current_date:
                    # 更新当天的收盘价
                    historical_daily.iloc[-1, historical_daily.columns.get_loc('close')] = current_close
                    # 更新当天的最高价和最低价（如果当前值更极端）
                    if current_high > historical_daily.iloc[-1]['high']:
                        historical_daily.iloc[-1, historical_daily.columns.get_loc('high')] = current_high
                    if current_low < historical_daily.iloc[-1]['low']:
                        historical_daily.iloc[-1, historical_daily.columns.get_loc('low')] = current_low
                else:
                    # 如果当天没有日频数据，添加一行
                    new_row = historical_daily.iloc[-1].copy()
                    new_row['close'] = current_close
                    new_row['high'] = current_high
                    new_row['low'] = current_low
                    new_row.name = pd.Timestamp(current_date)
                    new_row_df = pd.DataFrame([new_row], index=[pd.Timestamp(current_date)])
                    historical_daily = pd.concat([historical_daily, new_row_df])
                
                # 生成技术指标信号
                signals_continuous, signals_discrete = self.signal_generator.generate_technical_signal(historical_daily)
                tech_signal = round(signals_continuous.iloc[-1], 2) if len(signals_continuous) > 0 else 1.00
                
                # 计算ADX指标
                if len(historical_daily) >= self.adx_period:
                    adx = talib.ADX(
                        historical_daily['high'].values, 
                        historical_daily['low'].values, 
                        historical_daily['close'].values, 
                        timeperiod=self.adx_period
                    )
                    adx_value = adx[-1] if not np.isnan(adx[-1]) else 25.0
                else:
                    adx_value = 25.0
                
                # 计算肯特纳通道
                if len(historical_daily) >= max(self.keltner_period, self.keltner_atr_period):
                    # 计算中线（EMA）
                    middle_line = talib.EMA(historical_daily['close'].values, timeperiod=self.keltner_period)
                    
                    # 计算ATR
                    atr = talib.ATR(
                        historical_daily['high'].values,
                        historical_daily['low'].values,
                        historical_daily['close'].values,
                        timeperiod=self.keltner_atr_period
                    )
                    
                    # 计算上下线
                    if not np.isnan(middle_line[-1]) and not np.isnan(atr[-1]):
                        keltner_mid = middle_line[-1]
                        keltner_up = keltner_mid + (atr[-1] * self.keltner_multiplier)
                        keltner_low = keltner_mid - (atr[-1] * self.keltner_multiplier)
                    else:
                        keltner_mid = current_close
                        keltner_up = current_close * 1.05  # 默认上浮5%
                        keltner_low = current_close * 0.95  # 默认下浮5%
                else:
                    keltner_mid = current_close
                    keltner_up = current_close * 1.05
                    keltner_low = current_close * 0.95
                
                keltner_upper.append(keltner_up)
                keltner_middle.append(keltner_mid)
                keltner_lower.append(keltner_low)
                
            else:
                tech_signal = 1.00
                adx_value = 25.0
                keltner_upper.append(current_close * 1.05)
                keltner_middle.append(current_close)
                keltner_lower.append(current_close * 0.95)
            
            tech_signals.append(tech_signal)
            adx_values.append(adx_value)
        
        # 将计算结果添加到15分钟数据
        min15_klines['tech_signal'] = tech_signals
        min15_klines['adx'] = adx_values
        min15_klines['keltner_upper'] = keltner_upper
        min15_klines['keltner_middle'] = keltner_middle
        min15_klines['keltner_lower'] = keltner_lower
        
        print("技术指标信号计算完成")
        
        # 处理新闻信号（与trend_market.py相同的方式）
        combined_data = min15_klines.copy()
        
        if not news_signals.empty:
            from tqdm import tqdm
            
            print(f"正在处理 {len(news_signals)} 个新闻信号...")
            
            # 将新闻信号转换为DataFrame，便于处理
            if isinstance(news_signals, pd.DataFrame):
                news_series = news_signals.iloc[:, 0]  # 取第一列
            else:
                news_series = news_signals
            
            news_df = pd.DataFrame({
                'timestamp': news_series.index,
                'news_signal': news_series.values
            })
            
            # 为每个新闻信号找到对应的下一个15分钟K线时间点
            print("正在映射新闻信号到15分钟K线...")
            tqdm.pandas(desc="映射新闻信号")
            news_df['next_15min'] = news_df['timestamp'].progress_apply(
                lambda x: min15_klines[min15_klines.index > x].index[0] 
                if len(min15_klines[min15_klines.index > x]) > 0 else None
            )
            
            # 过滤掉没有对应15分钟K线的新闻
            news_df = news_df.dropna(subset=['next_15min'])
            
            # 处理重复索引：如果多个新闻信号映射到同一个15分钟K线，取最新的信号值
            news_df_unique = news_df.drop_duplicates(subset=['next_15min'], keep='last')
            
            news_series = pd.Series(
                data=news_df_unique['news_signal'].values,
                index=news_df_unique['next_15min']
            )
            
            # 合并到15分钟数据上
            print("正在合并新闻信号到K线数据...")
            combined_data = combined_data.merge(
                news_series.to_frame('news_signal'), 
                left_index=True, 
                right_index=True, 
                how='left'
            )
            
            # 向前填充新闻信号
            print("正在向前填充新闻信号...")
            combined_data['news_signal'] = combined_data['news_signal'].fillna(method='ffill')
            print("新闻信号处理完成")
        else:
            print("没有新闻信号数据，使用默认值")
            combined_data['news_signal'] = np.nan
        
        # 设置默认值
        combined_data['news_signal'] = combined_data['news_signal'].fillna(1.0)
        
        print("数据准备完成")
        return combined_data
    
    def update_account_status(self, current_price: float):
        """更新账户状态"""
        self.btc_value = self.btc_amount * current_price
        self.total_value = self.cash + self.btc_value
        
        # 更新持仓信息
        for position in self.positions:
            position['current_price'] = current_price
            position['current_value'] = position['amount'] * current_price
            position['pnl_ratio'] = (current_price - position['buy_price']) / position['buy_price']
    
    def get_btc_ratio(self) -> float:
        """获取当前BTC仓位比例"""
        if self.total_value <= 0:
            return 0
        return self.btc_value / self.total_value
    
    def reset_daily_trading_flags(self, current_date):
        """重置每日交易标志"""
        if self.last_trade_date is None or self.last_trade_date != current_date:
            self.today_bought = False
            self.today_added_position = False
            self.today_sold = False
    
    def can_buy_today(self) -> bool:
        """检查今天是否可以买入（技术指标买入）"""
        return not self.today_bought and not self.today_added_position
    
    def can_add_position_today(self) -> bool:
        """检查今天是否可以加仓（支撑位加仓）"""
        return not self.today_added_position
    
    def can_trade_today(self) -> bool:
        """检查今天是否可以执行任何交易操作（当天卖出后禁止所有操作）"""
        return not self.today_sold
    
    def determine_market_phase(self, news_signal: float) -> str:
        """确定市场阶段"""
        if news_signal >= self.news_high_threshold:
            return 'bullish'
        elif news_signal <= self.news_low_threshold:
            return 'bearish'
        else:
            return 'oscillation'
    
    def get_adx_level(self, adx_value: float) -> str:
        """根据ADX值确定趋势强度等级"""
        if adx_value < self.adx_low_threshold:
            return 'weak'
        elif adx_value <= self.adx_high_threshold:
            return 'medium'
        else:
            return 'strong'
    
    def get_trading_thresholds(self, adx_level: str) -> Tuple[float, float]:
        """根据ADX等级获取买卖阈值"""
        if adx_level == 'weak':
            return 999.0, self.weak_trend_sell_threshold  # 弱趋势只卖出，不买入
        elif adx_level == 'medium':
            return self.medium_trend_buy_threshold, self.medium_trend_sell_threshold
        else:  # strong
            return self.strong_trend_buy_threshold, self.strong_trend_sell_threshold
    
    def get_base_position_ratio(self, adx_level: str) -> float:
        """根据ADX等级获取基础仓位比例"""
        if adx_level == 'weak':
            return 0.15 # 弱趋势基础仓位
        elif adx_level == 'medium':
            return self.medium_trend_base_ratio
        else:  # strong
            return self.strong_trend_base_ratio
    
    def get_stop_levels(self, adx_level: str) -> Tuple[float, float]:
        """根据ADX等级获取止盈止损水平"""
        if adx_level == 'weak':
            return 0.08, 0.05 # 弱趋势止盈止损
        elif adx_level == 'medium':
            return self.medium_trend_stop_profit, self.medium_trend_stop_loss
        else:  # strong
            return self.strong_trend_stop_profit, self.strong_trend_stop_loss
    

    
    def check_weekly_drop(self, current_price: float, timestamp: pd.Timestamp, data: pd.DataFrame) -> bool:
        """
        检查是否出现周跌幅超过阈值的情况，并考虑冷却期
        
        Args:
            current_price: 当前价格
            timestamp: 当前时间
            data: 数据DataFrame
            
        Returns:
            bool: 是否触发周跌幅加仓条件
        """
        # 检查冷却期：如果距离上次周跌幅加仓不足冷却期，则不触发
        if self.last_weekly_drop_buy_date is not None:
            days_since_last_buy = (timestamp.date() - self.last_weekly_drop_buy_date).days
            if days_since_last_buy < self.weekly_drop_cooldown_days:
                return False
        
        # 获取一周前的时间点
        week_ago = timestamp - timedelta(days=7)
        
        # 找到一周前最接近的数据点
        week_ago_data = data[data.index <= week_ago]
        if len(week_ago_data) == 0:
            return False
        
        week_ago_price = week_ago_data.iloc[-1]['close']
        
        # 计算跌幅
        drop_ratio = (week_ago_price - current_price) / week_ago_price
        
        return drop_ratio >= self.weekly_drop_threshold
    

    
    def execute_buy(self, price: float, buy_ratio: float, timestamp: pd.Timestamp, reason: str):
        """执行买入操作"""
        if self.total_value <= 0:
            return
        
        # 计算买入金额
        buy_value = self.total_value * buy_ratio
        
        # 检查最大仓位限制
        projected_btc_value = self.btc_value + buy_value
        projected_ratio = projected_btc_value / self.total_value
        
        if projected_ratio > self.max_btc_ratio:
            max_buy_value = self.total_value * self.max_btc_ratio - self.btc_value
            buy_value = max_buy_value
        
        # 限制买入金额不超过可用现金
        if buy_value > self.cash:
            buy_value = self.cash
        
        if buy_value <= 0:
            return
        
        # 执行买入
        buy_amount = buy_value / price
        trading_fee_amount = buy_value * self.trading_fee
        
        self.btc_amount += buy_amount
        self.cash -= (buy_value + trading_fee_amount)
        
        # 记录交易
        trade = {
            'timestamp': timestamp,
            'action': 'buy',
            'price': price,
            'amount': buy_amount,
            'value': buy_value,
            'trading_fee': trading_fee_amount,
            'reason': reason,
            'cash_after': self.cash,
            'btc_amount_after': self.btc_amount,
            'total_value_after': self.total_value
        }
        self.trades.append(trade)
        
        # 记录持仓
        position = {
            'buy_time': timestamp,
            'buy_price': price,
            'amount': buy_amount,
            'buy_value': buy_value,
            'current_price': price,
            'current_value': buy_value,
            'pnl_ratio': 0
        }
        self.positions.append(position)
        
        print(f"{timestamp}: 买入 {buy_amount:.6f} BTC @ {price:.2f}, 原因: {reason}")
        
        # 更新交易约束状态
        current_date = timestamp.date()
        if "加仓" in reason or "支撑位" in reason:
            self.today_added_position = True
        else:
            self.today_bought = True
        self.last_trade_date = current_date
    
    def execute_sell(self, price: float, sell_ratio: float, timestamp: pd.Timestamp, reason: str):
        """执行卖出操作（增加最低仓位比例保护）"""
        if self.btc_amount <= 0:
            return
        
        # 计算当前BTC仓位比例
        current_btc_ratio = self.get_btc_ratio()
        
        # 计算卖出后的预期BTC仓位比例
        sell_amount = self.btc_amount * sell_ratio
        remaining_btc_amount = self.btc_amount - sell_amount
        remaining_btc_value = remaining_btc_amount * price
        
        # 计算卖出后的总资产价值（近似）
        sell_value = sell_amount * price
        trading_fee_amount = sell_value * self.trading_fee
        projected_cash = self.cash + (sell_value - trading_fee_amount)
        projected_total_value = projected_cash + remaining_btc_value
        
        # 计算卖出后的BTC仓位比例
        projected_btc_ratio = remaining_btc_value / projected_total_value if projected_total_value > 0 else 0
        
        # 检查是否违反最低仓位比例限制
        if projected_btc_ratio < self.min_btc_ratio:
            # 计算允许的最大卖出比例，保持最低仓位
            if current_btc_ratio <= self.min_btc_ratio:
                # 如果当前仓位已经等于或低于最低仓位，不允许卖出
                print(f"{timestamp}: 卖出被阻止 - 当前仓位({current_btc_ratio:.2%})已达最低限制({self.min_btc_ratio:.2%})")
                return
            
            # 计算保持最低仓位的最大卖出比例
            target_btc_value = self.total_value * self.min_btc_ratio
            max_sell_value = self.btc_value - target_btc_value
            max_sell_amount = max_sell_value / price if price > 0 else 0
            max_sell_ratio = max_sell_amount / self.btc_amount if self.btc_amount > 0 else 0
            
            if max_sell_ratio <= 0:
                print(f"{timestamp}: 卖出被阻止 - 无法在保持最低仓位({self.min_btc_ratio:.2%})的情况下卖出")
                return
            
            # 调整卖出比例
            original_sell_ratio = sell_ratio
            sell_ratio = min(sell_ratio, max_sell_ratio)
            sell_amount = self.btc_amount * sell_ratio
            
            print(f"{timestamp}: 卖出比例从 {original_sell_ratio:.2%} 调整为 {sell_ratio:.2%} (保持最低仓位限制)")
        
        # 执行卖出
        sell_value = sell_amount * price
        trading_fee_amount = sell_value * self.trading_fee
        
        self.btc_amount -= sell_amount
        self.cash += (sell_value - trading_fee_amount)
        
        # 记录交易
        trade = {
            'timestamp': timestamp,
            'action': 'sell',
            'price': price,
            'amount': sell_amount,
            'value': sell_value,
            'trading_fee': trading_fee_amount,
            'reason': reason,
            'cash_after': self.cash,
            'btc_amount_after': self.btc_amount,
            'total_value_after': self.total_value
        }
        self.trades.append(trade)
        
        # 更新持仓记录（FIFO原则）
        remaining_sell = sell_amount
        positions_to_remove = []
        
        for i, position in enumerate(self.positions):
            if remaining_sell <= 0:
                break
            
            if position['amount'] <= remaining_sell:
                remaining_sell -= position['amount']
                positions_to_remove.append(i)
            else:
                position['amount'] -= remaining_sell
                position['current_value'] = position['amount'] * price
                remaining_sell = 0
        
        # 移除已完全卖出的持仓
        for i in reversed(positions_to_remove):
            self.positions.pop(i)
        
        print(f"{timestamp}: 卖出 {sell_amount:.6f} BTC @ {price:.2f}, 原因: {reason}")
        
        # 更新交易约束状态
        current_date = timestamp.date()
        self.today_sold = True
        self.last_trade_date = current_date
        
        # 更新账户状态并检查最终仓位比例
        self.update_account_status(price)
        final_btc_ratio = self.get_btc_ratio()
        print(f"{timestamp}: 卖出后BTC仓位比例: {final_btc_ratio:.2%} (最低限制: {self.min_btc_ratio:.2%})")
        print(f"{timestamp}: 当天已卖出，今日不再执行任何交易操作")
    
    def check_stop_conditions(self, current_price: float, timestamp: pd.Timestamp, adx_level: str) -> bool:
        """检查止盈止损条件"""
        stop_profit, stop_loss = self.get_stop_levels(adx_level)
        
        stop_profit_positions = []
        stop_loss_positions = []
        
        for i, position in enumerate(self.positions):
            pnl_ratio = (current_price - position['buy_price']) / position['buy_price']
            
            if pnl_ratio >= stop_profit:
                stop_profit_positions.append(i)
            elif pnl_ratio <= -stop_loss:
                stop_loss_positions.append(i)
        
        traded = False
        
        # 处理止盈
        if stop_profit_positions:
            profit_close_amount = sum(self.positions[i]['amount'] for i in stop_profit_positions)
            profit_close_ratio = profit_close_amount / self.btc_amount if self.btc_amount > 0 else 0
            
            if profit_close_ratio > 0:
                reason = f"止盈({adx_level}): {len(stop_profit_positions)}笔持仓"
                self.execute_sell(current_price, profit_close_ratio, timestamp, reason)
                traded = True
        
        # 处理止损
        if stop_loss_positions:
            loss_close_amount = sum(self.positions[i]['amount'] for i in stop_loss_positions)
            loss_close_ratio = loss_close_amount / self.btc_amount if self.btc_amount > 0 else 0
            
            if loss_close_ratio > 0:
                reason = f"止损({adx_level}): {len(stop_loss_positions)}笔持仓"
                self.execute_sell(current_price, loss_close_ratio, timestamp, reason)
                traded = True
        
        return traded
    
    def get_keltner_support_resistance_levels(self, row: pd.Series) -> Tuple[float, float, float]:
        """
        获取肯特纳通道的支撑压力位
        
        Args:
            row: 当前数据行
            
        Returns:
            Tuple[float, float, float]: (下线, 中线, 上线)
            注意：当肯特纳数据不足时，返回None表示无效
        """
        keltner_upper = row['keltner_upper']
        keltner_middle = row['keltner_middle'] 
        keltner_lower = row['keltner_lower']
        
        # 如果没有有效的肯特纳数据，返回None
        if pd.isna(keltner_upper) or pd.isna(keltner_middle) or pd.isna(keltner_lower):
            return None, None, None
        
        return keltner_lower, keltner_middle, keltner_upper
    
    def determine_keltner_position(self, current_price: float, keltner_lower: float, 
                                  keltner_middle: float, keltner_upper: float) -> str:
        """
        判断当前价格在肯特纳通道中的位置
        
        Args:
            current_price: 当前价格
            keltner_lower: 下线
            keltner_middle: 中线
            keltner_upper: 上线
            
        Returns:
            str: 'upper' (上线和中线之间), 'lower' (下线和中线之间), 'above' (上线之上), 'below' (下线之下)
        """
        if current_price > keltner_upper:
            return 'above'
        elif current_price < keltner_lower:
            return 'below'
        elif current_price > keltner_middle:
            return 'upper'  # 在中线和上线之间
        else:
            return 'lower'  # 在中线和下线之间
    
    def check_keltner_breakthrough(self, data: pd.DataFrame, current_timestamp: pd.Timestamp, 
                                   lookback_days: int = 7) -> Tuple[str, str]:
        """
        检查肯特纳通道突破情况
        
        Args:
            data: 完整的数据DataFrame
            current_timestamp: 当前时间戳
            lookback_days: 回看天数（默认7天）
            
        Returns:
            Tuple[str, str]: (突破类型, 突破方向)
            突破类型: 'upper', 'middle', 'lower', 'none'
            突破方向: 'up', 'down', 'none'
        """
        # 计算7天前的时间戳
        week_ago = current_timestamp - timedelta(days=lookback_days)
        
        # 获取当前数据
        current_row = data.loc[current_timestamp]
        
        # 获取过去一周的历史数据
        historical_data = data[(data.index >= week_ago) & (data.index < current_timestamp)]
        
        # 如果历史数据不足，返回无突破
        if len(historical_data) < lookback_days * 24:  # 至少需要一定数量的数据点
            return 'none', 'none'
        
        current_high = current_row['high']
        current_low = current_row['low']
        current_upper = current_row['keltner_upper']
        current_middle = current_row['keltner_middle']
        current_lower = current_row['keltner_lower']
        
        # 检查过去一周是否都在某个水平之下/之上
        historical_highs = historical_data['high']
        historical_lows = historical_data['low']
        historical_uppers = historical_data['keltner_upper']
        historical_middles = historical_data['keltner_middle']
        historical_lowers = historical_data['keltner_lower']
        
        # 上穿上线：过去一周最高价都不超过上线，当前最高价突破上线
        if len(historical_highs) > 0 and (historical_highs <= historical_uppers).all() and current_high > current_upper:
            return 'upper', 'up'
        
        # 上穿中线：过去一周最高价都不超过中线，当前最高价突破中线
        elif len(historical_highs) > 0 and (historical_highs <= historical_middles).all() and current_high > current_middle:
            return 'middle', 'up'
        
        # 下穿下线：过去一周最低价都不低于下线，当前最低价跌破下线
        elif len(historical_lows) > 0 and (historical_lows >= historical_lowers).all() and current_low < current_lower:
            return 'lower', 'down'
        
        # 下穿中线：过去一周最低价都不低于中线，当前最低价跌破中线
        elif len(historical_lows) > 0 and (historical_lows >= historical_middles).all() and current_low < current_middle:
            return 'middle', 'down'
        
        return 'none', 'none'
    
    def get_adx_based_ratios(self, adx_level: str) -> Dict[str, float]:
        """
        根据ADX等级获取各种交易比例
        
        Args:
            adx_level: ADX等级 ('medium' 或 'strong')
            
        Returns:
            Dict: 包含各种比例的字典
        """
        if adx_level == 'medium':
            return {
                'base_buy_ratio': self.medium_trend_base_ratio,
                'upper_sell_ratio': self.medium_trend_upper_sell_ratio,      # 上穿上线减仓比例
                'middle_sell_ratio': self.medium_trend_middle_sell_ratio,    # 上穿中线减仓比例
                'lower_buy_ratio': self.medium_trend_lower_buy_ratio,        # 下穿下线加仓比例
                'middle_buy_ratio': self.medium_trend_middle_buy_ratio,      # 下穿中线加仓比例
                'buy_threshold': self.medium_trend_buy_threshold,
                'sell_threshold': self.medium_trend_sell_threshold,
                'stop_profit': self.medium_trend_stop_profit,
                'stop_loss': self.medium_trend_stop_loss
            }
        elif adx_level == 'strong':
            return {
                'base_buy_ratio': self.strong_trend_base_ratio,
                'upper_sell_ratio': self.strong_trend_upper_sell_ratio,      # 上穿上线减仓比例（更大）
                'middle_sell_ratio': self.strong_trend_middle_sell_ratio,    # 上穿中线减仓比例（更大）
                'lower_buy_ratio': self.strong_trend_lower_buy_ratio,        # 下穿下线加仓比例（更小）
                'middle_buy_ratio': self.strong_trend_middle_buy_ratio,      # 下穿中线加仓比例（更小）
                'buy_threshold': self.strong_trend_buy_threshold,
                'sell_threshold': self.strong_trend_sell_threshold,
                'stop_profit': self.strong_trend_stop_profit,
                'stop_loss': self.strong_trend_stop_loss
            }
        else: # weak
            return {
                'base_buy_ratio': 0.0, # 弱趋势不买入
                'upper_sell_ratio': self.weak_trend_sell_threshold, # 弱趋势卖出阈值
                'middle_sell_ratio': self.weak_trend_sell_threshold, # 弱趋势卖出阈值
                'lower_buy_ratio': 0.0, # 弱趋势不加仓
                'middle_buy_ratio': 0.0, # 弱趋势不加仓
                'buy_threshold': 0.0, # 弱趋势不买入
                'sell_threshold': self.weak_trend_sell_threshold, # 弱趋势卖出阈值
                'stop_profit': 0.0, # 弱趋势不止盈
                'stop_loss': 0.0 # 弱趋势不止损
            }
    
    def run_strategy(self, data: pd.DataFrame) -> Dict:
        """
        重新设计的震荡市场策略
        
        核心逻辑：
        1. 只在震荡阶段执行交易
        2. ADX < 20时不操作
        3. ADX 20-40 和 >40 使用不同参数
        4. 基于MA的动态支撑压力位
        5. 穿越检测和延迟交易
        """
        from tqdm import tqdm
        
        print(f"开始运行重新设计的震荡市场策略，共 {len(data)} 个15分钟K线...")
        
        # 统计各阶段的时间点数量
        phase_stats = {'oscillation': 0, 'bullish': 0, 'bearish': 0, 'trading_points': 0}
        
        # 延迟交易标记
        pending_trades = []  # 存储待执行的交易
        
        prev_row = None
        for timestamp, row in tqdm(data.iterrows(), 
                                  total=len(data), 
                                  desc="运行策略"):
            current_price = row['close']
            current_high = row['high']
            current_low = row['low']
            current_open = row['open']
            tech_signal = row['tech_signal']
            adx_value = row['adx']
            news_signal = row['news_signal']
            current_date = timestamp.date()
            
            # 重置每日交易标志
            self.reset_daily_trading_flags(current_date)
            
            # 更新账户状态
            self.update_account_status(current_price)
            
            # 确定市场阶段
            new_phase = self.determine_market_phase(news_signal)
            phase_stats[new_phase] += 1
            
            # 检查市场阶段是否发生变化
            if new_phase != self.market_phase:
                print(f"{timestamp}: 市场阶段从 {self.market_phase} 转为 {new_phase}")
                self.market_phase = new_phase
                self.current_news_signal = news_signal
            
            # 确定ADX等级
            adx_level = self.get_adx_level(adx_value)
            
            # 执行待处理的交易（基于上一期的穿越检测）
            if pending_trades:
                for trade_info in pending_trades:
                    if trade_info['action'] == 'buy':
                        self.execute_buy(current_open, trade_info['ratio'], timestamp, trade_info['reason'])
                    elif trade_info['action'] == 'sell':
                        self.execute_sell(current_open, trade_info['ratio'], timestamp, trade_info['reason'])
                pending_trades.clear()
            
            # 检查周跌幅加仓条件（在任何ADX阶段都适用）
            weekly_drop_triggered = self.check_weekly_drop(current_price, timestamp, data)
            if weekly_drop_triggered:
                # 检查是否可以加仓（当天未加仓）
                if self.can_add_position_today():
                    self.execute_buy(current_price, self.weekly_drop_buy_ratio, timestamp, f"周跌幅加仓({adx_level})")
                    # 记录周跌幅加仓日期，用于冷却期计算
                    self.last_weekly_drop_buy_date = timestamp.date()
                else:
                    print(f"{timestamp}: 周跌幅加仓被阻止 - 当天已加仓")
            
            # 只在震荡阶段执行交易策略
            if self.market_phase == 'oscillation':
                phase_stats['trading_points'] += 1
                
                # 检查当天是否已卖出（如果已卖出则不再执行任何操作）
                if not self.can_trade_today():
                    # 当天已卖出，跳过所有交易逻辑
                    pass
                # ADX < 20时（弱趋势）：只执行技术指标减仓，不买入
                elif adx_value < self.adx_low_threshold:
                    if self.btc_amount > 0 and tech_signal <= self.weak_trend_sell_threshold:
                        # 弱趋势技术指标减仓
                        self.execute_sell(current_price, self.weak_trend_sell_ratio, timestamp, f"弱趋势技术指标减仓{self.weak_trend_sell_ratio:.1%}(ADX={adx_value:.1f})")
                # ADX >= 20时才执行完整交易策略
                elif adx_value >= self.adx_low_threshold:
                    # 获取当前ADX等级下的交易参数
                    if adx_level in ['medium', 'strong']:
                        ratios = self.get_adx_based_ratios(adx_level)
                    
                        # 1. 检查止盈止损
                        if self.btc_amount > 0:
                            stop_profit_positions = []
                            stop_loss_positions = []
                            
                            for pos_idx, position in enumerate(self.positions):
                                pnl_ratio = (current_price - position['buy_price']) / position['buy_price']
                                
                                if pnl_ratio >= ratios['stop_profit']:
                                    stop_profit_positions.append(pos_idx)
                                elif pnl_ratio <= -ratios['stop_loss']:
                                    stop_loss_positions.append(pos_idx)
                            
                            # 执行止盈止损（卖出所有持仓）
                            if stop_profit_positions or stop_loss_positions:
                                reason = f"止盈({len(stop_profit_positions)}笔)" if stop_profit_positions else f"止损({len(stop_loss_positions)}笔)"
                                self.execute_sell(current_price, 1.0, timestamp, f"{reason}({adx_level})")
                        
                        # 2. 检查合成信号
                        if tech_signal >= ratios['buy_threshold']:
                            # 检查是否可以买入（当天未买入且未加仓）
                            if self.can_buy_today():
                                # 检查是否同时突破肯特纳通道（信号冲突检查）
                                breakthrough_type, breakthrough_direction = self.check_keltner_breakthrough(data, timestamp)
                                
                                # 如果同时上穿肯特纳通道，则不执行技术指标买入（信号冲突）
                                if breakthrough_type in ['upper', 'middle'] and breakthrough_direction == 'up':
                                    print(f"{timestamp}: 技术指标买入信号被阻止 - 同时上穿肯特纳{breakthrough_type}线（信号冲突）")
                                else:
                                    # 合成信号买入
                                    self.execute_buy(current_price, ratios['base_buy_ratio'], timestamp, f"合成信号买入({adx_level})")
                            else:
                                print(f"{timestamp}: 技术指标买入信号被阻止 - 当天已买入或加仓")
                        
                        elif tech_signal <= ratios['sell_threshold'] and self.btc_amount > 0:
                            # 检查是否同时跌破肯特纳通道（信号冲突检查）
                            breakthrough_type, breakthrough_direction = self.check_keltner_breakthrough(data, timestamp)
                            
                            # 如果同时下穿肯特纳通道，则不执行技术指标卖出（信号冲突）
                            if breakthrough_type in ['lower', 'middle'] and breakthrough_direction == 'down':
                                print(f"{timestamp}: 技术指标卖出信号被阻止 - 同时下穿肯特纳{breakthrough_type}线（信号冲突）")
                            else:
                                # 合成信号卖出所有持仓
                                self.execute_sell(current_price, 1.0, timestamp, f"合成信号卖出({adx_level})")
                        
                        # 3. 肯特纳通道交易逻辑（只有在有持仓时才执行）
                        if self.btc_amount > 0:
                            # 检查肯特纳通道突破
                            breakthrough_type, breakthrough_direction = self.check_keltner_breakthrough(data, timestamp)
                            
                            if breakthrough_type != 'none' and breakthrough_direction != 'none':
                                if breakthrough_direction == 'up':
                                    # 上穿肯特纳通道，执行减仓
                                    if breakthrough_type == 'upper':
                                        # 上穿上线，下一期开盘价减仓
                                        current_ratio = self.get_btc_ratio()
                                        reduce_ratio = min(ratios['upper_sell_ratio'], current_ratio)
                                        if reduce_ratio > 0:
                                            pending_trades.append({
                                                'action': 'sell',
                                                'ratio': reduce_ratio,
                                                'reason': f"上穿肯特纳上线减仓({adx_level})"
                                            })
                                    elif breakthrough_type == 'middle':
                                        # 上穿中线，下一期开盘价减仓
                                        current_ratio = self.get_btc_ratio()
                                        reduce_ratio = min(ratios['middle_sell_ratio'], current_ratio)
                                        if reduce_ratio > 0:
                                            pending_trades.append({
                                                'action': 'sell',
                                                'ratio': reduce_ratio,
                                                'reason': f"上穿肯特纳中线减仓({adx_level})"
                                            })
                                
                                elif breakthrough_direction == 'down':
                                    # 下穿肯特纳通道，执行加仓
                                    if breakthrough_type == 'lower':
                                        # 下穿下线，检查是否可以加仓
                                        if self.can_add_position_today():
                                            pending_trades.append({
                                                'action': 'buy',
                                                'ratio': ratios['lower_buy_ratio'],
                                                'reason': f"下穿肯特纳下线加仓({adx_level})"
                                            })
                                        else:
                                            print(f"{timestamp}: 下穿肯特纳下线加仓被阻止 - 当天已加仓")
                                    elif breakthrough_type == 'middle':
                                        # 下穿中线，检查是否可以加仓
                                        if self.can_add_position_today():
                                            pending_trades.append({
                                                'action': 'buy',
                                                'ratio': ratios['middle_buy_ratio'],
                                                'reason': f"下穿肯特纳中线加仓({adx_level})"
                                            })
                                        else:
                                            print(f"{timestamp}: 下穿肯特纳中线加仓被阻止 - 当天已加仓")
                    
            else:
                # 在看多或看空阶段，不执行震荡策略，但可以检查止盈止损
                if self.btc_amount > 0 and self.can_trade_today():
                    # 在非震荡阶段使用中等趋势的止盈止损参数
                    medium_ratios = self.get_adx_based_ratios('medium')
                    
                    stop_profit_positions = []
                    stop_loss_positions = []
                    
                    for pos_idx, position in enumerate(self.positions):
                        pnl_ratio = (current_price - position['buy_price']) / position['buy_price']
                        
                        if pnl_ratio >= medium_ratios['stop_profit']:
                            stop_profit_positions.append(pos_idx)
                        elif pnl_ratio <= -medium_ratios['stop_loss']:
                            stop_loss_positions.append(pos_idx)
                    
                    if stop_profit_positions or stop_loss_positions:
                        reason = f"止盈({len(stop_profit_positions)}笔)" if stop_profit_positions else f"止损({len(stop_loss_positions)}笔)"
                        self.execute_sell(current_price, 1.0, timestamp, f"{reason}(非震荡)")
            
            # 保存当前行作为下一次循环的前一行
            prev_row = row
            
            # 记录状态
            keltner_lower, keltner_middle, keltner_upper = self.get_keltner_support_resistance_levels(row)
            record = {
                'timestamp': timestamp,
                'price': current_price,
                'high': current_high,
                'low': current_low,
                'cash': self.cash,
                'btc_amount': self.btc_amount,
                'btc_value': self.btc_value,
                'total_value': self.total_value,
                'btc_ratio': self.get_btc_ratio(),
                'tech_signal': tech_signal,
                'adx': adx_value,
                'adx_level': adx_level,
                'news_signal': news_signal,
                'market_phase': self.market_phase,
                'positions_count': len(self.positions),
                'keltner_lower': keltner_lower if keltner_lower is not None else np.nan,
                'keltner_middle': keltner_middle if keltner_middle is not None else np.nan,
                'keltner_upper': keltner_upper if keltner_upper is not None else np.nan,
                'weekly_drop_triggered': weekly_drop_triggered,
                'pending_trades_count': len(pending_trades),
                'today_bought': self.today_bought,
                'today_added_position': self.today_added_position,
                'today_sold': self.today_sold,
                'can_buy_today': self.can_buy_today(),
                'can_add_position_today': self.can_add_position_today(),
                'can_trade_today': self.can_trade_today(),
                'last_weekly_drop_buy_date': self.last_weekly_drop_buy_date,
                'days_since_last_weekly_drop_buy': (current_date - self.last_weekly_drop_buy_date).days if self.last_weekly_drop_buy_date else None
            }
            self.records.append(record)
        
        print("重新设计的震荡市场策略运行完成！")
        print(f"阶段统计: {phase_stats}")
        print(f"震荡阶段占比: {phase_stats['oscillation'] / len(data) * 100:.1f}%")
        print(f"实际交易时间点: {phase_stats['trading_points']}")
        
        return self.get_strategy_results()
    
    def get_strategy_results(self) -> Dict:
        """获取策略结果"""
        if not self.records:
            return {}
        
        # 计算总收益
        final_value = self.records[-1]['total_value']
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        # 计算最大回撤
        values = [record['total_value'] for record in self.records]
        peak = values[0]
        max_drawdown = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 交易统计
        buy_trades = [t for t in self.trades if t['action'] == 'buy']
        sell_trades = [t for t in self.trades if t['action'] == 'sell']
        
        # 计算总手续费
        total_trading_fees = sum(t.get('trading_fee', 0) for t in self.trades)
        
        # 计算胜率
        win_rate = self._calculate_win_rate(buy_trades, sell_trades)
        
        # 计算年化收益率
        records_df = pd.DataFrame(self.records)
        records_df['date'] = pd.to_datetime(records_df['timestamp']).dt.date
        trading_days = records_df['date'].nunique()
        annualized_return = self._calculate_annualized_return(total_return, trading_days)
        
        # 计算夏普比率
        sharpe_ratio = self._calculate_sharpe_ratio(records_df)
        
        # 计算最大连续亏损天数
        max_consecutive_losses = self._calculate_max_consecutive_losses(records_df)
        
        # 计算盈亏比
        profit_loss_ratio = self._calculate_profit_loss_ratio(buy_trades, sell_trades)
        
        # 计算波动率
        volatility = self._calculate_volatility(records_df)
        
        # ADX等级分布统计
        adx_distribution = records_df['adx_level'].value_counts().to_dict()
        
        # 市场阶段分布统计
        phase_distribution = records_df['market_phase'].value_counts().to_dict()
        
        # 计算震荡阶段的交易统计
        oscillation_records = records_df[records_df['market_phase'] == 'oscillation']
        oscillation_ratio = len(oscillation_records) / len(records_df) if len(records_df) > 0 else 0
        
        results = {
            'strategy_type': 'volatile_market',
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'max_consecutive_losses': max_consecutive_losses,
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_trading_fees': total_trading_fees,
            'final_btc_amount': self.btc_amount,
            'final_cash': self.cash,
            'final_btc_ratio': self.get_btc_ratio(),
            'adx_distribution': adx_distribution,
            'phase_distribution': phase_distribution,
            'oscillation_ratio': oscillation_ratio,
            'trades': self.trades,
            'positions': self.positions,
            'records': pd.DataFrame(self.records)
        }
        
        return results
    
    def _calculate_win_rate(self, buy_trades: List, sell_trades: List) -> float:
        """计算胜率"""
        if not buy_trades or not sell_trades:
            return 0.0
        
        import copy
        buy_queue = copy.deepcopy(buy_trades)
        sell_queue = copy.deepcopy(sell_trades)
        
        wins = 0
        total_trades = 0
        
        for sell_trade in sell_queue:
            sell_price = sell_trade['price']
            sell_amount = sell_trade['amount']
            remaining_sell = sell_amount
            
            while remaining_sell > 0 and buy_queue:
                buy_trade = buy_queue[0]
                buy_price = buy_trade['price']
                buy_amount = buy_trade['amount']
                
                matched_amount = min(remaining_sell, buy_amount)
                
                if sell_price > buy_price:
                    wins += 1
                total_trades += 1
                
                remaining_sell -= matched_amount
                buy_trade['amount'] -= matched_amount
                
                if buy_trade['amount'] <= 0:
                    buy_queue.pop(0)
        
        return wins / total_trades if total_trades > 0 else 0.0
    
    def _calculate_annualized_return(self, total_return: float, trading_days: int) -> float:
        """计算年化收益率"""
        if trading_days <= 0:
            return 0.0
        
        years = trading_days / 252.0
        if years <= 0:
            return 0.0
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return
    
    def _calculate_sharpe_ratio(self, records_df: pd.DataFrame) -> float:
        """计算夏普比率"""
        if len(records_df) < 2:
            return 0.0
        
        daily_values = records_df.groupby('date')['total_value'].last().sort_index()
        
        if len(daily_values) < 2:
            return 0.0
        
        daily_returns = []
        for i in range(1, len(daily_values)):
            daily_return = (daily_values.iloc[i] - daily_values.iloc[i-1]) / daily_values.iloc[i-1]
            daily_returns.append(daily_return)
        
        if not daily_returns:
            return 0.0
        
        avg_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)
        
        if std_daily_return == 0:
            return 0.0
        
        sharpe_ratio = (avg_daily_return * 252) / (std_daily_return * np.sqrt(252))
        return sharpe_ratio
    
    def _calculate_max_consecutive_losses(self, records_df: pd.DataFrame) -> int:
        """计算最大连续亏损天数"""
        if len(records_df) < 2:
            return 0
        
        daily_values = records_df.groupby('date')['total_value'].last().sort_index()
        
        if len(daily_values) < 2:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for i in range(1, len(daily_values)):
            if daily_values.iloc[i] < daily_values.iloc[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_profit_loss_ratio(self, buy_trades: List, sell_trades: List) -> float:
        """计算盈亏比"""
        if not buy_trades or not sell_trades:
            return 0.0
        
        import copy
        buy_queue = copy.deepcopy(buy_trades)
        sell_queue = copy.deepcopy(sell_trades)
        
        profits = []
        losses = []
        
        for sell_trade in sell_queue:
            sell_price = sell_trade['price']
            sell_amount = sell_trade['amount']
            remaining_sell = sell_amount
            
            while remaining_sell > 0 and buy_queue:
                buy_trade = buy_queue[0]
                buy_price = buy_trade['price']
                buy_amount = buy_trade['amount']
                
                matched_amount = min(remaining_sell, buy_amount)
                pnl = (sell_price - buy_price) * matched_amount
                
                if pnl > 0:
                    profits.append(pnl)
                elif pnl < 0:
                    losses.append(abs(pnl))
                
                remaining_sell -= matched_amount
                buy_trade['amount'] -= matched_amount
                
                if buy_trade['amount'] <= 0:
                    buy_queue.pop(0)
        
        if not profits:
            return 0.0
        
        if not losses:
            return 999.0
        
        avg_profit = np.mean(profits)
        avg_loss = np.mean(losses)
        
        return avg_profit / avg_loss if avg_loss > 0 else 0.0
    
    def _calculate_volatility(self, records_df: pd.DataFrame) -> float:
        """计算年化波动率"""
        if len(records_df) < 2:
            return 0.0
        
        daily_values = records_df.groupby('date')['total_value'].last().sort_index()
        
        if len(daily_values) < 2:
            return 0.0
        
        daily_returns = []
        for i in range(1, len(daily_values)):
            daily_return = (daily_values.iloc[i] - daily_values.iloc[i-1]) / daily_values.iloc[i-1]
            daily_returns.append(daily_return)
        
        if not daily_returns:
            return 0.0
        
        daily_volatility = np.std(daily_returns)
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility





# 使用示例
def main():
    """震荡市场策略使用示例"""
    print("="*60)
    print("BTC 震荡市场交易策略")
    print("="*60)
    
    # 创建策略实例
    print("正在初始化震荡市场策略...")
    strategy = VolatileMarketStrategy(
        min_btc_ratio=0.1,
        max_btc_ratio=0.8,
        initial_cash=100000,
        
        # 新闻信号阈值
        news_high_threshold=1.5,
        news_low_threshold=0.5,
        
        # ADX参数
        adx_period=14,
        adx_low_threshold=20,
        adx_high_threshold=40,
        
        # 中等趋势参数
        medium_trend_buy_threshold=1.5,
        medium_trend_sell_threshold=0.5,
        medium_trend_base_ratio=0.25,
        medium_trend_stop_profit=0.08,
        medium_trend_stop_loss=0.05,
        
        # 强趋势参数
        strong_trend_buy_threshold=1.6,
        strong_trend_sell_threshold=0.4,
        strong_trend_base_ratio=0.15,
        strong_trend_stop_profit=0.04,
        strong_trend_stop_loss=0.03,
        
                 # 弱趋势参数
         weak_trend_sell_threshold=0.6,
         weak_trend_sell_ratio=0.3,
         
         # 周跌幅加仓参数（含冷却期）
         weekly_drop_cooldown_days=7,
         
         # 肯特纳通道参数
        keltner_period=50,
        keltner_atr_period=50,
        keltner_multiplier=3.75,
        
        # 中等趋势肯特纳通道交易参数
        medium_trend_upper_sell_ratio=0.20,
        medium_trend_middle_sell_ratio=0.15,
        medium_trend_lower_buy_ratio=0.15,
        medium_trend_middle_buy_ratio=0.10,
        
        # 强趋势肯特纳通道交易参数
        strong_trend_upper_sell_ratio=0.25,
        strong_trend_middle_sell_ratio=0.20,
        strong_trend_lower_buy_ratio=0.10,
        strong_trend_middle_buy_ratio=0.08
    )
    
    # 示例数据
    dates_15min = pd.date_range('2024-01-01', '2024-01-31', freq='15min')
    min15_klines = pd.DataFrame({
        'open': np.random.randn(len(dates_15min)).cumsum() + 50000,
        'high': np.random.randn(len(dates_15min)).cumsum() + 52000,
        'low': np.random.randn(len(dates_15min)).cumsum() + 48000,
        'close': np.random.randn(len(dates_15min)).cumsum() + 50000,
        'volume': np.random.randint(100, 1000, len(dates_15min))
    }, index=dates_15min)
    
    # 确保high >= close >= low
    min15_klines['high'] = np.maximum(min15_klines['high'], min15_klines['close'])
    min15_klines['low'] = np.minimum(min15_klines['low'], min15_klines['close'])
    
    # 生成示例新闻信号（模拟不同市场阶段）
    news_times = pd.date_range('2024-01-01', '2024-01-31', freq='6H') + pd.Timedelta(minutes=np.random.randint(0, 60))
    # 创建一个包含多个阶段的新闻信号：前1/3看多，中间1/3震荡，后1/3看空
    news_values = []
    for i in range(len(news_times)):
        if i < len(news_times) // 3:
            # 前1/3时间：看多信号
            news_values.append(np.random.choice([1.6, 1.7, 1.8], 1)[0])
        elif i < 2 * len(news_times) // 3:
            # 中间1/3时间：震荡信号
            news_values.append(np.random.choice([0.8, 1.0, 1.2], 1)[0])
        else:
            # 后1/3时间：看空信号
            news_values.append(np.random.choice([0.3, 0.4, 0.5], 1)[0])
    
    news_signals = pd.Series(news_values, index=news_times)
    
    # 准备数据
    print("\n正在准备数据...")
    data = strategy.prepare_data(min15_klines, news_signals)
    
    # 运行策略
    print("\n正在运行策略...")
    results = strategy.run_strategy(data)
    
    # 显示结果
    print("\n" + "="*60)
    print("震荡市场策略回测完成!")
    print("="*60)
    print(f"策略类型: {results['strategy_type']}")
    print(f"初始资金: {results['initial_cash']:,.2f}")
    print(f"最终价值: {results['final_value']:,.2f}")
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"年化收益率: {results['annualized_return']:.2%}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"胜率: {results['win_rate']:.2%}")
    print(f"盈亏比: {results['profit_loss_ratio']:.2f}")
    print(f"总交易次数: {results['total_trades']}")
    print(f"最终BTC持仓: {results['final_btc_amount']:.6f}")
    print(f"最终仓位比例: {results['final_btc_ratio']:.2%}")
    print(f"ADX等级分布: {results['adx_distribution']}")
    print(f"市场阶段分布: {results['phase_distribution']}")
    print(f"震荡阶段占比: {results['oscillation_ratio']:.2%}")
    print("="*60)
    
    return strategy, results


if __name__ == "__main__":
    strategy, results = main()
