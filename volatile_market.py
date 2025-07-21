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
                 
                 # 仓位参数
                 medium_trend_base_ratio: float = 0.25,     # 中等趋势基础仓位比例
                 strong_trend_base_ratio: float = 0.15,     # 强趋势基础仓位比例
                 
                 # 止盈止损参数
                 medium_trend_stop_profit: float = 0.08,    # 中等趋势止盈比例
                 medium_trend_stop_loss: float = 0.05,      # 中等趋势止损比例
                 strong_trend_stop_profit: float = 0.04,    # 强趋势止盈比例
                 strong_trend_stop_loss: float = 0.03,      # 强趋势止损比例
                 
                 # MA支撑压力位参数
                 ma_periods: List[int] = [30, 120],         # MA周期列表
                 
                 # 中等趋势MA支撑压力位参数
                 medium_trend_support_buy_ratio: float = 0.15,     # 中等趋势支撑位加仓比例
                 medium_trend_resistance_sell_ratio: float = 0.20, # 中等趋势压力位减仓比例
                 
                 # 强趋势MA支撑压力位参数
                 strong_trend_support_buy_ratio: float = 0.10,     # 强趋势支撑位加仓比例
                 strong_trend_resistance_sell_ratio: float = 0.25, # 强趋势压力位减仓比例
                 
                 # 周跌幅加仓参数
                 weekly_drop_threshold: float = 0.15,       # 周跌幅阈值（15%）
                 weekly_drop_buy_ratio: float = 0.1,        # 周跌幅加仓比例
                 
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
        
        # 仓位参数
        self.medium_trend_base_ratio = medium_trend_base_ratio
        self.strong_trend_base_ratio = strong_trend_base_ratio
        
        # 止盈止损参数
        self.medium_trend_stop_profit = medium_trend_stop_profit
        self.medium_trend_stop_loss = medium_trend_stop_loss
        self.strong_trend_stop_profit = strong_trend_stop_profit
        self.strong_trend_stop_loss = strong_trend_stop_loss
        
        # MA支撑压力位参数
        self.ma_periods = ma_periods
        self.medium_trend_support_buy_ratio = medium_trend_support_buy_ratio
        self.medium_trend_resistance_sell_ratio = medium_trend_resistance_sell_ratio
        self.strong_trend_support_buy_ratio = strong_trend_support_buy_ratio
        self.strong_trend_resistance_sell_ratio = strong_trend_resistance_sell_ratio
        
        # 周跌幅加仓参数
        self.weekly_drop_threshold = weekly_drop_threshold
        self.weekly_drop_buy_ratio = weekly_drop_buy_ratio
        
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
        ma_values = {period: [] for period in self.ma_periods}
        
        # 技术指标所需的最大回看期数
        max_lookback = max(
            self.signal_generator.macd_params['slow'] + self.signal_generator.macd_params['signal'],
            self.signal_generator.boll_params['period'],
            self.signal_generator.rsi_params['period'],
            self.adx_period,
            max(self.ma_periods)
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
                
                # 计算各个MA
                for period in self.ma_periods:
                    if len(historical_daily) >= period:
                        ma = talib.SMA(historical_daily['close'].values, timeperiod=period)
                        ma_value = ma[-1] if not np.isnan(ma[-1]) else current_close
                    else:
                        ma_value = current_close
                    ma_values[period].append(ma_value)
                
            else:
                tech_signal = 1.00
                adx_value = 25.0
                for period in self.ma_periods:
                    ma_values[period].append(current_close)
            
            tech_signals.append(tech_signal)
            adx_values.append(adx_value)
        
        # 将计算结果添加到15分钟数据
        min15_klines['tech_signal'] = tech_signals
        min15_klines['adx'] = adx_values

        
        for period in self.ma_periods:
            min15_klines[f'ma_{period}'] = ma_values[period]
        
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
            return 1.3, 0.7 # 弱趋势阈值
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
    
    def check_ma_support_resistance(self, current_price: float, current_high: float, current_low: float, 
                                   row: pd.Series, prev_row: pd.Series = None) -> Tuple[bool, bool]:
        """
        检查是否触及MA支撑位或压力位
        支撑位：K线从上向下穿过MA线（最低价穿过）
        压力位：K线从下向上穿过MA线（最高价穿过）
        
        Args:
            current_price: 当前收盘价
            current_high: 当前最高价
            current_low: 当前最低价
            row: 当前时间点的数据
            prev_row: 前一个时间点的数据
            
        Returns:
            Tuple[bool, bool]: (是否触及支撑位, 是否触及压力位)
        """
        is_support = False
        is_resistance = False
        
        # 如果没有前一个时间点的数据，无法判断穿越
        if prev_row is None:
            return is_support, is_resistance
        
        prev_high = prev_row['high']
        prev_low = prev_row['low']
        
        for period in self.ma_periods:
            ma_value = row[f'ma_{period}']
            prev_ma_value = prev_row[f'ma_{period}']
            
            if pd.isna(ma_value) or pd.isna(prev_ma_value):
                continue
            
            # 支撑位判断：前一K线在MA上方，当前K线最低价穿过MA
            if prev_low > prev_ma_value and current_low <= ma_value:
                is_support = True
            
            # 压力位判断：前一K线在MA下方，当前K线最高价穿过MA
            if prev_high < prev_ma_value and current_high >= ma_value:
                is_resistance = True
        
        return is_support, is_resistance
    
    def check_weekly_drop(self, current_price: float, timestamp: pd.Timestamp, data: pd.DataFrame) -> bool:
        """
        检查是否出现周跌幅超过阈值的情况
        
        Args:
            current_price: 当前价格
            timestamp: 当前时间
            data: 数据DataFrame
            
        Returns:
            bool: 是否触发周跌幅加仓条件
        """
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
    
    def get_ma_support_resistance_levels(self, row: pd.Series, current_price: float) -> Tuple[float, float]:
        """
        获取当前的支撑位和压力位
        
        Args:
            row: 当前数据行
            current_price: 当前价格
            
        Returns:
            Tuple[float, float]: (支撑位, 压力位)
        """
        ma_values = []
        for period in self.ma_periods:
            ma_value = row[f'ma_{period}']
            if not pd.isna(ma_value):
                ma_values.append(ma_value)
        
        if len(ma_values) == 0:
            return current_price, current_price
        
        ma_values.sort()
        
        # 找到支撑位和压力位
        support_level = None
        resistance_level = None
        
        for ma_value in ma_values:
            if ma_value < current_price:
                support_level = ma_value  # 价格下方最近的MA作为支撑位
            elif ma_value > current_price and resistance_level is None:
                resistance_level = ma_value  # 价格上方最近的MA作为压力位
                break
        
        # 如果没有找到支撑位或压力位，使用边界值
        if support_level is None:
            support_level = min(ma_values) if ma_values else current_price
        if resistance_level is None:
            resistance_level = max(ma_values) if ma_values else current_price
        
        return support_level, resistance_level
    
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
    
    def get_ma_support_resistance_levels_new(self, row: pd.Series) -> Tuple[float, float]:
        """
        重新设计的MA支撑压力位获取方法
        压力位：价格上方最近的MA
        支撑位：价格下方最近的MA
        
        Args:
            row: 当前数据行
            
        Returns:
            Tuple[float, float]: (支撑位, 压力位)
            注意：当MA数据不足时，返回None表示无效
        """
        current_price = row['close']
        ma_values = []
        
        for period in self.ma_periods:
            ma_value = row[f'ma_{period}']
            if not pd.isna(ma_value):
                ma_values.append(ma_value)
        
        # 如果没有有效的MA数据，返回None（表示无法使用MA策略）
        if len(ma_values) == 0:
            return None, None
        
        # 找到支撑位和压力位
        support_candidates = [ma for ma in ma_values if ma < current_price]  # 价格下方的MA
        resistance_candidates = [ma for ma in ma_values if ma > current_price]  # 价格上方的MA
        
        # 支撑位：价格下方最近的MA（最高的）
        support_level = max(support_candidates) if support_candidates else min(ma_values)
        
        # 压力位：价格上方最近的MA（最低的）
        resistance_level = min(resistance_candidates) if resistance_candidates else max(ma_values)
        
        return support_level, resistance_level
    
    def check_ma_breakthrough_new(self, current_high: float, current_low: float, 
                                 prev_high: float, prev_low: float,
                                 support_level: float, resistance_level: float) -> Tuple[bool, bool]:
        """
        检查MA突破情况（新逻辑）
        
        Args:
            current_high: 当前最高价
            current_low: 当前最低价
            prev_high: 前一期最高价
            prev_low: 前一期最低价
            support_level: 支撑位
            resistance_level: 压力位
            
        Returns:
            Tuple[bool, bool]: (突破支撑位向下, 突破压力位向上)
        """
        # 突破支撑位向下：前一期最低价在支撑位上方，当前最低价穿过支撑位
        break_support_down = prev_low > support_level and current_low <= support_level
        
        # 突破压力位向上：前一期最高价在压力位下方，当前最高价穿过压力位
        break_resistance_up = prev_high < resistance_level and current_high >= resistance_level
        
        return break_support_down, break_resistance_up
    
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
                'support_add_ratio': self.medium_trend_support_buy_ratio,  # 支撑位加仓比例
                'resistance_reduce_ratio': self.medium_trend_resistance_sell_ratio,  # 压力位减仓比例
                'buy_threshold': self.medium_trend_buy_threshold,
                'sell_threshold': self.medium_trend_sell_threshold,
                'stop_profit': self.medium_trend_stop_profit,
                'stop_loss': self.medium_trend_stop_loss
            }
        else:  # strong
            return {
                'base_buy_ratio': self.strong_trend_base_ratio,
                'support_add_ratio': self.strong_trend_support_buy_ratio,  # 支撑位加仓比例（更小）
                'resistance_reduce_ratio': self.strong_trend_resistance_sell_ratio,  # 压力位减仓比例（更大）
                'buy_threshold': self.strong_trend_buy_threshold,
                'sell_threshold': self.strong_trend_sell_threshold,
                'stop_profit': self.strong_trend_stop_profit,
                'stop_loss': self.strong_trend_stop_loss
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
        for i, (timestamp, row) in enumerate(tqdm(data.iterrows(), 
                                                 total=len(data), 
                                                 desc="运行策略")):
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
                else:
                    print(f"{timestamp}: 周跌幅加仓被阻止 - 当天已加仓")
            
            # 只在震荡阶段执行交易策略
            if self.market_phase == 'oscillation':
                phase_stats['trading_points'] += 1
                
                # 检查当天是否已卖出（如果已卖出则不再执行任何操作）
                if not self.can_trade_today():
                    # 当天已卖出，跳过所有交易逻辑
                    pass
                # 只有在ADX >= 20时才执行交易（ADX < 20时不操作）
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
                                # 检查是否同时突破压力位（信号冲突检查）
                                support_level, resistance_level = self.get_ma_support_resistance_levels_new(row)
                                
                                # 如果MA数据不足，直接执行买入
                                if support_level is None or resistance_level is None:
                                    self.execute_buy(current_price, ratios['base_buy_ratio'], timestamp, f"合成信号买入({adx_level})")
                                elif prev_row is not None:
                                    break_support_down, break_resistance_up = self.check_ma_breakthrough_new(
                                        current_high, current_low,
                                        prev_row['high'], prev_row['low'],
                                        support_level, resistance_level
                                    )
                                    
                                    if break_resistance_up:
                                        print(f"{timestamp}: 技术指标买入信号被阻止 - 同时突破压力位向上（信号冲突）")
                                    else:
                                        # 合成信号买入
                                        self.execute_buy(current_price, ratios['base_buy_ratio'], timestamp, f"合成信号买入({adx_level})")
                                else:
                                    # 没有前一行数据时直接买入
                                    self.execute_buy(current_price, ratios['base_buy_ratio'], timestamp, f"合成信号买入({adx_level})")
                            else:
                                print(f"{timestamp}: 技术指标买入信号被阻止 - 当天已买入或加仓")
                        
                        elif tech_signal <= ratios['sell_threshold'] and self.btc_amount > 0:
                            # 检查是否同时跌破支撑位（信号冲突检查）
                            support_level, resistance_level = self.get_ma_support_resistance_levels_new(row)
                            
                            # 如果MA数据不足，直接执行卖出
                            if support_level is None or resistance_level is None:
                                self.execute_sell(current_price, 1.0, timestamp, f"合成信号卖出({adx_level})")
                            elif prev_row is not None:
                                break_support_down, break_resistance_up = self.check_ma_breakthrough_new(
                                    current_high, current_low,
                                    prev_row['high'], prev_row['low'],
                                    support_level, resistance_level
                                )
                                
                                if break_support_down:
                                    print(f"{timestamp}: 技术指标卖出信号被阻止 - 同时跌破支撑位向下（信号冲突）")
                                else:
                                    # 合成信号卖出所有持仓
                                    self.execute_sell(current_price, 1.0, timestamp, f"合成信号卖出({adx_level})")
                            else:
                                # 没有前一行数据时直接卖出
                                self.execute_sell(current_price, 1.0, timestamp, f"合成信号卖出({adx_level})")
                        
                        # 3. MA支撑压力位逻辑（只有在有持仓且有前一期数据且MA数据有效时才执行）
                        if self.btc_amount > 0 and prev_row is not None:
                            # 获取支撑压力位
                            support_level, resistance_level = self.get_ma_support_resistance_levels_new(row)
                            prev_support_level, prev_resistance_level = self.get_ma_support_resistance_levels_new(prev_row)
                            
                            # 只有当前和前一期的MA数据都有效时才执行MA策略
                            if (support_level is not None and resistance_level is not None and
                                prev_support_level is not None and prev_resistance_level is not None):
                                
                                # 检查MA突破
                                break_support_down, break_resistance_up = self.check_ma_breakthrough_new(
                                    current_high, current_low,
                                    prev_row['high'], prev_row['low'],
                                    support_level, resistance_level
                                )
                                
                                if break_support_down:
                                    # 检查是否可以加仓（当天未加仓）
                                    if self.can_add_position_today():
                                        # 突破支撑位向下，下一期开盘价加仓
                                        pending_trades.append({
                                            'action': 'buy',
                                            'ratio': ratios['support_add_ratio'],
                                            'reason': f"突破支撑位加仓({adx_level})"
                                        })
                                    else:
                                        print(f"{timestamp}: 支撑位加仓被阻止 - 当天已加仓")
                                
                                elif break_resistance_up:
                                    # 突破压力位向上，下一期开盘价减仓
                                    current_ratio = self.get_btc_ratio()
                                    reduce_ratio = min(ratios['resistance_reduce_ratio'], current_ratio)
                                    if reduce_ratio > 0:
                                        pending_trades.append({
                                            'action': 'sell',
                                            'ratio': reduce_ratio,
                                            'reason': f"突破压力位减仓({adx_level})"
                                        })
                            else:
                                # MA数据不足时，跳过MA支撑压力位策略
                                pass
                    
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
            support_level, resistance_level = self.get_ma_support_resistance_levels_new(row)
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
                'support_level': support_level if support_level is not None else np.nan,
                'resistance_level': resistance_level if resistance_level is not None else np.nan,
                'weekly_drop_triggered': weekly_drop_triggered,
                'pending_trades_count': len(pending_trades),
                'today_bought': self.today_bought,
                'today_added_position': self.today_added_position,
                'today_sold': self.today_sold,
                'can_buy_today': self.can_buy_today(),
                'can_add_position_today': self.can_add_position_today(),
                'can_trade_today': self.can_trade_today()
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
        
        # 弱趋势参数
        weak_trend_buy_threshold=1.3,
        weak_trend_sell_threshold=0.7,
        weak_trend_base_ratio=0.15,
        weak_trend_stop_profit=0.08,
        weak_trend_stop_loss=0.05,
        
        # 中等趋势参数
        medium_trend_buy_threshold=1.5,
        medium_trend_sell_threshold=0.5,
        medium_trend_base_ratio=0.25,
        medium_trend_stop_profit=0.08,
        medium_trend_stop_loss=0.05,
        
        # 强趋势参数
        strong_trend_buy_threshold=1.5,
        strong_trend_sell_threshold=0.5,
        strong_trend_base_ratio=0.15,
        strong_trend_stop_profit=0.04,
        strong_trend_stop_loss=0.03,
        
        # MA参数
        ma_periods=[20, 30, 60],
        medium_trend_support_buy_ratio=0.15,
        medium_trend_resistance_sell_ratio=0.20,
        strong_trend_support_buy_ratio=0.10,
        strong_trend_resistance_sell_ratio=0.25
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
