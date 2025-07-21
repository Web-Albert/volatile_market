import pandas as pd
import numpy as np
import pickle
import talib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SignalGenerator:
    """
    综合信号生成器，包含三个模块：
    1. 机器学习模型信号生成
    2. 技术指标信号生成（MACD、BOLL、RSI）
    3. 信号融合
    """
    
    def __init__(self, 
                 ml_model_path: str = None,
                 ml_weight: float = 0.6,
                 tech_weight: float = 0.4,
                 macd_weight: float = 1/3,
                 boll_weight: float = 1/3,
                 rsi_weight: float = 1/3):
        """
        初始化信号生成器
        
        Args:
            ml_model_path: 机器学习模型文件路径
            ml_weight: 机器学习信号权重
            tech_weight: 技术指标信号权重
            macd_weight: MACD信号权重
            boll_weight: BOLL信号权重
            rsi_weight: RSI信号权重
        """
        self.ml_model_path = ml_model_path
        self.ml_model = None
        
        # 权重设置
        self.ml_weight = ml_weight
        self.tech_weight = tech_weight
        self.macd_weight = macd_weight
        self.boll_weight = boll_weight
        self.rsi_weight = rsi_weight
        
        # 技术指标参数
        self.macd_params = {'fast': 12, 'slow': 26, 'signal': 9}
        self.boll_params = {'period': 20, 'std': 2}
        self.rsi_params = {'period': 14, 'buy_threshold': 30, 'sell_threshold': 70}
        
        # 加载机器学习模型
        if ml_model_path:
            self.load_ml_model()
    
    def load_ml_model(self):
        """加载机器学习模型"""
        try:
            with open(self.ml_model_path, 'rb') as f:
                self.ml_model = pickle.load(f)
            print(f"成功加载模型: {self.ml_model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.ml_model = None
    
    def generate_ml_signal(self, features: pd.DataFrame) -> pd.Series:
        """
        生成机器学习信号
        
        Args:
            features: 特征数据
            
        Returns:
            pd.Series: 机器学习信号 (0, 1, 2)
        """
        if self.ml_model is None:
            # 如果没有模型，返回中性信号
            return pd.Series([1] * len(features), index=features.index)
        
        try:
            # 预测
            predictions = self.ml_model.predict(features)
            
            # 转换为Series
            ml_signal = pd.Series(predictions, index=features.index)
            
            return ml_signal
        except Exception as e:
            print(f"机器学习信号生成失败: {e}")
            return pd.Series([1] * len(features), index=features.index)
    
    def generate_macd_signal(self, kline_data: pd.DataFrame) -> pd.Series:
        """
        生成MACD信号
        
        Args:
            kline_data: K线数据，包含close列
            
        Returns:
            pd.Series: MACD信号 (0, 1, 2)
        """
        close = kline_data['close'].values
        
        # 计算MACD
        macd, macd_signal, macd_hist = talib.MACD(close,
                                                  fastperiod=self.macd_params['fast'],
                                                  slowperiod=self.macd_params['slow'],
                                                  signalperiod=self.macd_params['signal'])
        
        # 初始化信号
        signals = pd.Series([1] * len(kline_data), index=kline_data.index)
        
        # 计算信号条件
        for i in range(1, len(signals)):
            curr_macd, curr_signal = macd[i], macd_signal[i]
            prev_macd, prev_signal = macd[i-1], macd_signal[i-1]
            curr_hist, prev_hist = macd_hist[i], macd_hist[i-1]
            
            # 跳过NaN值
            if pd.isna(curr_macd) or pd.isna(curr_signal) or pd.isna(curr_hist):
                continue
            
            # DIF和DEA都为正，且MACD柱状图转正 -> 看多信号
            if (curr_macd > 0 and curr_signal > 0 and 
                prev_hist <= 0 and curr_hist > 0):
                signals.iloc[i] = 2
            
            # DIF和DEA都为负，且MACD柱状图转负 -> 看空信号
            elif (curr_macd < 0 and curr_signal < 0 and 
                  prev_hist >= 0 and curr_hist < 0):
                signals.iloc[i] = 0
            
            # 其他情况保持不动
            else:
                signals.iloc[i] = 1
        
        return signals
    
    def generate_boll_signal(self, kline_data: pd.DataFrame) -> pd.Series:
        """
        生成BOLL信号
        
        Args:
            kline_data: K线数据，包含close列
            
        Returns:
            pd.Series: BOLL信号 (0, 1, 2)
        """
        close = kline_data['close'].values
        
        # 计算布林带
        upper, middle, lower = talib.BBANDS(close,
                                           timeperiod=self.boll_params['period'],
                                           nbdevup=self.boll_params['std'],
                                           nbdevdn=self.boll_params['std'])
        
        # 初始化信号
        signals = pd.Series([1] * len(kline_data), index=kline_data.index)
        
        # 计算信号条件
        for i in range(1, len(signals)):
            curr_close, prev_close = close[i], close[i-1]
            curr_upper, curr_middle, curr_lower = upper[i], middle[i], lower[i]
            prev_upper, prev_middle, prev_lower = upper[i-1], middle[i-1], lower[i-1]
            
            # 跳过NaN值
            if pd.isna(curr_upper) or pd.isna(curr_middle) or pd.isna(curr_lower):
                continue
            
            # 上穿上轨或上穿中线 -> 看多信号
            if ((prev_close <= prev_upper and curr_close > curr_upper) or
                (prev_close <= prev_middle and curr_close > curr_middle)):
                signals.iloc[i] = 2
            
            # 下穿下轨或下穿中线 -> 看空信号
            elif ((prev_close >= prev_lower and curr_close < curr_lower) or
                  (prev_close >= prev_middle and curr_close < curr_middle)):
                signals.iloc[i] = 0
            
            # 在区间内震荡 -> 保持不动
            else:
                signals.iloc[i] = 1
        
        return signals
    
    def generate_rsi_signal(self, kline_data: pd.DataFrame) -> pd.Series:
        """
        生成RSI信号
        
        Args:
            kline_data: K线数据，包含close列
            
        Returns:
            pd.Series: RSI信号 (0, 1, 2)
        """
        close = kline_data['close'].values
        
        # 计算RSI
        rsi = talib.RSI(close, timeperiod=self.rsi_params['period'])
        
        # 初始化信号
        signals = pd.Series([1] * len(kline_data), index=kline_data.index)
        
        # 计算信号条件
        for i in range(len(signals)):
            curr_rsi = rsi[i]
            
            # 跳过NaN值
            if pd.isna(curr_rsi):
                continue
            
            # RSI小于买入阈值 -> 看多信号
            if curr_rsi < self.rsi_params['buy_threshold']:
                signals.iloc[i] = 2
            
            # RSI大于卖出阈值 -> 看空信号
            elif curr_rsi > self.rsi_params['sell_threshold']:
                signals.iloc[i] = 0
            
            # 区间震荡 -> 保持不动
            else:
                signals.iloc[i] = 1
        
        return signals
    
    def generate_technical_signal(self, kline_data: pd.DataFrame) -> tuple:
        """
        生成技术指标综合信号
        
        Args:
            kline_data: K线数据
            
        Returns:
            tuple: (连续值信号, 离散值信号)
                - 连续值信号: pd.Series，用于阈值判断
                - 离散值信号: pd.Series，离散化后的信号 (0, 1, 2)
        """
        # 生成各个技术指标信号
        macd_signal = self.generate_macd_signal(kline_data)
        boll_signal = self.generate_boll_signal(kline_data)
        rsi_signal = self.generate_rsi_signal(kline_data)
        
        # 加权平均得到连续值信号
        tech_signal_continuous = (macd_signal * self.macd_weight + 
                                 boll_signal * self.boll_weight + 
                                 rsi_signal * self.rsi_weight)
        
        # 将连续值转换为离散信号
        tech_signal_discrete = pd.Series([1] * len(tech_signal_continuous), 
                                        index=tech_signal_continuous.index)
        
        for i in range(len(tech_signal_continuous)):
            if tech_signal_continuous.iloc[i] >= 1.5:
                tech_signal_discrete.iloc[i] = 2
            elif tech_signal_continuous.iloc[i] <= 0.5:
                tech_signal_discrete.iloc[i] = 0
            else:
                tech_signal_discrete.iloc[i] = 1
        
        return tech_signal_continuous, tech_signal_discrete
    
    def generate_final_signal(self, 
                            kline_data: pd.DataFrame, 
                            features: pd.DataFrame = None) -> Dict:
        """
        生成最终综合信号
        
        Args:
            kline_data: K线数据
            features: 机器学习特征数据（可选）
            
        Returns:
            Dict: 包含各种信号的字典
        """
        # 生成机器学习信号
        if features is not None and self.ml_model is not None:
            ml_signal = self.generate_ml_signal(features)
        else:
            ml_signal = pd.Series([1] * len(kline_data), index=kline_data.index)
        
        # 生成技术指标信号
        tech_signal_continuous, tech_signal_discrete = self.generate_technical_signal(kline_data)
        
        # 生成最终信号（加权平均）
        final_signal_continuous = (ml_signal * self.ml_weight + 
                                 tech_signal_continuous * self.tech_weight)
        
        # 将连续值转换为离散信号
        final_signal = pd.Series([1] * len(final_signal_continuous), 
                                index=final_signal_continuous.index)
        
        for i in range(len(final_signal_continuous)):
            if final_signal_continuous.iloc[i] >= 1.5:
                final_signal.iloc[i] = 2
            elif final_signal_continuous.iloc[i] <= 0.5:
                final_signal.iloc[i] = 0
            else:
                final_signal.iloc[i] = 1
        
        # 返回详细信号信息
        return {
            'final_signal': final_signal,
            'ml_signal': ml_signal,
            'tech_signal_continuous': tech_signal_continuous,
            'tech_signal_discrete': tech_signal_discrete,
            'macd_signal': self.generate_macd_signal(kline_data),
            'boll_signal': self.generate_boll_signal(kline_data),
            'rsi_signal': self.generate_rsi_signal(kline_data),
            'final_signal_continuous': final_signal_continuous
        }
    
    def update_weights(self, 
                      ml_weight: float = None,
                      tech_weight: float = None,
                      macd_weight: float = None,
                      boll_weight: float = None,
                      rsi_weight: float = None):
        """
        更新信号权重
        
        Args:
            ml_weight: 机器学习信号权重
            tech_weight: 技术指标信号权重
            macd_weight: MACD信号权重
            boll_weight: BOLL信号权重
            rsi_weight: RSI信号权重
        """
        if ml_weight is not None:
            self.ml_weight = ml_weight
        if tech_weight is not None:
            self.tech_weight = tech_weight
        if macd_weight is not None:
            self.macd_weight = macd_weight
        if boll_weight is not None:
            self.boll_weight = boll_weight
        if rsi_weight is not None:
            self.rsi_weight = rsi_weight
        
        print("信号权重更新完成")
    
    def update_technical_params(self, 
                               macd_params: Dict = None,
                               boll_params: Dict = None,
                               rsi_params: Dict = None):
        """
        更新技术指标参数
        
        Args:
            macd_params: MACD参数字典
            boll_params: BOLL参数字典
            rsi_params: RSI参数字典
        """
        if macd_params:
            self.macd_params.update(macd_params)
        if boll_params:
            self.boll_params.update(boll_params)
        if rsi_params:
            self.rsi_params.update(rsi_params)
        
        print("技术指标参数更新完成")

# 使用示例
def main():
    # 创建信号生成器
    signal_gen = SignalGenerator(
        ml_model_path='path/to/your/model.pkl',  # 替换为实际模型路径
        ml_weight=0.6,
        tech_weight=0.4,
        macd_weight=0.4,
        boll_weight=0.3,
        rsi_weight=0.3
    )
    
    # 示例K线数据
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    kline_data = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(100).cumsum() + 100,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'volume': np.random.randint(1000, 10000, 100)
    })
    kline_data.set_index('timestamp', inplace=True)
    
    # 生成信号
    signals = signal_gen.generate_final_signal(kline_data)
    
    # 显示结果
    print("信号生成完成!")
    print(f"最终信号分布: {signals['final_signal'].value_counts().to_dict()}")
    print(f"MACD信号分布: {signals['macd_signal'].value_counts().to_dict()}")
    print(f"BOLL信号分布: {signals['boll_signal'].value_counts().to_dict()}")
    print(f"RSI信号分布: {signals['rsi_signal'].value_counts().to_dict()}")
    
    return signal_gen, signals

if __name__ == "__main__":
    main() 