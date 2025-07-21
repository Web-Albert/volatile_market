# BTC 震荡市场交易策略

## �� 概述

本策略专门针对震荡市场设计，**使用日线结合15分钟K线数据**，结合新闻信号判断市场阶段，**只在震荡阶段执行交易**。策略核心特点：

- 🔄 **阶段判断**：根据新闻信号自动识别看多、看空、震荡三个市场阶段
- 📊 **ADX指标**：使用ADX指标判断趋势强度，ADX < 20时不操作，≥ 20时分为中、强两个等级
- 🎯 **合成信号**：结合MACD、RSI、BOLL技术指标生成交易信号
- 📈 **MA穿越**：使用多个MA（30、60、120）判断支撑压力位，基于K线穿越MA触发交易
- ⚡ **高频交易**：基于15分钟K线数据进行高频交易决策
- 📉 **周跌幅加仓**：当价格相比一周前下跌15%以上时自动加仓

## 🏗️ 文件结构

```
volatile_market/
├── volatile_market.py              # 震荡市场策略核心代码
├── volatile_strategy_config.py     # 策略参数配置文件
├── run_volatile_market.py          # 策略运行脚本
├── signal_generator.py             # 技术指标信号生成器
├── trend_market.py                 # 趋势市场策略（参考）
└── README_volatile_market.md       # 本说明文档
```

## 🚀 快速开始

### 1. 环境准备

确保安装了以下依赖包：

```bash
pip install pandas numpy talib tqdm
```

### 2. 配置参数

编辑 `volatile_strategy_config.py` 文件，修改以下关键参数：

- **数据路径**：修改 `DAILY_KLINES_PATH`、`MIN15_KLINES_PATH` 和 `NEWS_SIGNALS_PATH`
- **策略参数**：根据需要调整各种阈值和比例

### 3. 运行策略

```bash
python run_volatile_market.py
```

## ⚙️ 策略参数说明

### 基础参数
- `MIN_BTC_RATIO`: 最低BTC仓位比例 (默认30%)
- `MAX_BTC_RATIO`: 最大BTC仓位比例 (默认80%)
- `INITIAL_CASH`: 初始资金 (默认100,000 USDT)

### 新闻信号阈值
- `NEWS_HIGH_THRESHOLD`: 新闻信号高阈值，超过此值进入看多阶段 (默认1.4)
- `NEWS_LOW_THRESHOLD`: 新闻信号低阈值，低于此值进入看空阶段 (默认0.55)

### ADX参数
- `ADX_PERIOD`: ADX计算周期 (默认14)
- `ADX_LOW_THRESHOLD`: ADX低阈值，低于此值不操作 (默认20)
- `ADX_HIGH_THRESHOLD`: ADX高阈值，高于此值为强趋势 (默认40)

### 交易阈值（根据ADX等级）

#### 中等趋势 (20 ≤ ADX ≤ 40)
- `MEDIUM_TREND_BUY_THRESHOLD`: 买入阈值 (默认1.5)
- `MEDIUM_TREND_SELL_THRESHOLD`: 卖出阈值 (默认0.5)
- `MEDIUM_TREND_BASE_RATIO`: 基础仓位比例 (默认25%)
- `MEDIUM_TREND_STOP_PROFIT`: 止盈比例 (默认6%)
- `MEDIUM_TREND_STOP_LOSS`: 止损比例 (默认3%)

#### 强趋势 (ADX > 40)
- `STRONG_TREND_BUY_THRESHOLD`: 买入阈值 (默认1.6)
- `STRONG_TREND_SELL_THRESHOLD`: 卖出阈值 (默认0.4)
- `STRONG_TREND_BASE_RATIO`: 基础仓位比例 (默认15%)
- `STRONG_TREND_STOP_PROFIT`: 止盈比例 (默认4%)
- `STRONG_TREND_STOP_LOSS`: 止损比例 (默认3%)

### MA支撑压力位参数
- `MA_PERIODS`: MA周期列表 (默认[30, 60, 120])
- `SUPPORT_BUY_RATIO`: 触及支撑位买入比例 (默认20%)
- `RESISTANCE_SELL_RATIO`: 触及压力位卖出比例 (默认20%)

### 周跌幅加仓参数
- `WEEKLY_DROP_THRESHOLD`: 周跌幅阈值 (默认15%)
- `WEEKLY_DROP_BUY_RATIO`: 周跌幅加仓比例 (默认10%)

#### MA支撑压力位逻辑说明
- **支撑位判断**：前一K线在MA上方，当前K线最低价穿过MA线
  - 触发条件：`prev_low > prev_ma_value and current_low <= ma_value`
  - 触发操作：加仓买入
- **压力位判断**：前一K线在MA下方，当前K线最高价穿过MA线
  - 触发条件：`prev_high < prev_ma_value and current_high >= ma_value`
  - 触发操作：减仓卖出

## 📊 策略逻辑

### 1. 数据准备
- 使用日线K线数据计算技术指标
- 对于每个15分钟K线，更新当天的收盘价、最高价、最低价
- 重新计算技术指标和ADX值

### 2. 市场阶段判断
```python
if news_signal >= 1.4:
    market_phase = 'bullish'    # 看多阶段
elif news_signal <= 0.55:
    market_phase = 'bearish'    # 看空阶段
else:
    market_phase = 'oscillation'  # 震荡阶段
```

### 3. ADX等级判断
```python
if adx < 20:
    adx_level = 'weak'      # 弱趋势（不操作）
elif adx <= 40:
    adx_level = 'medium'    # 中等趋势
else:
    adx_level = 'strong'    # 强趋势
```

### 4. 交易决策（仅在震荡阶段且ADX ≥ 20）

#### 合成信号交易
- **买入条件**：技术指标合成信号 ≥ 买入阈值
- **卖出条件**：技术指标合成信号 ≤ 卖出阈值（卖出所有持仓）

#### MA穿越交易（仅在有持仓时）
- **支撑位加仓**：K线从上向下穿过MA线时加仓
- **压力位减仓**：K线从下向上穿过MA线时减仓

#### 周跌幅加仓
- **加仓条件**：当前价格相比一周前下跌 ≥ 15%
- **适用范围**：任何ADX阶段和市场阶段

#### 止盈止损
- 根据ADX等级设置不同的止盈止损比例
- 强趋势时止盈止损更紧，中等趋势时更宽松

## 📈 输出结果

策略运行完成后会生成以下文件：

### 统计报告
- `volatile_strategy_statistics_YYYYMMDD_HHMMSS.csv`: 策略统计结果
- `volatile_backtest_report_YYYYMMDD_HHMMSS.md`: 回测报告摘要

### 交易记录
- `volatile_trading_records_YYYYMMDD_HHMMSS.csv`: 详细交易记录
- `volatile_trading_summary_YYYYMMDD_HHMMSS.csv`: 交易汇总分析
- `volatile_position_changes_YYYYMMDD_HHMMSS.csv`: 仓位变化记录

### 账户状态
- `volatile_records_YYYYMMDD_HHMMSS.csv`: 每15分钟账户状态
- `volatile_equity_curve_YYYYMMDD_HHMMSS.csv`: 收益曲线数据
- `volatile_current_positions_YYYYMMDD_HHMMSS.csv`: 当前持仓信息

### 分析报告
- `volatile_market_analysis_YYYYMMDD_HHMMSS.xlsx`: 市场分析（ADX分布、阶段分布）
- `volatile_strategy_parameters_YYYYMMDD_HHMMSS.csv`: 策略参数配置

## 🔧 参数调优建议

### 1. 新闻信号阈值
- 提高 `NEWS_HIGH_THRESHOLD` 和 `NEWS_LOW_THRESHOLD` 可以增加震荡阶段的比例
- 降低阈值会减少震荡阶段，增加趋势阶段

### 2. ADX阈值
- 调整 `ADX_LOW_THRESHOLD` 可以改变开始交易的条件
- 调整 `ADX_HIGH_THRESHOLD` 可以改变强趋势的判断标准

### 3. 交易阈值
- 提高买入阈值、降低卖出阈值可以减少交易频率
- 降低买入阈值、提高卖出阈值可以增加交易频率

### 4. 仓位比例
- 根据风险承受能力调整各种仓位比例
- 强趋势时建议降低仓位，中等趋势时可以适当增加仓位

### 5. 止盈止损
- 根据市场波动性调整止盈止损比例
- 高波动市场建议设置更宽松的止损

## ⚠️ 注意事项

1. **数据质量**：确保日线K线数据、15分钟K线数据和新闻信号数据的质量和完整性
2. **参数调优**：建议在历史数据上进行充分的回测和参数优化
3. **风险控制**：合理设置最大仓位和止损比例，控制风险
4. **实盘验证**：在实盘使用前建议先进行小资金测试
5. **监控运行**：定期检查策略运行状态和交易记录

## 🤝 与趋势策略的结合

本策略可以与 `trend_market.py` 中的趋势策略结合使用：

- **趋势阶段**：使用趋势策略
- **震荡阶段**：使用震荡策略
- **阶段切换**：根据新闻信号自动切换策略

详见 `volatile_market.py` 中的 `HybridTradingStrategy` 类。

## 📞 技术支持

如有问题或建议，请查看代码注释或联系开发团队。 