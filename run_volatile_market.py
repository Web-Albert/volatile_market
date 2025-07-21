#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BTC震荡市场交易策略运行文件
使用配置文件中的参数运行策略
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# 导入策略类和配置
from volatile_market import VolatileMarketStrategy
from volatile_strategy_config import get_volatile_strategy_params, get_volatile_data_paths, print_volatile_config

def load_data():
    """加载数据"""
    print("正在加载数据...")
    
    paths = get_volatile_data_paths()
    
    # 加载日线数据
    print("加载日线K线数据...")
    daily_klines = pd.read_csv(paths['daily_klines'], index_col='open_time', parse_dates=['open_time'],
                               usecols=['open_time','open','high','low','close','volume'])
    
    # 加载15分钟数据
    print("加载15分钟K线数据...")
    with open(paths['min15_klines'], 'rb') as f:
        min15_klines = pickle.load(f)
    min15_klines = min15_klines[['time','open','high','low','close','volume']] 
    min15_klines = min15_klines.set_index('time')
    
    # 加载新闻信号数据
    print("加载新闻信号数据...")
    news_signals = pd.read_csv(paths['news_signals'], index_col='time', parse_dates=['time'],
                               usecols=['time','important_news_signal'])
    
    print(f"数据加载完成:")
    print(f"  日线数据: {len(daily_klines)} 条")
    print(f"  15分钟数据: {len(min15_klines)} 条")
    print(f"  新闻信号: {len(news_signals)} 条")
    
    return daily_klines, min15_klines, news_signals

def save_volatile_strategy_results(results, strategy, save_dir="volatile_strategy_results"):
    """
    保存震荡市场策略回测结果到文件
    
    Args:
        results: 策略结果字典
        strategy: 策略实例，用于获取参数配置
        save_dir: 保存目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 生成时间戳用于文件命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存策略统计结果
    stats_data = {
        '指标': ['策略类型', '初始资金', '最终价值', '总收益率', '年化收益率', '最大回撤', '波动率', 
                '夏普比率', '胜率', '盈亏比', '最大连续亏损天数', '总交易次数', 
                '买入次数', '卖出次数', '总手续费', '最终BTC持仓', '最终现金', '最终仓位比例', '震荡阶段占比'],
        '数值': [
            results['strategy_type'],
            f"{results['initial_cash']:,.2f}",
            f"{results['final_value']:,.2f}",
            f"{results['total_return']:.2%}",
            f"{results.get('annualized_return', 0):.2%}",
            f"{results['max_drawdown']:.2%}",
            f"{results.get('volatility', 0):.2%}",
            f"{results.get('sharpe_ratio', 0):.2f}",
            f"{results.get('win_rate', 0):.2%}",
            f"{results.get('profit_loss_ratio', 0):.2f}",
            f"{results.get('max_consecutive_losses', 0)}",
            results['total_trades'],
            results.get('buy_trades', len([t for t in results['trades'] if t['action'] == 'buy'])),
            results.get('sell_trades', len([t for t in results['trades'] if t['action'] == 'sell'])),
            f"{results['total_trading_fees']:,.2f}",
            f"{results['final_btc_amount']:.6f}",
            f"{results['final_cash']:,.2f}",
            f"{results['final_btc_ratio']:.2%}",
            f"{results.get('oscillation_ratio', 0):.2%}"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    stats_file = os.path.join(save_dir, f"volatile_strategy_statistics_{timestamp}.csv")
    stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"📊 策略统计结果已保存到: {stats_file}")
    
    # 2. 保存交易记录
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_file = os.path.join(save_dir, f"volatile_trading_records_{timestamp}.csv")
        trades_df.to_csv(trades_file, index=False, encoding='utf-8-sig')
        print(f"💰 交易记录已保存到: {trades_file}")
        
        # 保存交易汇总分析
        buy_trades = trades_df[trades_df['action'] == 'buy']
        sell_trades = trades_df[trades_df['action'] == 'sell']
        
        trade_summary = {
            '交易类型': ['买入交易', '卖出交易'],
            '交易次数': [len(buy_trades), len(sell_trades)],
            '平均金额': [buy_trades['value'].mean() if len(buy_trades) > 0 else 0,
                        sell_trades['value'].mean() if len(sell_trades) > 0 else 0],
            '总金额': [buy_trades['value'].sum() if len(buy_trades) > 0 else 0,
                      sell_trades['value'].sum() if len(sell_trades) > 0 else 0]
        }
        
        trade_summary_df = pd.DataFrame(trade_summary)
        trade_summary_file = os.path.join(save_dir, f"volatile_trading_summary_{timestamp}.csv")
        trade_summary_df.to_csv(trade_summary_file, index=False, encoding='utf-8-sig')
        print(f"📈 交易汇总已保存到: {trade_summary_file}")
    
    # 3. 保存每15分钟账户状态
    records_df = results['records']
    records_file = os.path.join(save_dir, f"volatile_records_{timestamp}.csv")
    records_df.to_csv(records_file, encoding='utf-8-sig')
    print(f"📅 每15分钟账户状态已保存到: {records_file}")
    
    # 4. 保存仓位变化记录
    position_changes = []
    for trade in results['trades']:
        position_changes.append({
            'timestamp': trade['timestamp'],
            'action': trade['action'],
            'price': trade['price'],
            'amount': trade['amount'],
            'btc_amount_after': trade['btc_amount_after'],
            'cash_after': trade['cash_after'],
            'total_value_after': trade['total_value_after'],
            'btc_ratio_after': trade['btc_amount_after'] * trade['price'] / trade['total_value_after'] if trade['total_value_after'] > 0 else 0,
            'reason': trade['reason']
        })
    
    if position_changes:
        position_df = pd.DataFrame(position_changes)
        position_file = os.path.join(save_dir, f"volatile_position_changes_{timestamp}.csv")
        position_df.to_csv(position_file, index=False, encoding='utf-8-sig')
        print(f"📊 仓位变化记录已保存到: {position_file}")
    
    # 5. 保存收益曲线数据
    equity_curve = records_df[['timestamp', 'total_value', 'btc_value', 'cash', 'btc_ratio']].copy()
    equity_curve['cumulative_return'] = (equity_curve['total_value'] / results['initial_cash'] - 1) * 100
    equity_file = os.path.join(save_dir, f"volatile_equity_curve_{timestamp}.csv")
    equity_curve.to_csv(equity_file, index=False, encoding='utf-8-sig')
    print(f"📈 收益曲线数据已保存到: {equity_file}")
    
    # 6. 保存当前持仓信息
    if results.get('positions'):
        current_positions = []
        for pos in results['positions']:
            current_positions.append({
                'buy_time': pos['buy_time'],
                'buy_price': pos['buy_price'],
                'amount': pos['amount'],
                'buy_value': pos['buy_value'],
                'current_price': pos.get('current_price', pos['buy_price']),
                'current_value': pos.get('current_value', pos['buy_value']),
                'pnl_ratio': pos.get('pnl_ratio', 0),
                'unrealized_pnl': pos.get('current_value', pos['buy_value']) - pos['buy_value']
            })
        
        if current_positions:
            positions_df = pd.DataFrame(current_positions)
            positions_file = os.path.join(save_dir, f"volatile_current_positions_{timestamp}.csv")
            positions_df.to_csv(positions_file, index=False, encoding='utf-8-sig')
            print(f"📊 当前持仓信息已保存到: {positions_file}")
    
    # 7. 保存ADX和市场阶段分析
    adx_analysis = {
        'ADX等级': list(results.get('adx_distribution', {}).keys()),
        '时间点数量': list(results.get('adx_distribution', {}).values()),
        '占比': [f"{v/sum(results.get('adx_distribution', {}).values())*100:.1f}%" 
                for v in results.get('adx_distribution', {}).values()]
    }
    
    phase_analysis = {
        '市场阶段': list(results.get('phase_distribution', {}).keys()),
        '时间点数量': list(results.get('phase_distribution', {}).values()),
        '占比': [f"{v/sum(results.get('phase_distribution', {}).values())*100:.1f}%" 
                for v in results.get('phase_distribution', {}).values()]
    }
    
    adx_df = pd.DataFrame(adx_analysis)
    phase_df = pd.DataFrame(phase_analysis)
    
    analysis_file = os.path.join(save_dir, f"volatile_market_analysis_{timestamp}.csv")
    with pd.ExcelWriter(analysis_file.replace('.csv', '.xlsx')) as writer:
        adx_df.to_excel(writer, sheet_name='ADX分析', index=False)
        phase_df.to_excel(writer, sheet_name='市场阶段分析', index=False)
    print(f"📊 市场分析已保存到: {analysis_file.replace('.csv', '.xlsx')}")
    
    # 8. 保存策略参数配置
    strategy_params = {
        '参数名': [
            '最低BTC仓位比例', '最大BTC仓位比例', '初始资金', '交易手续费',
            '新闻信号高阈值', '新闻信号低阈值',
            'ADX计算周期', 'ADX低阈值', 'ADX高阈值',
            '中等趋势买入阈值', '中等趋势卖出阈值', '中等趋势基础仓位比例',
            '中等趋势止盈比例', '中等趋势止损比例',
            '强趋势买入阈值', '强趋势卖出阈值', '强趋势基础仓位比例',
            '强趋势止盈比例', '强趋势止损比例',
            'MA周期', '中等趋势支撑位加仓比例', '中等趋势压力位减仓比例',
            '强趋势支撑位加仓比例', '强趋势压力位减仓比例',
            '周跌幅阈值', '周跌幅加仓比例'
        ],
        '参数值': [
            f"{strategy.min_btc_ratio:.2%}",
            f"{strategy.max_btc_ratio:.2%}",
            f"{strategy.initial_cash:,.2f}",
            f"{strategy.trading_fee:.4%}",
            f"{strategy.news_high_threshold:.2f}",
            f"{strategy.news_low_threshold:.2f}",
            f"{strategy.adx_period}",
            f"{strategy.adx_low_threshold}",
            f"{strategy.adx_high_threshold}",
            f"{strategy.medium_trend_buy_threshold:.2f}",
            f"{strategy.medium_trend_sell_threshold:.2f}",
            f"{strategy.medium_trend_base_ratio:.2%}",
            f"{strategy.medium_trend_stop_profit:.2%}",
            f"{strategy.medium_trend_stop_loss:.2%}",
            f"{strategy.strong_trend_buy_threshold:.2f}",
            f"{strategy.strong_trend_sell_threshold:.2f}",
            f"{strategy.strong_trend_base_ratio:.2%}",
            f"{strategy.strong_trend_stop_profit:.2%}",
            f"{strategy.strong_trend_stop_loss:.2%}",
            f"{strategy.ma_periods}",
            f"{strategy.medium_trend_support_buy_ratio:.2%}",
            f"{strategy.medium_trend_resistance_sell_ratio:.2%}",
            f"{strategy.strong_trend_support_buy_ratio:.2%}",
            f"{strategy.strong_trend_resistance_sell_ratio:.2%}",
            f"{strategy.weekly_drop_threshold:.2%}",
            f"{strategy.weekly_drop_buy_ratio:.2%}"
        ]
    }
    
    params_df = pd.DataFrame(strategy_params)
    params_file = os.path.join(save_dir, f"volatile_strategy_parameters_{timestamp}.csv")
    params_df.to_csv(params_file, index=False, encoding='utf-8-sig')
    print(f"⚙️ 策略参数已保存到: {params_file}")
    
    # 9. 生成回测报告摘要
    records_data = results['records']
    report_content = f"""
# BTC震荡市场交易策略回测报告

## 📊 策略表现
- **策略类型**: {results['strategy_type']}
- **初始资金**: {results['initial_cash']:,.2f}
- **最终价值**: {results['final_value']:,.2f}
- **总收益率**: {results['total_return']:.2%}
- **年化收益率**: {results.get('annualized_return', 0):.2%}
- **最大回撤**: {results['max_drawdown']:.2%}
- **波动率**: {results.get('volatility', 0):.2%}
- **夏普比率**: {results.get('sharpe_ratio', 0):.2f}

## 📈 交易统计
- **总交易次数**: {results['total_trades']}
- **买入次数**: {len([t for t in results['trades'] if t['action'] == 'buy'])}
- **卖出次数**: {len([t for t in results['trades'] if t['action'] == 'sell'])}
- **总手续费**: {results['total_trading_fees']:,.2f}
- **胜率**: {results.get('win_rate', 0):.2%}
- **盈亏比**: {results.get('profit_loss_ratio', 0):.2f}
- **最大连续亏损天数**: {results.get('max_consecutive_losses', 0)}

## 🔄 市场阶段分析
- **震荡阶段占比**: {results.get('oscillation_ratio', 0):.2%}
- **市场阶段分布**: {results.get('phase_distribution', {})}
- **ADX等级分布**: {results.get('adx_distribution', {})}

## 💰 最终持仓
- **BTC持仓**: {results['final_btc_amount']:.6f}
- **现金余额**: {results['final_cash']:,.2f}
- **仓位比例**: {results['final_btc_ratio']:.2%}
- **当前持仓笔数**: {len(results.get('positions', []))}

## ⚙️ 策略参数配置
- **最低BTC仓位比例**: {strategy.min_btc_ratio:.2%}
- **最大BTC仓位比例**: {strategy.max_btc_ratio:.2%}
- **新闻信号高阈值**: {strategy.news_high_threshold:.2f}
- **新闻信号低阈值**: {strategy.news_low_threshold:.2f}
- **ADX计算周期**: {strategy.adx_period}
- **ADX低阈值**: {strategy.adx_low_threshold}
- **ADX高阈值**: {strategy.adx_high_threshold}
- **中等趋势买入阈值**: {strategy.medium_trend_buy_threshold:.2f}
- **强趋势买入阈值**: {strategy.strong_trend_buy_threshold:.2f}
- **MA周期**: {strategy.ma_periods}
- **中等趋势支撑位加仓比例**: {strategy.medium_trend_support_buy_ratio:.2%}
- **中等趋势压力位减仓比例**: {strategy.medium_trend_resistance_sell_ratio:.2%}
- **强趋势支撑位加仓比例**: {strategy.strong_trend_support_buy_ratio:.2%}
- **强趋势压力位减仓比例**: {strategy.strong_trend_resistance_sell_ratio:.2%}

## 📅 回测时间
- **开始时间**: {records_data['timestamp'].min()}
- **结束时间**: {records_data['timestamp'].max()}
- **回测时间点数**: {len(records_data)}

## 📁 保存文件
- 策略统计: volatile_strategy_statistics_{timestamp}.csv
- 交易记录: volatile_trading_records_{timestamp}.csv
- 交易汇总: volatile_trading_summary_{timestamp}.csv
- 账户状态: volatile_records_{timestamp}.csv
- 仓位变化: volatile_position_changes_{timestamp}.csv
- 收益曲线: volatile_equity_curve_{timestamp}.csv
- 当前持仓: volatile_current_positions_{timestamp}.csv
- 市场分析: volatile_market_analysis_{timestamp}.xlsx
- 策略参数: volatile_strategy_parameters_{timestamp}.csv
"""
    
    report_file = os.path.join(save_dir, f"volatile_backtest_report_{timestamp}.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"📝 回测报告已保存到: {report_file}")
    
    print(f"\n✅ 所有结果已保存到目录: {save_dir}")
    return timestamp

def run_volatile_strategy():
    """运行震荡市场策略"""
    print("="*60)
    print("BTC 震荡市场交易策略")
    print("="*60)
    
    # 打印配置信息
    print_volatile_config()
    
    # 加载数据
    daily_klines, min15_klines, news_signals = load_data()
    
    # 获取策略参数
    params = get_volatile_strategy_params()
    
    # 创建策略实例
    print("\n正在初始化震荡市场策略...")
    strategy = VolatileMarketStrategy(**params)
    
    # 准备数据
    print("\n正在准备数据...")
    data = strategy.prepare_data(daily_klines, min15_klines, news_signals)
    print(f"数据准备完成，共 {len(data)} 个时间点")
    print(f"数据列: {list(data.columns)}")
    
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
    print(f"年化收益率: {results.get('annualized_return', 0):.2%}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"夏普比率: {results.get('sharpe_ratio', 0):.2f}")
    print(f"胜率: {results.get('win_rate', 0):.2%}")
    print(f"盈亏比: {results.get('profit_loss_ratio', 0):.2f}")
    print(f"波动率: {results.get('volatility', 0):.2%}")
    print(f"总交易次数: {results['total_trades']}")
    print(f"总手续费: {results['total_trading_fees']:,.2f}")
    print(f"最终BTC持仓: {results['final_btc_amount']:.6f}")
    print(f"最终仓位比例: {results['final_btc_ratio']:.2%}")
    print(f"震荡阶段占比: {results.get('oscillation_ratio', 0):.2%}")
    print(f"ADX等级分布: {results.get('adx_distribution', {})}")
    print(f"市场阶段分布: {results.get('phase_distribution', {})}")
    print("="*60)
    
    # 保存结果
    print("\n正在保存回测结果...")
    save_timestamp = save_volatile_strategy_results(results, strategy, "volatile_strategy_results")
    print(f"\n回测结果已保存，时间戳: {save_timestamp}")
    
    return strategy, results

if __name__ == "__main__":
    try:
        strategy, results = run_volatile_strategy()
        print("\n震荡市场策略运行成功！")
    except Exception as e:
        print(f"\n策略运行出错: {e}")
        import traceback
        traceback.print_exc()
