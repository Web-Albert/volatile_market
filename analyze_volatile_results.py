#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
震荡市场策略回测结果分析工具
用于分析和可视化保存的震荡市场策略回测结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_latest_volatile_results(results_dir: str = "volatile_strategy_results"):
    """
    加载最新的震荡市场策略回测结果文件
    
    Args:
        results_dir: 结果文件目录
        
    Returns:
        dict: 包含各种数据的字典
    """
    if not os.path.exists(results_dir):
        print(f"❌ 结果目录不存在: {results_dir}")
        return None
    
    # 查找最新的时间戳
    files = glob.glob(os.path.join(results_dir, "volatile_*.csv"))
    if not files:
        print("❌ 未找到震荡市场策略结果文件")
        return None
    
    # 从文件名中提取时间戳
    timestamps = []
    for file in files:
        filename = os.path.basename(file)
        if "_" in filename:
            parts = filename.split("_")
            if len(parts) >= 3:
                # 提取完整的时间戳：日期_时间
                date_part = parts[-2]  # 20250714
                time_part = parts[-1].replace(".csv", "")  # 223225
                timestamp = f"{date_part}_{time_part}"
                timestamps.append(timestamp)
    
    if not timestamps:
        print("❌ 无法识别时间戳")
        return None
    
    latest_timestamp = max(timestamps)
    print(f"📅 加载最新震荡市场策略结果: {latest_timestamp}")
    
    # 加载各种数据文件
    results = {}
    
    try:
        # 策略统计
        stats_file = os.path.join(results_dir, f"volatile_strategy_statistics_{latest_timestamp}.csv")
        if os.path.exists(stats_file):
            results['statistics'] = pd.read_csv(stats_file, encoding='utf-8-sig')
        
        # 交易记录
        trades_file = os.path.join(results_dir, f"volatile_trading_records_{latest_timestamp}.csv")
        if os.path.exists(trades_file):
            results['trades'] = pd.read_csv(trades_file, encoding='utf-8-sig')
            results['trades']['timestamp'] = pd.to_datetime(results['trades']['timestamp'])
        
        # 每15分钟记录
        records_file = os.path.join(results_dir, f"volatile_records_{latest_timestamp}.csv")
        if os.path.exists(records_file):
            results['records'] = pd.read_csv(records_file, encoding='utf-8-sig')
            results['records']['timestamp'] = pd.to_datetime(results['records']['timestamp'])
        
        # 收益曲线
        equity_file = os.path.join(results_dir, f"volatile_equity_curve_{latest_timestamp}.csv")
        if os.path.exists(equity_file):
            results['equity_curve'] = pd.read_csv(equity_file, encoding='utf-8-sig')
            results['equity_curve']['timestamp'] = pd.to_datetime(results['equity_curve']['timestamp'])
        
        # 仓位变化
        position_file = os.path.join(results_dir, f"volatile_position_changes_{latest_timestamp}.csv")
        if os.path.exists(position_file):
            results['position_changes'] = pd.read_csv(position_file, encoding='utf-8-sig')
            results['position_changes']['timestamp'] = pd.to_datetime(results['position_changes']['timestamp'])
        
        # 当前持仓
        current_pos_file = os.path.join(results_dir, f"volatile_current_positions_{latest_timestamp}.csv")
        if os.path.exists(current_pos_file):
            results['current_positions'] = pd.read_csv(current_pos_file, encoding='utf-8-sig')
            results['current_positions']['buy_time'] = pd.to_datetime(results['current_positions']['buy_time'])
        
        # 市场分析数据（Excel文件）
        analysis_file = os.path.join(results_dir, f"volatile_market_analysis_{latest_timestamp}.xlsx")
        if os.path.exists(analysis_file):
            results['adx_analysis'] = pd.read_excel(analysis_file, sheet_name='ADX分析')
            results['phase_analysis'] = pd.read_excel(analysis_file, sheet_name='市场阶段分析')
        
        # 将时间戳信息保存到结果中
        results['timestamp'] = latest_timestamp
        
        # 从statistics中提取初始资金信息
        if 'statistics' in results:
            stats_df = results['statistics']
            # 查找初始资金行
            initial_cash_row = stats_df[stats_df['指标'] == '初始资金']
            if not initial_cash_row.empty:
                # 提取数值并转换为float（去掉逗号）
                initial_cash_str = initial_cash_row['数值'].iloc[0]
                results['initial_cash'] = float(initial_cash_str.replace(',', ''))
            else:
                results['initial_cash'] = 100000  # 默认值
        else:
            results['initial_cash'] = 100000  # 默认值
        
        print(f"✅ 成功加载 {len(results)-2} 个数据文件")
        return results
        
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        return None

def analyze_volatile_trading_performance(results: dict):
    """分析震荡市场策略交易表现"""
    if 'trades' not in results:
        print("❌ 缺少交易记录数据")
        return
    
    trades = results['trades']
    
    print("\n" + "="*60)
    print("📊 震荡市场策略交易表现分析")
    print("="*60)
    
    # 基本统计
    buy_trades = trades[trades['action'] == 'buy']
    sell_trades = trades[trades['action'] == 'sell']
    
    print(f"总交易次数: {len(trades)}")
    print(f"买入次数: {len(buy_trades)}")
    print(f"卖出次数: {len(sell_trades)}")
    
    # 交易金额统计
    if len(buy_trades) > 0:
        print(f"\n💰 买入交易分析:")
        print(f"平均买入金额: {buy_trades['value'].mean():,.2f}")
        print(f"最大买入金额: {buy_trades['value'].max():,.2f}")
        print(f"最小买入金额: {buy_trades['value'].min():,.2f}")
        print(f"总买入金额: {buy_trades['value'].sum():,.2f}")
    
    if len(sell_trades) > 0:
        print(f"\n💸 卖出交易分析:")
        print(f"平均卖出金额: {sell_trades['value'].mean():,.2f}")
        print(f"最大卖出金额: {sell_trades['value'].max():,.2f}")
        print(f"最小卖出金额: {sell_trades['value'].min():,.2f}")
        print(f"总卖出金额: {sell_trades['value'].sum():,.2f}")
    
    # 交易原因分析（震荡市场策略特有）
    print(f"\n📋 交易原因分析:")
    reason_counts = trades['reason'].value_counts()
    for reason, count in reason_counts.items():
        print(f"{reason}: {count}次")
    
    # ADX等级下的交易分析
    print(f"\n📊 ADX等级交易分析:")
    adx_trades = {}
    for _, trade in trades.iterrows():
        reason = trade['reason']
        if '(medium)' in reason:
            adx_level = 'medium'
        elif '(strong)' in reason:
            adx_level = 'strong'
        else:
            adx_level = 'other'
        
        if adx_level not in adx_trades:
            adx_trades[adx_level] = 0
        adx_trades[adx_level] += 1
    
    for level, count in adx_trades.items():
        print(f"{level}趋势交易: {count}次")

def analyze_market_phases(results: dict):
    """分析市场阶段和ADX分布"""
    if 'records' not in results:
        print("❌ 缺少记录数据")
        return
    
    records = results['records']
    
    print("\n" + "="*60)
    print("📈 市场阶段和ADX分析")
    print("="*60)
    
    # 市场阶段分析
    phase_counts = records['market_phase'].value_counts()
    total_points = len(records)
    
    print("🔄 市场阶段分布:")
    for phase, count in phase_counts.items():
        percentage = count / total_points * 100
        print(f"{phase}: {count}个时间点 ({percentage:.1f}%)")
    
    # ADX等级分析
    if 'adx_level' in records.columns:
        adx_counts = records['adx_level'].value_counts()
        print(f"\n📊 ADX等级分布:")
        for level, count in adx_counts.items():
            percentage = count / total_points * 100
            print(f"{level}: {count}个时间点 ({percentage:.1f}%)")
    
    # 震荡阶段的ADX分布
    oscillation_records = records[records['market_phase'] == 'oscillation']
    if len(oscillation_records) > 0 and 'adx_level' in oscillation_records.columns:
        print(f"\n🌊 震荡阶段ADX分布:")
        osc_adx_counts = oscillation_records['adx_level'].value_counts()
        osc_total = len(oscillation_records)
        for level, count in osc_adx_counts.items():
            percentage = count / osc_total * 100
            print(f"{level}: {count}个时间点 ({percentage:.1f}%)")

def plot_volatile_equity_curve(results: dict, save_path: str = None):
    """绘制震荡市场策略收益曲线"""
    if 'equity_curve' not in results:
        print("❌ 缺少收益曲线数据")
        return
    
    equity_curve = results['equity_curve']
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12))
    
    # 总资产曲线与买入持有BTC对比
    ax1_twin = ax1.twinx()
    
    # 绘制策略总资产曲线（主坐标轴）
    line1 = ax1.plot(equity_curve['timestamp'], equity_curve['total_value'], 'b-', linewidth=2, label='震荡策略总资产')
    ax1.set_ylabel('策略总资产 (USDT)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # 计算买入持有BTC的净值曲线
    if 'records' in results and 'price' in results['records'].columns:
        records = results['records']
        
        # 获取初始资金和初始BTC价格
        initial_cash = results.get('initial_cash', 100000)
        initial_price = records['price'].iloc[0]
        
        # 计算如果全部买入BTC的数量
        btc_amount_hold = initial_cash / initial_price
        
        # 计算买入持有的净值曲线
        hold_values = records['price'] * btc_amount_hold
        
        # 绘制买入持有BTC净值曲线（副坐标轴）
        line2 = ax1_twin.plot(records['timestamp'], hold_values, 'r--', linewidth=2, label='买入持有BTC', alpha=0.7)
        ax1_twin.set_ylabel('买入持有BTC净值 (USDT)', fontsize=12, color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
    else:
        ax1.legend()
    
    ax1.set_title('震荡策略总资产变化曲线与买入持有BTC对比', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 累计收益率曲线
    ax2.plot(equity_curve['timestamp'], equity_curve['cumulative_return'], 'g-', linewidth=2, label='累计收益率')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_title('累计收益率曲线', fontsize=14, fontweight='bold')
    ax2.set_ylabel('累计收益率 (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # BTC仓位比例曲线
    ax3.plot(equity_curve['timestamp'], equity_curve['btc_ratio'] * 100, 'orange', linewidth=2, label='BTC仓位比例')
    ax3.set_title('BTC仓位比例变化', fontsize=14, fontweight='bold')
    ax3.set_ylabel('仓位比例 (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 市场阶段和ADX分布（如果有记录数据）
    if 'records' in results:
        records = results['records']
        
        # 创建市场阶段的颜色映射
        phase_colors = {'oscillation': 'green', 'bullish': 'red', 'bearish': 'blue'}
        
        # 绘制市场阶段背景
        for phase in records['market_phase'].unique():
            phase_data = records[records['market_phase'] == phase]
            ax4.scatter(phase_data['timestamp'], phase_data['adx'], 
                       c=phase_colors.get(phase, 'gray'), alpha=0.6, s=10, label=f'{phase}阶段')
        
        ax4.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='ADX=20')
        ax4.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='ADX=40')
        ax4.set_title('ADX指标与市场阶段分布', fontsize=14, fontweight='bold')
        ax4.set_ylabel('ADX值', fontsize=12)
        ax4.set_xlabel('时间', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 震荡策略收益曲线图已保存到: {save_path}")
    
    plt.show()

def plot_volatile_trading_analysis(results: dict, save_path: str = None):
    """绘制震荡市场策略交易分析图"""
    if 'trades' not in results:
        print("❌ 缺少交易记录数据")
        return
    
    trades = results['trades']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 交易次数按月统计
    trades['month'] = trades['timestamp'].dt.to_period('M')
    monthly_trades = trades.groupby('month').size()
    monthly_trades.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('每月交易次数', fontsize=12, fontweight='bold')
    ax1.set_ylabel('交易次数')
    ax1.tick_params(axis='x', rotation=45)
    
    # 买入卖出比例
    action_counts = trades['action'].value_counts()
    ax2.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    ax2.set_title('买入卖出比例', fontsize=12, fontweight='bold')
    
    # ADX等级交易分布
    adx_trades = {}
    for _, trade in trades.iterrows():
        reason = trade['reason']
        if '(medium)' in reason:
            adx_level = 'medium'
        elif '(strong)' in reason:
            adx_level = 'strong'
        else:
            adx_level = 'other'
        
        if adx_level not in adx_trades:
            adx_trades[adx_level] = 0
        adx_trades[adx_level] += 1
    
    ax3.bar(adx_trades.keys(), adx_trades.values(), color=['lightblue', 'orange', 'lightgray'])
    ax3.set_title('ADX等级交易分布', fontsize=12, fontweight='bold')
    ax3.set_ylabel('交易次数')
    
    # 交易原因统计（震荡策略特有）
    reason_counts = trades['reason'].value_counts().head(10)  # 只显示前10个
    reason_counts.plot(kind='bar', ax=ax4, color='lightpink')
    ax4.set_title('交易原因统计 (前10)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('次数')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 震荡策略交易分析图已保存到: {save_path}")
    
    plt.show()

def plot_market_phase_analysis(results: dict, save_path: str = None):
    """绘制市场阶段分析图"""
    if 'records' not in results:
        print("❌ 缺少记录数据")
        return
    
    records = results['records']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 市场阶段分布饼图
    phase_counts = records['market_phase'].value_counts()
    colors = {'oscillation': 'lightgreen', 'bullish': 'lightcoral', 'bearish': 'lightblue'}
    phase_colors = [colors.get(phase, 'gray') for phase in phase_counts.index]
    
    ax1.pie(phase_counts.values, labels=phase_counts.index, autopct='%1.1f%%', colors=phase_colors)
    ax1.set_title('市场阶段分布', fontsize=12, fontweight='bold')
    
    # ADX等级分布
    if 'adx_level' in records.columns:
        adx_counts = records['adx_level'].value_counts()
        ax2.bar(adx_counts.index, adx_counts.values, color=['lightblue', 'orange', 'red'])
        ax2.set_title('ADX等级分布', fontsize=12, fontweight='bold')
        ax2.set_ylabel('时间点数量')
    
    # 震荡阶段的ADX分布
    oscillation_records = records[records['market_phase'] == 'oscillation']
    if len(oscillation_records) > 0 and 'adx_level' in oscillation_records.columns:
        osc_adx_counts = oscillation_records['adx_level'].value_counts()
        ax3.bar(osc_adx_counts.index, osc_adx_counts.values, color=['lightblue', 'orange', 'red'])
        ax3.set_title('震荡阶段ADX等级分布', fontsize=12, fontweight='bold')
        ax3.set_ylabel('时间点数量')
    
    # ADX值随时间变化
    if 'adx' in records.columns:
        ax4.plot(records['timestamp'], records['adx'], 'purple', alpha=0.7, linewidth=1)
        ax4.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='ADX=20')
        ax4.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='ADX=40')
        ax4.set_title('ADX指标时间序列', fontsize=12, fontweight='bold')
        ax4.set_ylabel('ADX值')
        ax4.set_xlabel('时间')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 市场阶段分析图已保存到: {save_path}")
    
    plt.show()

def generate_volatile_analysis_report(results: dict, output_file: str = "volatile_analysis_report.txt"):
    """生成震荡市场策略详细分析报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("BTC震荡市场交易策略详细分析报告\n")
        f.write("="*60 + "\n\n")
        
        # 基本统计
        if 'statistics' in results:
            f.write("📊 策略基本统计\n")
            f.write("-"*30 + "\n")
            stats = results['statistics']
            for _, row in stats.iterrows():
                f.write(f"{row['指标']}: {row['数值']}\n")
            f.write("\n")
        
        # 交易分析
        if 'trades' in results:
            trades = results['trades']
            f.write("💰 交易详细分析\n")
            f.write("-"*30 + "\n")
            
            buy_trades = trades[trades['action'] == 'buy']
            sell_trades = trades[trades['action'] == 'sell']
            
            f.write(f"买入交易统计:\n")
            f.write(f"  - 次数: {len(buy_trades)}\n")
            if len(buy_trades) > 0:
                f.write(f"  - 平均金额: {buy_trades['value'].mean():,.2f}\n")
                f.write(f"  - 总金额: {buy_trades['value'].sum():,.2f}\n")
            f.write(f"\n")
            
            f.write(f"卖出交易统计:\n")
            f.write(f"  - 次数: {len(sell_trades)}\n")
            if len(sell_trades) > 0:
                f.write(f"  - 平均金额: {sell_trades['value'].mean():,.2f}\n")
                f.write(f"  - 总金额: {sell_trades['value'].sum():,.2f}\n")
            f.write(f"\n")
            
            f.write("交易原因分析:\n")
            reason_counts = trades['reason'].value_counts()
            for reason, count in reason_counts.items():
                f.write(f"  - {reason}: {count}次\n")
            f.write("\n")
            
            # ADX等级交易分析
            f.write("ADX等级交易分析:\n")
            adx_trades = {}
            for _, trade in trades.iterrows():
                reason = trade['reason']
                if '(medium)' in reason:
                    adx_level = 'medium'
                elif '(strong)' in reason:
                    adx_level = 'strong'
                else:
                    adx_level = 'other'
                
                if adx_level not in adx_trades:
                    adx_trades[adx_level] = 0
                adx_trades[adx_level] += 1
            
            for level, count in adx_trades.items():
                f.write(f"  - {level}趋势交易: {count}次\n")
            f.write("\n")
        
        # 市场阶段分析
        if 'records' in results:
            records = results['records']
            f.write("🔄 市场阶段分析\n")
            f.write("-"*30 + "\n")
            
            phase_counts = records['market_phase'].value_counts()
            total_points = len(records)
            
            f.write("市场阶段分布:\n")
            for phase, count in phase_counts.items():
                percentage = count / total_points * 100
                f.write(f"  - {phase}: {count}个时间点 ({percentage:.1f}%)\n")
            f.write("\n")
            
            # ADX等级分析
            if 'adx_level' in records.columns:
                adx_counts = records['adx_level'].value_counts()
                f.write("ADX等级分布:\n")
                for level, count in adx_counts.items():
                    percentage = count / total_points * 100
                    f.write(f"  - {level}: {count}个时间点 ({percentage:.1f}%)\n")
                f.write("\n")
        
        # 当前持仓分析
        if 'current_positions' in results:
            positions = results['current_positions']
            f.write("📊 当前持仓分析\n")
            f.write("-"*30 + "\n")
            f.write(f"持仓笔数: {len(positions)}\n")
            if len(positions) > 0:
                f.write(f"总持仓金额: {positions['current_value'].sum():,.2f}\n")
                f.write(f"未实现盈亏: {positions['unrealized_pnl'].sum():,.2f}\n")
                f.write(f"平均盈亏比例: {positions['pnl_ratio'].mean():.2%}\n")
            f.write("\n")
    
    print(f"📝 震荡策略详细分析报告已保存到: {output_file}")

def main():
    """主函数"""
    print("🌊 震荡市场策略回测结果分析工具")
    print("="*60)
    
    # 加载最新结果
    results = load_latest_volatile_results(r".\volatile_strategy_results")
    if not results:
        return
    
    # 分析交易表现
    analyze_volatile_trading_performance(results)
    
    # 分析市场阶段
    analyze_market_phases(results)
    
    # 获取时间戳用于文件命名
    timestamp = results.get('timestamp', 'unknown')
    
    # 确保输出目录存在
    output_dir = r".\analyze_volatile_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制收益曲线
    print("\n📈 生成震荡策略收益曲线图...")
    plot_volatile_equity_curve(results, rf".\analyze_volatile_results\volatile_equity_curve_analysis_{timestamp}.png")
    
    # 绘制交易分析图
    print("\n📊 生成震荡策略交易分析图...")
    plot_volatile_trading_analysis(results, rf".\analyze_volatile_results\volatile_trading_analysis_{timestamp}.png")
    
    # 绘制市场阶段分析图
    print("\n🔄 生成市场阶段分析图...")
    plot_market_phase_analysis(results, rf".\analyze_volatile_results\volatile_market_phase_analysis_{timestamp}.png")
    
    # 生成详细报告
    print("\n📝 生成震荡策略详细分析报告...")
    generate_volatile_analysis_report(results, rf".\analyze_volatile_results\volatile_analysis_report_{timestamp}.txt")
    
    print("\n✅ 震荡市场策略分析完成！")

if __name__ == "__main__":
    main() 