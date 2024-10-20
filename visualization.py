import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate
import time

# 示例数据生成函数，生成一个足迹图数据框
# 假设数据有columns: 'timestamp', 'duration', 'price_level', 'bid_size', 'ask_size', 'volume', 'delta'
def generate_sample_data():
    data = []
    num_bars = 50  # 生成 10 个 k-bar
    prev_close = 100  # 设置初始价格
    duration = 60  # 每个 bar 持续 60 秒
    PRICE_LEVEL_HEIGHT = 1  # 设置每个 price level 的高度，模拟数据暂时固定
    for bar_index in range(num_bars):
        num_price_levels = np.random.randint(10, 20)  # 每个 k-bar 有随机数量的价格级别
        high = prev_close + (PRICE_LEVEL_HEIGHT * (np.random.randint(10, 20) - 1))
        low = prev_close - (PRICE_LEVEL_HEIGHT * (np.random.randint(10, 20) - 1))
        open_price = prev_close
        close_price = high if bar_index % 2 == 0 else low
        price_levels = np.arange(low, high + PRICE_LEVEL_HEIGHT, PRICE_LEVEL_HEIGHT)
        timestamp = bar_index * duration
        for price_level in price_levels:
            bid_size = np.random.randint(10, 50)
            ask_size = np.random.randint(10, 50)
            volume = np.random.randint(50, 200)
            delta = np.random.randint(-30, 30)
            data.append([timestamp, duration, price_level, bid_size, ask_size, volume, delta])
        prev_close = close_price  # 模拟相交区域：当前 close 作为下一个 open
    columns = ['timestamp', 'duration', 'price_level', 'bid_size', 'ask_size', 'volume', 'delta']
    return pd.DataFrame(data, columns=columns)

# 交互式可视化足迹图的函数
def create_footprint_chart(df, font_size=12, max_delta=50, max_volume=200):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    padding = 0.05  # 设置上下的 padding，防止最上和最下的 price level 被裁剪
    PRICE_LEVEL_HEIGHT = 1  # 全局变量定义的价格级别高度
    
    # 为每个 k-bar 绘制矩阵式的价格级别数据
    unique_timestamps = df['timestamp'].unique()
    for timestamp in unique_timestamps:
        bar_data = df[df['timestamp'] == timestamp]
        price_levels = bar_data['price_level']
        volumes = bar_data['volume']
        deltas = bar_data['delta']
        price_level_height = PRICE_LEVEL_HEIGHT
        duration = bar_data['duration'].iloc[0]
        
        # 绘制 k-bar 边框和背景填充为浅灰色
        high = price_levels.max()
        low = price_levels.min()
        open_price = price_levels.iloc[0]
        close_price = price_levels.iloc[-1]
        y0 = low
        y1 = high + price_level_height
        
        # 绘制 k-bar 的背景填充（浅灰色）
        fig.add_shape(type="rect",
                      x0=timestamp - duration / 2, x1=timestamp + duration / 2,
                      y0=y0, y1=y1,
                      fillcolor="lightgrey",
                      opacity=0.3)
        
        # 绘制每个价格级别的矩阵，无边框，并可视化 delta 和 volume
        for idx, price_level in enumerate(price_levels):
            delta = deltas.iloc[idx]
            volume = volumes.iloc[idx]
            delta_fill = max(0, min(1, abs(delta) / max_delta))
            volume_fill = max(0, min(1, volume / max_volume))
            x0_delta = timestamp - duration / 2
            x1_delta = x0_delta + delta_fill * (duration / 2)  # delta 可视化在左半部分
            x0_volume = timestamp
            x1_volume = x0_volume + volume_fill * (duration / 2)  # volume 可视化在右半部分
            
            # 填充 delta 部分
            fig.add_shape(type="rect",
                          x0=x0_delta, x1=x1_delta,
                          y0=price_level, 
                          y1=price_level + price_level_height,
                          fillcolor="green" if delta > 0 else "red",
                          opacity=0.6)
            
            # 填充 volume 部分
            fig.add_shape(type="rect",
                          x0=x0_volume, x1=x1_volume,
                          y0=price_level, 
                          y1=price_level + price_level_height,
                          fillcolor="grey",
                          opacity=0.6)
            
            # 绘制矩阵内的文本（bid_size, ask_size, volume, delta），字体大小随缩放调整，位置在 price_level + price_level_height / 2
            fig.add_trace(go.Scattergl(
                x=[timestamp],
                y=[price_level + price_level_height / 2],
                mode='text',
                text=f"{bar_data['bid_size'].iloc[idx]} X {bar_data['ask_size'].iloc[idx]} | {bar_data['volume'].iloc[idx]}  {bar_data['delta'].iloc[idx]}",
                textposition='middle center',
                textfont=dict(color='black', size=font_size),
                name=f'Price Level {price_level}',
                hoverinfo='none'
            ))
        
        # # 绘制 open/close 部分，类似传统 k-bar 的实体部分（确保在最上层），边框为黑色
        # fig.add_shape(type="rect",
        #               x0=timestamp - duration / 2, x1=timestamp + duration / 2,
        #               y0=min(open_price, close_price), y1=max(open_price, close_price),
        #               line=dict(color="black", width=2),
        #               layer='above')
    
    # 设置交互式布局，启用滚轮缩放和拖拽
    fig.update_layout(
        title='Interactive Footprint Chart with Price Levels',
        xaxis_title='Timestamp',
        yaxis_title='Price Level',
        barmode='overlay',
        template='plotly_white',  # 修改为白色背景
        dragmode='pan',  # 启用拖拽模式
        xaxis=dict(
            fixedrange=False,  # 允许缩放
            tickmode='linear',
            dtick=duration,
            showspikes=True,  # 显示 x 轴上的延长线
            spikemode='across',
            spikesnap='cursor',
            spikedash='solid',
            spikethickness=1,
            spikecolor='black',
            showline=True,
            showgrid=True,
            showticklabels=True,
            ticklabelposition='outside top',
            tickfont=dict(size=10),
            title_standoff=15
        ),
        yaxis=dict(
            fixedrange=False,  # 允许缩放
            scaleratio=1,  # 保持比例
            showspikes=True,  # 显示 y 轴上的延长线
            spikemode='across',
            spikesnap='cursor',
            spikedash='solid',
            spikethickness=1,
            spikecolor='black',
            showline=True,
            showgrid=True,
            showticklabels=True,
            ticklabelposition='outside right',
            tickfont=dict(size=10),
            title_standoff=15
        ),
        height=800,  # 设置图表高度，使其占据更大空间
        autosize=True  # 自动调整大小
    )
    return fig

# Dash 应用程序初始化
app = dash.Dash(__name__)
df = generate_sample_data()
initial_figure = create_footprint_chart(df)

app.layout = html.Div([
    dcc.Graph(id='footprint-chart', figure=initial_figure, style={'height': '100vh'}, config={'scrollZoom': True}),
    dcc.Store(id='zoom-level', data=1)
])

# 回调函数监听缩放事件，并动态调整字体大小
# @app.callback(
#     Output('footprint-chart', 'figure'),
#     [Input('footprint-chart', 'relayoutData')],
#     [State('footprint-chart', 'figure')],
#     prevent_initial_call=True
# )
# def update_font_size(relayout_data, figure):
#     if relayout_data is None:
#         raise PreventUpdate
#     try:
#         if 'xaxis.range' in relayout_data:
#             x0, x1 = relayout_data['xaxis.range']
#         elif 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
#             x0 = relayout_data['xaxis.range[0]']
#             x1 = relayout_data['xaxis.range[1]']
#         else:
#             raise PreventUpdate
        
#         zoom_level = abs(x1 - x0)
#         exp_font_size = 600 / zoom_level * 12
#         new_font_size = max(6, min(24, exp_font_size))
#         # 使用 update_traces 来只更新字体大小
#         updated_figure = go.Figure(figure)
#         if exp_font_size < 6:
#             updated_figure.update_traces(visible=False)
#         else:
#             updated_figure.update_traces(visible=True, textfont=dict(size=new_font_size))
#         return updated_figure
#     except KeyError:
#         raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)