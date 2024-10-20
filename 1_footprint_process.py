import pandas as pd
import json
import os
import time
from common import FootprintBar, Tick

BTC_duration = 60 * 5
BTC_scale = 100
BTC_volumePrecision = 2
BTC_pricePrecision = 1

for file in os.listdir('./rawdata'):
    filename = file.split('.')[0]
    if os.path.exists(f'./footprint/{filename}.json'):
        continue
    df = pd.read_csv(f'./rawdata/{file}')
    total_rows = len(df)
    start_time = time.time()
    last_print_time = start_time

    footprintList = {}
    currentFootprint = FootprintBar(BTC_duration, BTC_scale, BTC_volumePrecision, BTC_pricePrecision)
    for row in df.itertuples():
        if row.Index % 10000 == 0:
            current_time = time.time()

            if current_time - last_print_time >= 10:
                progress = (row.Index + 1) / total_rows * 100
                print(f"已处理 {row.Index + 1}/{total_rows} 行，进度: {progress:.2f}%")
                last_print_time = current_time

        tick = Tick()
        tick.timestamp = int(row.time // 1000)
        tick.price = float(row.price)
        tick.size = float(row.qty)
        tick.isBuy = row.is_buyer_maker == False
        if currentFootprint.handle_tick(tick):
            continue
        footprintList[currentFootprint.timestamp] = currentFootprint
        currentFootprint = FootprintBar(BTC_duration, BTC_scale, BTC_volumePrecision, BTC_pricePrecision)
        if currentFootprint.handle_tick(tick) == False:
            print(tick)
            print(currentFootprint)
            raise Exception('handle_tick failed')
    
    if currentFootprint.timestamp != 0:
        footprintList[currentFootprint.timestamp] = currentFootprint

    for footprint in footprintList.values():
        footprint.end_handle_tick()

    with open(f'./footprint/{filename}.json', 'w') as f:
        data = {k: v.to_dict() for k, v in footprintList.items()}
        json.dump(data, f, indent=4)
