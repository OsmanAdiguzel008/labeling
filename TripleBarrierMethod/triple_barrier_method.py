# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 00:29:57 2022

@author: oadiguzel
"""

import myml.prado.data_structure as ds
import myml.prado.labeling as lb
import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import datetime as dt
from dateutil.relativedelta import relativedelta
import random 

def getTEvents(gRaw,h): 
    # gRaw should be rate or index. Type of it series with datetime index.
    " SELECT EXTREME VALUES WITH GIVEN THRESHOLD(h)."
    tEvents,sPos,sNeg = [],0,0 
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos,sNeg = max(0,sPos+diff.loc[i]), min(0,sNeg+diff.loc[i]) 
        if sNeg<-h:
            sNeg=0;tEvents.append(i) 
        elif sPos>h:
            sPos=0;tEvents.append(i) 
    return pd.DatetimeIndex(tEvents)

def event_dates_to_frame(close,event_dates):
    df_ = pd.DataFrame(index=close.index,columns=["event"])
    for i in event_dates:
        for j in df_.index:
            if i == j:
                df_.xs(j)['event'] = i
    return df_

def define_upandlow(close, df_):
    df_["close"] = close
    df_["mean"] = df_.close.rolling(24,min_periods=1).mean().bfill()
    df_["std"] = df_.close.rolling(24,min_periods=1).std().bfill()
    df_.dropna(inplace=True)
    upper = df_["close"] + df_["std"]
    lower = df_["close"] - df_["std"]
    return upper, lower
    

def triple_barrier_method(close, events, upper, lower, barriers = [1,1,1],
                          hlen=25, plotting=False, record=False):
    dfo_ = close.to_frame(name="close")
    hlenght = relativedelta(days=hlen)
    dfo_["event"] = events
    dfo_["upper"] = upper
    dfo_["lower"] = lower
    dfg_ = dfo_.copy()
    dfo_["upper"].ffill(limit=hlen,inplace=True)
    dfo_["lower"].ffill(limit=hlen,inplace=True)
    dfo_["event"] = dfo_["event"].ffill(limit=hlen)
    dfo_["output"] = np.nan
    
    sign_at = 0
    for i in range(1,len(dfo_)):
        if sign_at != dfo_.iloc[i]["event"]:
            line_crossed = False
            if (dfo_.iloc[i-1]["close"] < dfo_.iloc[i-1]["upper"]) & (
                                    dfo_.iloc[i]["close"] > dfo_.iloc[i]["upper"]):
                if barrier[0] == 1:
                dfo_.at[dfo_.index[i],"output"] = 1
                line_crossed = True
                sign_at = dfo_.iloc[i]["event"]
            elif (dfo_.iloc[i-1]["close"] > dfo_.iloc[i-1]["lower"]) & (
                                    dfo_.iloc[i]["close"] < dfo_.iloc[i]["lower"]):
                if barrier[1] == 1:
                    dfo_.at[dfo_.index[i],"output"] = -1
                    line_crossed = True
                    sign_at = dfo_.iloc[i]["event"]
            elif (line_crossed == False) & (dfo_.index[i] - relativedelta(
                                        days=hlen) == dfo_.iloc[i]["event"]):
                if barrier[2] == 1:
                    dfo_.at[dfo_.index[i],"output"] = 1
                    line_crossed = True
                    sign_at = dfo_.iloc[i]["event"]
            else:
                if dfo_.iloc[i]["upper"] > 0:
                    dfo_.at[dfo_.index[i],"output"] = 0
                    
                    
    dfg_ = dfg_.merge(dfo_["output"],right_index=True, left_index=True)
    dfg_ = dfg_.iloc[-100:]
    if plotting == True:
        mpl.plot(dfg_.close, color="black")
        for i,r in dfg_.dropna().iterrows():
            rgb = (random.random(), random.random(), random.random())
            mpl.vlines(x=r["event"], ymin=r["lower"], ymax=r["upper"], 
                       linestyle='solid', linewidth=2, color=rgb)
            mpl.vlines(x=r["event"]+hlenght, ymin=r["lower"], ymax=r["upper"], 
                       linestyle='dashed', color=rgb)
            mpl.hlines(y=r["upper"], xmin=r["event"], xmax=r["event"]+hlenght, 
                       linestyle='dashed', linewidth=2, color=rgb)
            mpl.hlines(y=r["lower"], xmin=r["event"], xmax=r["event"]+hlenght, 
                       linestyle='dashed', linewidth=2, color=rgb)
            mpl.hlines(y=r["close"], xmin=r["event"], xmax=r["event"]+hlenght, 
                       linestyle='dashed', linewidth=1, color=rgb)
        dfg_["upper"].ffill(inplace=True)
        dfg_["lower"].ffill(inplace=True)
        for i,r in dfg_.iterrows():
            if r["output"] == 1:
                mpl.plot(i,r["upper"]*1.03, marker="v", color="green")
            if r["output"] == -1:
                mpl.plot(i,r["lower"]*0.97, marker="^", color="red")
            #if r["output"] == 999:
            #    mpl.plot(i,r["upper"]*1.1, marker=".", color="blue")
        t = dt.datetime.today()
        if record == True:
            mpl.savefig(f"graphs/{t.year}{t.month}{t.day}{t.hour}{t.minute}_3BM.png")
    return dfo_["output"]
    
if __name__ == "__main__":
    df = pd.read_csv("cleared_price_daily.csv", index_col="datetime")
    df.index = pd.to_datetime(df.index)
    close = df.basic_materials
    event_dates = getTEvents(close.pct_change().dropna(),0.1)
    events_ = event_dates_to_frame(close,event_dates)
    upper,lower = define_upandlow(close, events_)
    events_ = event_dates_to_frame(close,event_dates)
    output = triple_barrier_method(close, events_, upper, lower, 
                                   hlen=25, plotting=True, record=False)
    
