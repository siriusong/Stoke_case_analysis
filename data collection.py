import numpy as np
import pandas as pd
# # 获取股票数据
import baostock as bs
 

# 获取股票数据，并将数据规范化
def get_data(
        code="sh.600000", start_date="2020-06-01", end_date="2021-03-26", frequency="15"
):
    #### 登陆baostock系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print("登录响应代码:" + lg.error_code)
    print("登录响应信息:" + lg.error_msg)
 
    data = bs.query_history_k_data_plus(
        code,
        fields="date,code,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="15",   #15分钟K线数据
        adjustflag="3",   #3：不复权  2：前复权  1：后复权
    )
    data_list = []
    while (data.error_code == "0") & data.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(data.get_row_data())
    # 转为dataframe
    result = pd.DataFrame(data_list, columns=data.fields)
 
    # 警惕：数据全是字符串：<class 'str'>
    # 把字符串转为数值
    result.open = result.open.astype("float64").round(2)
    result.close = result.close.astype("float64").round(2)
    result.high = result.high.astype("float64").round(2)
    result.low = result.low.astype("float64").round(2)
    result.volume = result.volume.astype("int")
    # date列转为时间类型
    result.date = pd.DatetimeIndex(result.date)
 
    # dataframe规范化
    data2 = pd.DataFrame(
        {
            "open": result["open"].values,
            "close": result["close"].values,
            "high": result["high"].values,
            "low": result["low"].values,
            "volume": result["volume"].values,
        },
        index=result["date"].values
    )
 
    #### 登出系统 ####
    bs.logout()
    return data2

# 获取格力电器的日线数据
data1 = get_data(code="sz.000651", start_date="2020-01-01", end_date="2022-04-28")
type(data1)
test=data1['volume']
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.plot(test)


