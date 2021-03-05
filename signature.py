import pandas as pd
import torch
import signatory
from qpython import qconnection

def read_from_db(path, date_str):
    q=qconnection.QConnection(host='127.0.0.1',port=2070,pandas=True)
    q.open()
    q('\l /home/azureuser/ifs/data/2019/12/30/levels')
    symb = q('select distinct sym from levels')
    symbols = '`'.join([x.decode() for x in symb['sym']])
    symbol_list = [x.decode() for x in symb['sym']]

    Date = date_str
    q("\l " + path + date_str)
    q('meta levels')
    df = q('select from levels where sym in `'+symbols)
    q.close()

    return df, symbol_list

def make_signature(path="/home/azureuser/ifs/data/", date_str="2019/12/30", freq='5T'):
    df, symbol_list = read_from_db(path, date_str)
    df = df[df.filter(regex='^time$|price|size|no|orders|^sym$').columns]
    # Columns after dropping
    # print("Num cols after filtering ", len(df.columns))

    df_g = df.groupby([pd.Grouper(key='time', freq=freq), 'sym']).mean().apply(lambda x: (x-min(x))/(max(x)-min(x))).unstack(fill_value=0).stack().fillna(0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_tensor = torch.from_numpy(df_g.values).float().to(device)

    rows = data_tensor.shape[0]
    cols = data_tensor.shape[1]
    batch = rows//len(symbol_list)

    path = data_tensor.reshape([batch, len(symbol_list), cols])
    sig = signatory.signature(path, 3)
    print("Signature ", sig)
    return sig

if __name__ == "__main__":
    make_signature()