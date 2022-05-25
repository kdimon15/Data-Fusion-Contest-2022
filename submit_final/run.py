from calendar import week
import sys
import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostRanker
import gc

def make_comb_features(dataframe):
    for x in range(24):
        dataframe[f'diff_hours_{x}'] = dataframe[f'click_h_{x}'] - dataframe[f'trans_h_{x}']

def main():
    with open("feature_list.pkl", "rb") as fp:
        feature_list = pickle.load(fp)
    
    data, output_path = sys.argv[1:]

    clickstream = pd.read_csv(f'{data}/clickstream.csv')
    clickstream['timestamp'] = pd.to_datetime(clickstream['timestamp'])
    clickstream['date'] = clickstream['timestamp'].dt.date.astype('category')
    clickstream['hour'] = clickstream['timestamp'].dt.hour.astype('category')
    clickstream['weekday'] = clickstream['timestamp'].dt.dayofweek.astype('category')

    clickstream_embed = clickstream.pivot_table(index = 'user_id', 
                            values=['timestamp'],
                            columns=['cat_id'],
                            aggfunc=['count']).fillna(0)
    clickstream_embed2 = clickstream.pivot_table(index = 'user_id', 
                                values=['timestamp'],
                                columns=['date'],
                                aggfunc=['count']).fillna(0)
    clickstream_embed3 = clickstream.pivot_table(index = 'user_id', 
                                values=['new_uid'],
                                aggfunc=['nunique']).fillna(0)
    clickstream_embed4 = clickstream.groupby('user_id')['timestamp'].apply(lambda x: np.max(x) - np.min(x)).dt.days.astype('int16').to_frame()
    clickstream_embed5 = clickstream.pivot_table(index = 'user_id', 
                                values=['timestamp'],
                                columns=['weekday','hour'],
                                aggfunc=['count']).fillna(0)
    clickstream_embed.columns = [f'rtk-{str(i[0])}-{str(i[2])}' for i in clickstream_embed.columns]
    clickstream_embed2.columns = [f'rtk-{str(i[0])}-{str(i[2])}' for i in clickstream_embed2.columns]
    clickstream_embed3.columns = [f'rtk-{str(i[0])}-{str(i[1])}' for i in clickstream_embed3.columns]
    clickstream_embed4.columns = [f'rtk-max_date_diff' for i in clickstream_embed4.columns]
    clickstream_embed5.columns = [f'rtk-{str(i[0])}-weekday-{str(i[2])}-nhour-{str(i[3])}' for i in clickstream_embed5.columns]
    clickstream_embed = clickstream_embed.merge(clickstream_embed2, left_on='user_id', right_index=True).merge(
                                                clickstream_embed3, left_on='user_id', right_index=True).merge(
                                                clickstream_embed4, left_on='user_id', right_index=True).merge(
                                                clickstream_embed5, left_on='user_id', right_index=True)

    clickstream_embed.loc[0] = np.empty(len(clickstream_embed.columns)) # добавляем user_id = 0

    clickstream['hour'] = clickstream['timestamp'].dt.hour
    cl_sv = pd.pivot_table(clickstream, index='user_id', columns='hour', values = 'timestamp', aggfunc = 'count').fillna(0)
    cl_sv['summs'] = cl_sv.sum(axis=1)
    for i in cl_sv.columns[:-1]:
        cl_sv[i] /= cl_sv['summs']
    cl_sv.columns = ['click_h_'+ str(i) for i in cl_sv.columns]

    not_found = []
    for x in range(24):
        if f'click_h_{x}' not in cl_sv.columns:
            not_found.append(f'click_h_{x}')
    cl_sv = cl_sv.assign(**dict.fromkeys(not_found, 0))

    del clickstream, clickstream_embed2, clickstream_embed3, clickstream_embed4, clickstream_embed5
    gc.collect()

    transactions = pd.read_csv(f'{data}/transactions.csv')
    transactions['transaction_dttm'] = pd.to_datetime(transactions['transaction_dttm'])
    transactions['date'] = transactions['transaction_dttm'].dt.date.astype('category')
    transactions['hour'] = transactions['transaction_dttm'].dt.hour.astype('category')
    transactions['weekday'] = transactions['transaction_dttm'].dt.dayofweek.astype('category')

    bankclient_embed = transactions.pivot_table(index = 'user_id', 
                                values=['transaction_amt'],
                                columns=['mcc_code'],
                                aggfunc=['sum', 'mean', 'count']).fillna(0)
    bankclient_embed.columns = [f'{str(i[0])}-{str(i[2])}' for i in bankclient_embed.columns]
    bankclient_embed2 = transactions.pivot_table(index = 'user_id', 
                                values=['transaction_amt'],
                                columns=['currency_rk'],
                                aggfunc=['sum', 'mean', 'count']).fillna(0)
    bankclient_embed2.columns = [f'{str(i[0])}-{str(i[2])}' for i in bankclient_embed2.columns]
    bankclient_embed3 = transactions.pivot_table(index = 'user_id', 
                                values=['transaction_dttm'],
                                columns=['date'],
                                aggfunc=['count']).fillna(0)
    bankclient_embed3.columns = [f'{str(i[0])}-{str(i[2])}' for i in bankclient_embed3.columns]
    bankclient_embed4 = transactions.pivot_table(index = 'user_id', 
                                values=['transaction_dttm'],
                                columns=['weekday','hour'],
                                aggfunc=['count']).fillna(0)
    bankclient_embed4.columns = [f'bnk-{str(i[0])}-weekday-{str(i[2])}-nhour-{str(i[3])}' for i in bankclient_embed4.columns]
    bankclient_embed5 = transactions.groupby('user_id')['transaction_dttm'].apply(lambda x: np.max(x) - np.min(x)).dt.days.astype('int16').to_frame()
    bankclient_embed5.columns = [f'bnk-max_date_diff' for i in bankclient_embed5.columns]
    bankclient_embed = bankclient_embed.merge(bankclient_embed2, left_on='user_id', right_index=True
                                            ).merge(bankclient_embed3, left_on='user_id', right_index=True
                                                    ).merge(bankclient_embed4, left_on='user_id', right_index=True
                                                            ).merge(bankclient_embed5, left_on='user_id', right_index=True)

    tr_sv = pd.pivot_table(transactions, index='user_id', columns='hour', values = 'transaction_amt', aggfunc = 'count').fillna(0)
    tr_sv['summs'] = tr_sv.sum(axis=1)
    for i in tr_sv.columns[:-1]:
        tr_sv[i] /= tr_sv['summs']
    tr_sv.columns = ['trans_h_'+ str(i) for i in tr_sv.columns]

    not_found = []
    for x in range(24):
        if f'trans_h_{x}' not in tr_sv.columns:
            not_found.append(f'trans_h_{x}')
    tr_sv = tr_sv.assign(**dict.fromkeys(not_found, 0))
    del transactions, bankclient_embed2, bankclient_embed3, bankclient_embed4
    gc.collect()


    dtype_clickstream = list()
    for x in clickstream_embed.dtypes.tolist():
        if x=='int64':
            dtype_clickstream.append('int16')
        elif(x=='float64'):
            dtype_clickstream.append('float32')
        else:
            dtype_clickstream.append('object')

    dtype_clickstream = dict(zip(clickstream_embed.columns.tolist(),dtype_clickstream))
    clickstream_embed = clickstream_embed.astype(dtype_clickstream)

    dtype_bankclient = list()
    for x in bankclient_embed.dtypes.tolist():
        if x=='int64':
            dtype_bankclient.append('int16')
        elif(x=='float64'):
            dtype_bankclient.append('float32')
        else:
            dtype_bankclient.append('object')

    dtype_bankclient = dict(zip(bankclient_embed.columns.tolist(),dtype_bankclient))
    bankclient_embed = bankclient_embed.astype(dtype_bankclient)

    list_of_rtk = list(clickstream_embed.index.unique())
    list_of_bank= list(bankclient_embed.index.unique())

    submission = pd.DataFrame(list_of_bank, columns=['bank'])
    submission['rtk'] = submission['bank'].apply(lambda x: list_of_rtk)

    models = []
    for x in range(1, 5):
        model = CatBoostRanker()
        model.load_model(f'catboost_{x}.cbm')
        models.append(model)

    submission_ready = []

    batch_size = 200
    num_of_batches = int((len(list_of_bank))/batch_size)+1

    for i in range(num_of_batches):
        bank_ids = list_of_bank[(i*batch_size):((i+1)*batch_size)]
        if len(bank_ids) != 0:
            part_of_submit = submission[submission['bank'].isin(bank_ids)].explode('rtk')
            part_of_submit = part_of_submit.merge(bankclient_embed, how='left', left_on='bank', right_index=True
                                        ).merge(clickstream_embed, how='left', left_on='rtk', right_index=True
                                                ).merge(cl_sv, how='left', left_on='rtk', right_index=True
                                                    ).merge(tr_sv, how='left', left_on='bank', right_index=True
                                                        ).fillna(0)
            make_comb_features(part_of_submit)

            not_found = []
            for feature in feature_list:
                if feature not in part_of_submit.columns:
                    not_found.append(feature)
                    
            part_of_submit = part_of_submit.assign(**dict.fromkeys(not_found, 0))

            part_of_submit['predicts'] = models[2].predict(part_of_submit[feature_list])
            part_of_submit['predicts'] += models[3].predict(part_of_submit[feature_list])
            part_of_submit['predicts'] /= 2

            part_of_submit['predicts'] += models[0].predict(part_of_submit[feature_list])
            part_of_submit['predicts'] += models[1].predict(part_of_submit[feature_list])

            part_of_submit['predicts'] /= 3
            
            part_of_submit = part_of_submit[['bank', 'rtk', 'predicts']]

            zeros_part = pd.DataFrame(bank_ids, columns=['bank'])
            zeros_part['rtk'] = 0.
            zeros_part['predicts'] = 6

            part_of_submit = pd.concat((part_of_submit, zeros_part))

            part_of_submit = part_of_submit.sort_values(by=['bank', 'predicts'], ascending=False).reset_index(drop=True)
            part_of_submit = part_of_submit.pivot_table(index='bank', values='rtk', aggfunc=list)
            part_of_submit['rtk'] = part_of_submit['rtk'].apply(lambda x: x[:100])
            part_of_submit['bank'] = part_of_submit.index
            part_of_submit = part_of_submit[['bank', 'rtk']]
            submission_ready.extend(part_of_submit.values)

    submission_final = np.array(submission_ready, dtype=object)

    print(submission_final.shape)
    np.savez(output_path, submission_final)

if __name__ == "__main__":
    main()
