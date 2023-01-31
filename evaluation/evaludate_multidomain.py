from tqdm import tqdm
from icecream import ic
import pandas as pd
import os
import ast
import json
import evaluate

ogpath = 'multidomain_predictions/hiporank/mbart_predictions/'
files = [f for f in os.listdir(ogpath) if os.path.isfile(os.path.join(ogpath, f))]

meteor = evaluate.load('meteor')
chrf = evaluate.load('chrf')

res = {}
overall_r = []
overall_m = []
overall_c = []
for f in tqdm(files, desc='{going through files}'):
    if not f.endswith('.json'):
        df = pd.read_csv(f'{ogpath}{f}')
        ic(f)
        grouped_df = df.groupby('domain')
        lang = f.split('_')[3]
        res[lang] = {}
        c_lang = []
        m_lang = []
        r_lang = []
        for domain, group in grouped_df:
            if domain not in ['animals', 'companies']:
                res[lang][domain] = {}
                c_temp, m_temp, r_temp = [], [], []
                for i, row in group.iterrows():
                    p = row['pred_text']
                    r = row['ref_text']

                    r_temp.append(100*ast.literal_eval(row['rouge'])['rouge-l']['f'])
                    c_temp.append(chrf.compute(predictions=[p], references=[r], word_order=2)['score'])
                    m_temp.append(100*meteor.compute(predictions=[p], references=[r])['meteor'])

                ic(domain)
                ic(sum(r_temp)/len(r_temp))
                ic(sum(c_temp)/len(c_temp))
                ic(sum(m_temp)/len(m_temp))

                res[lang][domain]['chrf++'] = sum(c_temp)/len(c_temp)
                res[lang][domain]['meteor'] = sum(m_temp)/len(m_temp)
                res[lang][domain]['rouge'] = sum(r_temp)/len(r_temp)

                r_lang.extend(r_temp)
                c_lang.extend(c_temp)
                m_lang.extend(m_temp)

        ic('average')
        ic(sum(r_lang)/len(r_lang))
        ic(sum(c_lang)/len(c_lang))
        ic(sum(m_lang)/len(m_lang))

        overall_r.extend(r_lang)
        overall_c.extend(c_lang)
        overall_m.extend(m_lang)

        res[lang]['average'] = {}
        res[lang]['average']['chrf++'] = sum(c_lang)/len(c_lang)
        res[lang]['average']['meteor'] = sum(m_lang)/len(m_lang)
        res[lang]['average']['rouge'] = sum(r_lang)/len(r_lang)

ic('overall')
ic(sum(overall_r)/len(overall_r))
ic(sum(overall_c)/len(overall_c))
ic(sum(overall_m)/len(overall_m))

with open('predictions/multidomain_predictions/salience_mbart.json', 'w') as fp:
    json.dump(res, fp)