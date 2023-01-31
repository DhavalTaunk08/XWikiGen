from tqdm import tqdm
from icecream import ic
import pandas as pd
import os
import evaluate
import ast
from rouge import Rouge

ogpath = 'multilingual_predictions/hiporank/mt5_predictions/'
files = [f for f in os.listdir(ogpath) if os.path.isfile(os.path.join(ogpath, f))]

meteor = evaluate.load('meteor')
chrf = evaluate.load('chrf')
rouge = Rouge()

overall_r = []
overall_c = []
overall_m = []
for f in tqdm(files, desc='{going through files}'):
    if 'books' in f or 'films' in f or 'politicians' in f or 'sportsman' in f or 'writers' in f:
        df = pd.read_csv(f'{ogpath}{f}')
        ic(f)
        c_lang = []
        m_lang = []
        r_lang = []
        for lang in tqdm(['bn_IN', 'en_XX', 'hi_IN', 'mr_IN', 'ml_IN', 'or_IN', 'pa_IN', 'ta_IN']):
            temp = df[df['tgt_lang'] == lang]
            c_temp, m_temp, r_temp = [], [], []
            for i, row in temp.iterrows():
                p = row['pred_text']
                r = row['ref_text']

                r_temp.append(100*rouge.get_scores(p, r)[0]['rouge-l']['f'])
                c_temp.append(chrf.compute(predictions=[p], references=[r], word_order=2)['score'])
                m_temp.append(100*meteor.compute(predictions=[p], references=[r])['meteor'])
            
            ic(lang)
            ic(sum(r_temp)/len(r_temp))
            ic(sum(c_temp)/len(c_temp))
            ic(sum(m_temp)/len(m_temp))

            c_lang.extend(c_temp)
            m_lang.extend(m_temp)
            r_lang.extend(r_temp)

        ic(domain)
        ic('average')
        ic(sum(r_lang)/len(r_lang))
        ic(sum(c_lang)/len(c_lang))
        ic(sum(m_lang)/len(m_lang))

        overall_r.extend(r_lang)
        overall_c.extend(c_lang)
        overall_m.extend(m_lang)

ic('overall average')
ic(sum(overall_r)/len(overall_r))
ic(sum(overall_c)/len(overall_c))
ic(sum(overall_m)/len(overall_m))
